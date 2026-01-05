#!/usr/bin/env python3
# app_upload_view.py
"""
Flask app: upload audio + enroll samples -> run local ASR + (optional) speaker assign -> view waveform + transcript -> generate LLM dataset
"""
from flask import Flask, request, jsonify, render_template_string, redirect, url_for, send_from_directory, abort, send_file
from flask_cors import CORS
import os, re, json, uuid, pathlib, math
from pathlib import Path
import tempfile

# audio & ML libs
import numpy as np
import librosa, soundfile as sf
import torch
import whisper
import types

# optional resemblyzer
try:
    from resemblyzer import VoiceEncoder
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_RESEMBLYZER = True
except Exception:
    HAVE_RESEMBLYZER = False

app = Flask(__name__)
CORS(app)

ROOT = Path(__file__).parent.resolve()
UPLOAD_FOLDER = ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

DOCTOR_THRESHOLD = float(os.environ.get("DOCTOR_THRESHOLD", "0.7"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # small/medium/large etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- utilities ----------
def ensure_mono_16k(src_path: str, dst_path: str):
    y, sr = librosa.load(src_path, sr=16000, mono=True)
    sf.write(dst_path, y, 16000)
    return dst_path

def compute_peaks(wav_path: str, n_peaks: int = 1024):
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if len(y) == 0:
            return []
        maxv = float(np.max(np.abs(y))) or 1.0
        y = y / maxv
        block = max(1, len(y) // n_peaks)
        peaks = []
        for i in range(0, len(y), block):
            seg = y[i:i+block]
            if seg.size == 0:
                peaks.append(0.0)
            else:
                idx = np.argmax(np.abs(seg))
                peaks.append(float(seg[idx]))
        if len(peaks) > n_peaks:
            peaks = peaks[:n_peaks]
        elif len(peaks) < n_peaks:
            peaks.extend([0.0] * (n_peaks - len(peaks)))
        return peaks
    except Exception as e:
        return []

# ---------- ASR (whisper) ----------
print(f"Loading whisper model '{WHISPER_MODEL}' on device {DEVICE} ...")
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
print("Whisper loaded.")

def run_asr_and_align(wav_path: str):
    # returns (result, aligned_words)
    result = whisper_model.transcribe(wav_path)
    aligned = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = text.split()
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start + max(0.01, len(text)/10)))
        seg_dur = max(1e-6, seg_end - seg_start)
        per_word = seg_dur / max(1, len(words))
        for i, w in enumerate(words):
            aligned.append({"word": w, "start": seg_start + i*per_word, "end": seg_start + (i+1)*per_word})
    return result, aligned

# ---------- enrollment embeddings ----------
def build_enroll_embeddings(enroll_dir: Path):
    if not HAVE_RESEMBLYZER:
        return {}
    encoder = VoiceEncoder()
    speakers = {}
    for f in sorted(enroll_dir.glob("*")):
        if not f.is_file(): continue
        if f.suffix.lower() not in [".wav", ".mp3", ".flac", ".m4a"]: continue
        wav, sr = librosa.load(str(f), sr=16000, mono=True)
        emb = encoder.embed_utterance(wav)
        stem = f.stem
        speaker = stem.split("_")[0] if "_" in stem else stem
        speakers.setdefault(speaker, []).append(emb)
    # DEBUG: log enrolled speakers
    debug_path = enroll_dir.parent / "debug_enroll.txt"
    try:
        with open(debug_path, "w") as df:
            df.write(f"Found files: {[f.name for f in enroll_dir.glob('*')]}\n")
            df.write(f"Identified speakers: {list(speakers.keys())}\n")
    except: pass

    avg = {}
    for sp, embs in speakers.items():
        avg[sp] = np.mean(np.vstack(embs), axis=0)
    return avg

def assign_speaker_per_word_simple(aligned_words, enroll_embs, encoder, wav_path, sim_threshold=0.7):
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    out = []
    for w in aligned_words:
        s = max(0, float(w['start']) - 0.03)
        e = min(len(y)/sr, float(w['end']) + 0.03)
        seg = y[int(s*sr):int(e*sr)]
        assigned = []
        if HAVE_RESEMBLYZER and len(seg) >= 160 and enroll_embs:
            try:
                emb = encoder.embed_utterance(seg)
                sims = {sp: float(cosine_similarity(emb.reshape(1,-1), sp_emb.reshape(1,-1))[0,0]) for sp, sp_emb in enroll_embs.items()}
                # collect all candidates (frontend decides display)
                for sp, sim in sorted(sims.items(), key=lambda kv: kv[1], reverse=True):
                    assigned.append({"speaker": sp, "score": float(sim)})
            except Exception:
                assigned = []
        out.append({"start": float(w['start']), "end": float(w['end']), "word": w.get('word',''), "assigned": assigned})
    return out

# ---------- doctor detection + pair production (same as earlier) ----------
def assigned_to_doctor_flag(assigned, doctor_threshold=DOCTOR_THRESHOLD):
    best_score = None
    label = "UNKNOWN"
    is_doc = False
    if not assigned:
        return label, False, 0.0
    for a in assigned:
        if isinstance(a, dict):
            sp = str(a.get('speaker','')).strip()
            sc = float(a.get('score',0.0) or 0.0)
        else:
            s = str(a).strip()
            # Try standard "Name (0.85)"
            m = re.match(r"^(.*?)\s*\((0?\.\d+|1(?:\.0+)?)\)\s*$", s)
            if m:
                sp = m.group(1).strip(); sc = float(m.group(2))
            else:
                # Try frontend display format "Name 85%"
                m2 = re.match(r"^(.*?)\s+(\d+)%\s*$", s)
                if m2:
                    sp = m2.group(1).strip(); sc = float(m2.group(2)) / 100.0
                else:
                    sp = s; sc = 0.0
        if not sp: continue
        if best_score is None or sc > best_score:
            best_score = sc; label = sp
        sl = sp.lower()
        if ('doctor' in sl) or ('dr.' in sl) or (sl.startswith('dr ')) or ('醫師' in sl) or ('醫生' in sl):
            if sc >= doctor_threshold:
                is_doc = True
    return label or "UNKNOWN", bool(is_doc), float(best_score or 0.0)

def produce_doctor_patient_pairs(words, output_dir_path, doctor_threshold=DOCTOR_THRESHOLD):
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # per-word
    per_word = []
    for w in words:
        assigned = w.get('assigned', []) or []
        label, is_doc, best = assigned_to_doctor_flag(assigned, doctor_threshold)
        per_word.append({'speaker': label, 'is_doctor': bool(is_doc), 'best_score': float(best), 'text': (w.get('word') or '').strip(), 'start': w.get('start'), 'end': w.get('end')})
    # group into turns
    turns = []
    cur = None
    for pw in per_word:
        if cur is None:
            cur = {'speaker': pw['speaker'], 'is_doctor': pw['is_doctor'], 'text': pw['text'], 'start': pw['start'], 'end': pw['end']}
        else:
            if pw['speaker'] == cur['speaker'] and pw['is_doctor'] == cur['is_doctor']:
                cur['text'] = (cur['text'] + ' ' + pw['text']).strip()
                cur['end'] = pw['end']
            else:
                turns.append(cur)
                cur = {'speaker': pw['speaker'], 'is_doctor': pw['is_doctor'], 'text': pw['text'], 'start': pw['start'], 'end': pw['end']}
    if cur is not None:
        turns.append(cur)
    # produce pairs
    sentence_split_re = re.compile(r'[\n。！？\.\!\?]+')
    pairs = []
    reasons = []
    last_doc = -1
    for i, t in enumerate(turns):
        if t.get('is_doctor'):
            pt_texts = []
            for j in range(last_doc + 1, i):
                if not turns[j].get('is_doctor'):
                    pieces = sentence_split_re.split(turns[j].get('text','') or '')
                    kept = []
                    for s in pieces:
                        s_strip = s.strip()
                        if not s_strip: continue
                        if re.search(r'\b(doctor|dr\.|dr\s|醫師|醫生)[:：\s]?', s_strip, flags=re.I): continue
                        kept.append(s_strip)
                    if kept: pt_texts.append("\n".join(kept))
            input_text = "\n".join(pt_texts).strip()
            output_text = (t.get('text') or '').strip()
            if input_text:
                instruction = ("以中文繁體、專業且溫和的語氣，根據以下病人陳述產生醫師回覆。"
                               "回覆應提供建議與必要的注意事項，但不得進行確診或給出處方；"
                               "若情況嚴重或有緊急徵兆，請建議病人立即就醫。")
                pairs.append({'instruction': instruction, 'input': input_text, 'output': output_text})
            else:
                reasons.append({'doctor_turn_index': i, 'reason': 'no patient text after cleaning between doctors'})
            last_doc = i
    if not any(t.get('is_doctor') for t in turns):
        reasons.append({'reason': 'no doctor turns detected with threshold', 'doctor_threshold': float(doctor_threshold)})
    out_json = output_dir / 'doctor_patient_pairs.json'
    out_jsonl = output_dir / 'llm_dataset.jsonl'
    meta = {'pairs': len(pairs), 'reasons': reasons, 'doctor_threshold': float(doctor_threshold)}
    out_obj = {'_meta': meta, 'pairs': pairs}
    with open(out_json, 'w', encoding='utf8') as f: json.dump(out_obj, f, ensure_ascii=False, indent=2)
    with open(out_jsonl, 'w', encoding='utf8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    return str(out_json), str(out_jsonl), len(pairs), reasons

# ---------- simple templates ----------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Upload Concert - Medical AI Transcription</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary-color: #0f4c75;       /* Deep Royal Blue */
      --secondary-color: #3282b8;     /* Lighter Blue */
      --accent-color: #bbe1fa;        /* Pale Blue */
      --text-color: #1b262c;          /* Dark Grey */
      --bg-color: #f4f7f6;            /* Off-white */
      --card-bg: #ffffff;
      --success-color: #28a745;
      --border-radius: 12px;
      --box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-color);
      color: var(--text-color);
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .container {
      width: 100%;
      max-width: 500px;
      background: var(--card-bg);
      padding: 40px;
      border-radius: var(--border-radius);
      box-shadow: 0 10px 25px rgba(0,0,0,0.1);
      text-align: center;
    }
    h1 {
      color: var(--primary-color);
      font-weight: 600;
      margin-bottom: 8px;
    }
    p.subtitle {
      color: #7f8c8d;
      margin-bottom: 32px;
      font-size: 0.95rem;
    }
    .form-group {
      text-align: left;
      margin-bottom: 20px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 500;
      color: var(--primary-color);
      font-size: 0.9rem;
    }
    input[type="file"] {
      display: block;
      width: 100%;
      padding: 10px;
      padding-left: 0;
      font-size: 0.9rem;
      border: 1px dashed #ccc;
      border-radius: 8px;
      background: #fafafa;
      cursor: pointer;
    }
    input[type="file"]:hover {
      background: #f0f0f0;
    }
    input[name="job_id"] {
      width: 100%;
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: 8px;
      font-size: 1rem;
      box-sizing: border-box;
      transition: border-color 0.2s;
    }
    input[name="job_id"]:focus {
      outline: none;
      border-color: var(--secondary-color);
    }
    button[type="submit"] {
      background-color: var(--primary-color);
      color: white;
      border: none;
      padding: 14px 24px;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      width: 100%;
      transition: background-color 0.2s, transform 0.1s;
      margin-top: 10px;
    }
    button[type="submit"]:hover {
      background-color: #0b3a5b;
    }
    button[type="submit"]:active {
      transform: translateY(1px);
    }
    hr {
      border: 0;
      height: 1px;
      background: #eee;
      margin: 30px 0;
    }
    .footer-note {
      font-size: 0.85rem;
      color: #95a5a6;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Medical Audio AI</h1>
    <p class="subtitle">Upload medical conversation audio for transcription and role identification.</p>
    
    <form method="post" action="/upload" enctype="multipart/form-data">
      <div class="form-group">
        <label>Main Audio File (wav/mp3/m4a)</label>
        <input type="file" name="audio" required>
      </div>
      
      <div class="form-group">
        <label>Enrollment Samples (Optional)</label>
        <input type="file" name="enroll_files" multiple>
        <div style="font-size:0.8rem; color:#888; margin-top:4px">Upload samples named like <code>Doctor_1.wav</code> to identify speakers.</div>
      </div>
      
      <div class="form-group">
        <label>Job ID (Optional)</label>
        <input name="job_id" placeholder="Auto-generate if empty">
      </div>
      
      <button type="submit">Upload & Process</button>
    </form>
    
    <hr>
    <p class="footer-note">Secure local processing using OpenAI Whisper & Resemblyzer.</p>
  </div>
</body>
</html>
"""

VIEW_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transcript Editor - Medical AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary-color: #0f4c75;
      --secondary-color: #3282b8;
      --accent-color: #bbe1fa;
      --text-color: #1b262c;
      --bg-color: #f4f7f6;
      --card-bg: #ffffff;
      --danger-color: #e74c3c;
      --success-color: #27ae60;
      --highlight-bg: #fff3cd;
      --highlight-text: #856404;
      --border-radius: 8px;
    }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-color);
      color: var(--text-color);
      margin: 0;
      padding: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    
    /* Header */
    header {
      background: var(--card-bg);
      padding: 12px 24px;
      border-bottom: 1px solid #ddd;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      z-index: 100;
    }
    header h1 {
      font-size: 1.25rem;
      margin: 0;
      color: var(--primary-color);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    header h1 span.job-id {
      font-size: 0.8em;
      background: #eee;
      padding: 2px 8px;
      border-radius: 4px;
      color: #666;
      font-weight: 400;
    }
    .header-controls {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    #status {
      font-size: 0.9rem;
      margin-right: 12px;
      font-weight: 500;
      display: none; /* hidden by default */
    }

    /* Main Grid */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 20px;
      gap: 20px;
    }

    /* Waveform Panel */
    .wave-panel {
      background: var(--card-bg);
      border-radius: var(--border-radius);
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
      padding: 20px;
      flex-shrink: 0;
    }
    .wave-controls {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 12px;
    }
    button.icon-btn {
      background: #eee;
      border: none;
      border-radius: 6px;
      width: 36px;
      height: 36px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s;
      font-size: 1.2rem;
    }
    button.icon-btn:hover {
      background: #ddd;
    }
    button.primary-btn {
      background: var(--primary-color);
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 0.2s;
    }
    button.primary-btn:hover {
      background: #0b3a5b;
    }
    button.secondary-btn {
      background: white;
      border: 1px solid #ccc;
      color: #333;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
    }
    button.secondary-btn:hover {
      background: #f9f9f9;
    }

    /* Transcript Panel */
    .transcript-panel {
      background: var(--card-bg);
      border-radius: var(--border-radius);
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
      flex: 1;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .transcript-header {
      padding: 12px 20px;
      border-bottom: 1px solid #eee;
      font-weight: 600;
      color: #7f8c8d;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      background: #fafafa;
      display: grid;
      grid-template-columns: 60px 80px 1fr 180px; /* ID Time Text Speaker */
      gap: 16px;
    }
    .transcript-body {
      overflow-y: auto;
      flex: 1;
      padding: 0;
    }
    
    /* Rows */
    .t-row {
      display: grid;
      grid-template-columns: 60px 80px 1fr 180px;
      gap: 16px;
      padding: 10px 20px;
      border-bottom: 1px solid #f0f0f0;
      align-items: baseline;
      transition: background 0.15s;
      cursor: pointer;
    }
    .t-row:hover {
      background: #f8f9fa;
    }
    .t-row.active {
      background-color: #e3f2fd; /* Light Blue highlight */
      border-left: 4px solid var(--secondary-color);
      padding-left: 16px; /* adjust for border */
    }
    
    .col-id { color: #999; font-size: 0.85rem; }
    .col-time { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #666; }
    .col-text { 
      font-size: 1rem; 
      line-height: 1.5; 
      outline: none;
      border-radius: 4px;
      padding: 2px 4px;
    }
    .col-text:focus {
      background: white;
      box-shadow: 0 0 0 2px var(--secondary-color);
    }
    .active .col-text {
      font-weight: 500;
      color: #000;
    }
    
    /* Assigned Speaker Badge */
    .col-speaker {
      font-size: 0.8rem;
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }
    .badge {
      padding: 2px 8px;
      border-radius: 12px;
      background: #eee;
      color: #555;
      font-weight: 500;
    }
    .badge.doctor {
      background: #e3f2fd;
      color: #000; /* Black text for > 0.7 */
      border: 1px solid #bbdefb;
    }
    .badge.neutral {
      color: #000; /* Black text for > 0.7 neutral */
    }
    .badge.unknown {
      background: #ffebee;
      color: #c62828;
    }
    
    /* Download Link */
    .download-link {
      display: block;
      margin-top: 10px;
      text-align: right;
      color: var(--primary-color);
      text-decoration: none;
      font-size: 0.9rem;
    }
    .download-link:hover { text-decoration: underline; }

  </style>
</head>
<body>

  <header>
    <h1>MedTranscribe <span class="job-id">{{ job_id }}</span></h1>
    <div class="header-controls">
      <span id="status"></span>
      <button class="secondary-btn" id="saveBtn">Save Draft</button>
      <button class="primary-btn" id="genBtn">Generate & Download</button>
    </div>
  </header>

  <main>
    <!-- Waveform Area -->
    <div class="wave-panel">
      <div id="waveform"></div>
      <div id="timeline"></div>
      
      <div class="wave-controls">
        <button class="icon-btn" id="playBtn" title="Play/Pause">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> 
          <!-- dynamic icon toggle handled in JS -->
        </button>
        <button class="icon-btn" id="zoomOut" title="Zoom Out">-</button>
        <button class="icon-btn" id="zoomIn" title="Zoom In">+</button>
        <div style="margin-left:auto; font-size:0.85rem; color:#666;">
          <strong>Enrolled Speakers:</strong> {{ enroll_list or 'None' }}
        </div>
      </div>
    </div>

    <!-- Transcript Area -->
    <div class="transcript-panel">
      <div class="transcript-header">
        <div>#</div>
        <div>Time</div>
        <div>Transcript (Click text to edit)</div>
        <div>Speaker</div>
      </div>
      <div class="transcript-body" id="transcript-body">
        {% for i,w in enumerate(words) %}
        <div class="t-row" data-idx="{{i}}">
          <div class="col-id">{{i+1}}</div>
          <div class="col-time" data-start="{{w.start}}" data-end="{{w.end}}">{{'%.1f'|format(w.start)}}s</div>
          <div class="col-text" contenteditable="true">{{ w.word }}</div>
          <div class="col-speaker">
            {% for a in w.assigned %}
              {{ a }} 
            {% endfor %}
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
  </main>
  
  <!-- WaveSurfer -->
  <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
  <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/plugin/wavesurfer.timeline.min.js"></script>

  <script>
    const jobId = {{ job_id | tojson }};
    const cleanedUrl = '/uploads/' + jobId + '/cleaned.wav';
    const DOCTOR_THRESHOLD = {{ doctor_threshold }};
    
    // -- Init WaveSurfer --
    // Robust plugin loading
    let plugins = [];
    if (window.WaveSurfer && window.WaveSurfer.timeline && typeof window.WaveSurfer.timeline.create === 'function') {
        plugins.push(WaveSurfer.timeline.create({ container: '#timeline' }));
    }
    
    const ws = WaveSurfer.create({
      container: '#waveform',
      waveColor: '#D1D5DB',
      progressColor: '#3282b8',
      cursorColor: '#0f4c75',
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      plugins: plugins
    });
    
    ws.load(cleanedUrl);
    
    // -- Controls --
    const playBtn = document.getElementById('playBtn');
    ws.on('play', () => { playBtn.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>'; });
    ws.on('pause', () => { playBtn.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>'; });
    
    playBtn.addEventListener('click', () => ws.playPause());
    document.getElementById('zoomIn').addEventListener('click', () => ws.zoom(ws.params.minPxPerSec + 20));
    document.getElementById('zoomOut').addEventListener('click', () => ws.zoom(Math.max(1, ws.params.minPxPerSec - 20)));

    // -- Transcript Logic --
    
    // Parse speaker badges
    function renderBadges() {
      document.querySelectorAll('.col-speaker').forEach(el => {
        if (el.dataset.rendered) return;
        const text = el.innerText.trim();
        
        el.innerText = ''; // clear
        
        if (!text) {
           const span = document.createElement('span');
           span.className = 'badge unknown';
           span.innerText = 'Unknown';
           el.appendChild(span);
           el.dataset.rendered = true;
           return;
        }

        // Split by simple logic (assuming list joined by space or comma in flask)
        // ...
        
        // improved regex to find Speaker (Score) patterns
        const matches = [...text.matchAll(/([^()]+?)\\s*\\((\\d*\\.?\\d+)\\)/g)];
        
        if (matches.length === 0) {
           // fallback plain text if not empty but no score format
           const span = document.createElement('span');
           span.className = 'badge unknown';
           span.innerText = text;
           el.appendChild(span);
           return;
        }

        const validMatches = [];
        matches.forEach(m => {
          const name = m[1].trim();
          const score = parseFloat(m[2]);
          validMatches.push({name, score});
        });

        if (validMatches.length === 0) {
           const span = document.createElement('span');
           span.className = 'badge unknown';
           span.innerText = 'Unknown';
           span.dataset.match = JSON.stringify({speaker:"Unknown", score:0.0}); 
           el.appendChild(span);
           return;
        }

        // Sort by score desc just in case
        validMatches.sort((a,b) => b.score - a.score);
        
        // Take top 1 or all? Usually just need best guess
        // User wants to see data. Let's show all candidates? Or just best?
        // "低於70的資料也請顯示..." implies showing the potential match
        
        validMatches.forEach(vm => {
            const span = document.createElement('span');
            // Store original data
            span.dataset.match = JSON.stringify({speaker: vm.name, score: vm.score});
            
            if (vm.score >= 0.7) {
                // High confidence: Black text (via CSS classes)
                let type = 'neutral';
                if (vm.name.toLowerCase().includes('dr') || vm.name.includes('醫師') || vm.name.includes('醫生')) {
                    if (vm.score >= DOCTOR_THRESHOLD) type = 'doctor';
                }
                span.className = `badge ${type}`;
                span.innerText = `${vm.name} ${Math.round(vm.score*100)}%`;
            } else {
                // Low confidence: Unknown + Score
                span.className = 'badge unknown';
                // Format: Unknown (Score%) or Unknown: Name (Score%)?
                // Adhering to "Unknown字樣" + "分數"
                // Let's do `Unknown (${Math.round(vm.score*100)}%)` 
                // Maybe tooltip the name?
                span.innerText = `Unknown (${Math.round(vm.score*100)}%)`;
                span.title = `Potential match: ${vm.name}`;
            }
            el.appendChild(span);
        });
        el.dataset.rendered = true;
      });
    }
    renderBadges(); // Initial render

    // Highlight & Seek
    const transcriptBody = document.getElementById('transcript-body');
    const rows = Array.from(document.querySelectorAll('.t-row'));
    
    // Create an index for faster lookup
    const timeIndex = rows.map((row, i) => ({
      idx: i,
      start: parseFloat(row.querySelector('.col-time').dataset.start),
      end: parseFloat(row.querySelector('.col-time').dataset.end),
      row: row
    }));

    function setActiveRow(index) {
      // Remove active class
      const current = document.querySelector('.t-row.active');
      if (current) current.classList.remove('active');
      
      if (index >= 0 && index < rows.length) {
        const row = rows[index];
        row.classList.add('active');
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }

    // 1. Highlight on Playback (audioprocess)
    ws.on('audioprocess', (time) => {
      const idx = timeIndex.findIndex(item => time >= item.start && time < item.end);
      if (idx !== -1) setActiveRow(idx);
    });
    
    // 2. Highlight on Seek (Clicking waveform)
    ws.on('seek', (progress) => {
      // progress is 0..1 in v6 seek event? or wait, in v6 'seek' passes progress (0..1).
      // We need time.
      const duration = ws.getDuration();
      if (duration) {
          const time = progress * duration;
          const idx = timeIndex.findIndex(item => time >= item.start && time < item.end);
          if (idx !== -1) setActiveRow(idx);
      }
    });
    
    // 3. Seek on Transcript Row Click
    rows.forEach((row, i) => {
      row.addEventListener('click', (e) => {
        // Prevent seek if clicking editable text (to allow editing)
        if (e.target.classList.contains('col-text')) {
           return; 
        }
        
        const start = timeIndex[i].start;
        // WaveSurfer v6: setCurrentTime(seconds)
        if (ws.setCurrentTime) {
            ws.setCurrentTime(start);
        } else if (ws.seekTo) {
            // fallback if setCurrentTime missing (seekTo takes 0..1)
            const dur = ws.getDuration();
            if (dur) ws.seekTo(start / dur);
        }
        setActiveRow(i);
      });
    });

    // -- API Actions --
    function showStatus(msg, type='success') {
      const el = document.getElementById('status');
      el.innerText = msg;
      el.style.display = 'block';
      el.style.color = type === 'error' ? 'var(--danger-color)' : 'var(--success-color)';
      setTimeout(() => { el.style.display = 'none'; }, 4000);
    }

    document.getElementById('saveBtn').addEventListener('click', async () => {
      showStatus('Saving...', 'neutral');
      
      // Collect data
      const words = [];
      document.querySelectorAll('.t-row').forEach(row => {
          const start = parseFloat(row.querySelector('.col-time').dataset.start);
          const end = parseFloat(row.querySelector('.col-time').dataset.end);
          const word = row.querySelector('.col-text').innerText.trim();
          
          let assigned = [];
          // Retrieve structured data from badges
          row.querySelectorAll('.badge').forEach(b => {
             if (b.dataset.match) {
                 try {
                    assigned.push(JSON.parse(b.dataset.match));
                 } catch(e) {
                    // fallback text
                    assigned.push(b.innerText);
                 }
             } else {
                 assigned.push(b.innerText);
             }
          });
          
          words.push({start, end, word, assigned});
      });

      try {
        const res = await fetch('/save_edited/' + jobId, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({words})
        });
        const data = await res.json();
        if (data.ok) showStatus('Saved successfully!');
        else showStatus('Save failed', 'error');
      } catch (e) {
        showStatus('Network error', 'error');
      }
    });

    document.getElementById('genBtn').addEventListener('click', async () => {
       showStatus('Generating dataset...', 'neutral');
       try {
         const res = await fetch('/generate_llm/' + jobId, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({}) });
         const data = await res.json();
         if (data.ok) {
           showStatus(`Generated ${data.pairs} pairs! Ready to download.`);
           setTimeout(() => {
              window.open('/download_llm/' + jobId, '_blank');
           }, 1000);
         } else {
           showStatus('Generation failed: ' + (data.error || 'Unknown'), 'error');
         }
       } catch (e) {
         showStatus('Error generating', 'error');
       }
    });

  </script>
</body>
</html>
"""

# ---------- routes ----------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/upload", methods=["POST"])
def upload():
    audio = request.files.get("audio")
    if not audio:
        return "audio required", 400
    enroll_files = request.files.getlist("enroll_files")
    job_id = request.form.get("job_id") or (uuid.uuid4().hex[:8])
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    # save audio
    audio_fn = audio.filename
    main_path = job_dir / audio_fn
    audio.save(str(main_path))
    # normalize to cleaned.wav
    cleaned = job_dir / "cleaned.wav"
    try:
        ensure_mono_16k(str(main_path), str(cleaned))
    except Exception as e:
        return f"failed resample: {e}", 500
    # save enroll files
    enroll_dir = job_dir / "enroll"
    enroll_dir.mkdir(exist_ok=True)
    for ef in enroll_files:
        if ef and ef.filename:
            ef.save(str(enroll_dir / ef.filename))
    # run ASR
    try:
        asr_res, aligned = run_asr_and_align(str(cleaned))
    except Exception as e:
        return f"ASR failed: {e}", 500
    # build enroll embeddings if possible
    enroll_embs = {}
    encoder = None
    if HAVE_RESEMBLYZER and any(enroll_dir.iterdir()):
        try:
            enroll_embs = build_enroll_embeddings(enroll_dir)
            encoder = VoiceEncoder()
        except Exception as e:
            enroll_embs = {}
    # assign speakers per word
    if HAVE_RESEMBLYZER and enroll_embs:
        words_with_speakers = assign_speaker_per_word_simple(aligned, enroll_embs, encoder, str(cleaned), sim_threshold=0.7)
    else:
        # no enroll: set assigned empty
        words_with_speakers = [ {"start": w['start'], "end": w['end'], "word": w['word'], "assigned": [] } for w in aligned ]
    # peaks
    peaks = compute_peaks(str(cleaned))
    # write result.json
    result = {"asr_raw": asr_res, "words": words_with_speakers, "peaks": peaks}
    with open(job_dir / "result.json", "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # redirect to view
    return redirect(url_for("view_job", job_id=job_id))

@app.route("/view/<job_id>")
def view_job(job_id):
    job_dir = UPLOAD_FOLDER / job_id
    if not job_dir.exists():
        return f"job {job_id} not found", 404
    rpath = job_dir / "result_edited.json"
    if not rpath.exists():
        rpath = job_dir / "result.json"

    words = []
    if rpath.exists():
        try:
            with open(rpath, "r", encoding="utf8") as f:
                saved = json.load(f)
            words = saved.get("words", [])
        except Exception:
            words = []
    enroll_list = [p.name for p in (job_dir / "enroll").glob("*")] if (job_dir / "enroll").exists() else []
    # prepare renderable words
    words_render = []
    for w in words:
        assigned_list = []
        for a in (w.get("assigned") or []):
            if isinstance(a, dict):
                assigned_list.append(f"{a.get('speaker')} ({a.get('score'):.2f})")
            else:
                assigned_list.append(str(a))
        words_render.append(types.SimpleNamespace(start=w.get("start",0), end=w.get("end",0), word=w.get("word",""), assigned=assigned_list))

        # words_render.append(pathlib.SimpleNamespace(start=w.get("start",0), end=w.get("end",0), word=w.get("word",""), assigned=assigned_list))
    # return render_template_string(VIEW_HTML, job_id=job_id, words=words_render, words_json=json.dumps([{"start":w.start,"end":w.end,"word":w.word,"assigned":w.assigned} for w in words_render], ensure_ascii=False), enroll_list=", ".join(enroll_list), doctor_threshold=DOCTOR_THRESHOLD)
    return render_template_string(
        VIEW_HTML,
        job_id=job_id,
        words=words_render,
        words_json=json.dumps([{"start": w.start, "end": w.end, "word": w.word, "assigned": w.assigned} for w in words_render], ensure_ascii=False),
        enroll_list=", ".join(enroll_list),
        doctor_threshold=DOCTOR_THRESHOLD,
        enumerate=enumerate
)




@app.route("/uploads/<job_id>/<path:filename>")
def uploaded_file(job_id, filename):
    p = UPLOAD_FOLDER / job_id
    if not p.exists(): abort(404)
    return send_from_directory(str(p), filename)

@app.route("/save_edited/<job_id>", methods=["POST"])
def save_edited(job_id):
    data = request.get_json(silent=True) or {}
    words = data.get("words", [])
    out = UPLOAD_FOLDER / job_id / "result_edited.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8") as f: json.dump({"words": words}, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True})

@app.route("/generate_llm/<job_id>", methods=["POST"])
def generate_llm_route(job_id):
    job_dir = UPLOAD_FOLDER / job_id
    if not job_dir.exists(): return jsonify({"ok": False, "error": "job not found"}), 404
    # load edited if exists, else result.json
    edited = job_dir / "result_edited.json"
    if edited.exists():
        with open(edited, "r", encoding="utf8") as f: saved = json.load(f); words = saved.get("words", [])
    else:
        rpath = job_dir / "result.json"
        if not rpath.exists(): return jsonify({"ok": False, "error": "no result.json"}), 400
        with open(rpath, "r", encoding="utf8") as f: saved = json.load(f); words = saved.get("words", [])
    out_json, out_jsonl, n_pairs, reasons = produce_doctor_patient_pairs(words, job_dir, doctor_threshold=DOCTOR_THRESHOLD)
    return jsonify({"ok": True, "pairs": n_pairs, "json": out_json, "jsonl": out_jsonl, "reasons": reasons})


# Download route for generated dataset
@app.route('/download_llm/<job_id>')
def download_llm(job_id):
    job_dir = UPLOAD_FOLDER / job_id
    p = job_dir / 'llm_dataset.jsonl'
    if not p.exists():
        return 'File not found', 404
    return send_file(str(p), as_attachment=True, download_name=p.name, mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
