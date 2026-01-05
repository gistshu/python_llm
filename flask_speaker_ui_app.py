"""
flask_speaker_ui_app.py

A single-file Flask web UI for local speaker-diarization + ASR + enrollment-based speaker assignment.

What this app provides
- Upload: a main audio file (wav/mp3) and an enrollment ZIP (or multiple wavs) for known speakers
- Server runs (locally) the pipeline: preprocess -> pyannote diarization -> whisper -> whisperx alignment -> per-word embedding assignment using resemblyzer
- Results displayed as an editable transcript (per-word rows). You can change speaker labels, then download JSON

Requirements (save as requirements.txt):
# ---------------- requirements.txt ----------------
# Note: pick a torch+torchaudio wheel appropriate for your system (CPU vs CUDA).
flask
whisperx
openai-whisper
pyannote.audio
resemblyzer
librosa
soundfile
ffmpeg-python
scikit-learn
numpy
# ------------------------------------------------

Installation 
1. python -m venv venv
# mac/linux
2. source venv/bin/activate
# windows (powershell)
# .\\venv\\Scripts\\Activate.ps1
3. pip install -r requirements.txt


Quick run
1. create venv and install dependencies (see requirements.txt). For pyannote you need HUGGINGFACE_TOKEN env var.
2. export HUGGINGFACE_TOKEN="hf_xxx" (or setx on Windows)
export HUGGINGFACE_TOKEN="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"
3. python flask_speaker_ui_app.py
4. open http://127.0.0.1:5000/

Notes
- This is a demo/PoC. For production you should add authentication, job queueing, file size limits, and better error handling.
- Models are large. Use a machine with sufficient RAM and (optionally) GPU.




"""

from flask import Flask, request, redirect, url_for, render_template_string, send_file, jsonify
import os
import tempfile
import shutil
from pathlib import Path
import zipfile
import json
import uuid
import traceback

# audio & ML libs
import librosa
import soundfile as sf
import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ASR & diarization
import whisper
import whisperx
from pyannote.audio import Pipeline
import torch

# CONFIG
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")  # required for pyannote
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.65"))
# DEVICE = "cuda" if whisperx.is_cuda_available() else "cpu"
DEVICE = "cpu"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# ------------ Helper functions (same logic as the previous pipeline) ------------

def ensure_mono_16k(src_path, dst_path):
    y, sr = librosa.load(src_path, sr=16000, mono=True)
    sf.write(dst_path, y, 16000)
    return dst_path

# Build enrollment embeddings: expects enroll_dir with wav files named like alice_1.wav, alice_2.wav
def build_enroll_embeddings(enroll_dir: Path, encoder: VoiceEncoder):
    speakers = {}
    for f in sorted(enroll_dir.glob("*.wav")):
        speaker = f.stem.split("_")[0]
        wav, sr = librosa.load(str(f), sr=16000)
        emb = encoder.embed_utterance(wav)
        speakers.setdefault(speaker, []).append(emb)
    avg = {}
    for sp, embs in speakers.items():
        avg[sp] = np.mean(np.vstack(embs), axis=0)
    return avg

# run pyannote diarization
def run_pyannote_diarization(wav_path):
    if HUGGINGFACE_TOKEN is None:
        raise RuntimeError("HUGGINGFACE_TOKEN not set. Register at Hugging Face and export token.")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN)
    diarization = pipeline(wav_path)
    diar_segs = []
    for segment, track, speaker in diarization.itertracks(yield_label=True):
        diar_segs.append({"start": float(segment.start), "end": float(segment.end), "label": speaker})
    return diar_segs

# ASR + whisperx align
def run_asr_and_align(wav_path: str, model_name="small"):
    # load whisper ASR model
    model = whisper.load_model(model_name, device=DEVICE)

    # 1) run whisper transcription (segment-level)
    result = model.transcribe(wav_path, language="zh", fp16=(DEVICE=="cuda"))

    # 2) try to run whisperx alignment to get word-level timestamps
    try:
        # load acoustic alignment model + metadata for whisperx
        model_a, metadata = whisperx.load_align_model(language_code=result.get('language', 'zh'), device=DEVICE)
        # NOTE: correct order: pass (segments, whisper_model, acoustic_model, metadata, audio_path, device=...)
        aligned_words = whisperx.align(result["segments"], model, model_a, metadata, wav_path, device=DEVICE)
        # aligned_words: list of dicts, each with "word","start","end", ...
        return result, aligned_words

    except Exception as e:
        # 如果 whisperx align 失敗：log 錯誤並回傳 segment-level fallback
        print("whisperx alignment failed:", repr(e))
        print("Falling back to segment-level timestamps from Whisper (coarser).")
        # Build coarse aligned_words from whisper segments:
        coarse = []
        for seg in result.get("segments", []):
            # seg has 'start','end','text'
            text = seg.get("text", "").strip()
            # Split into words and approximate equal time-slices (coarse fallback)
            words = text.split()
            if not words:
                continue
            seg_start, seg_end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
            seg_dur = max(1e-6, seg_end - seg_start)
            per_word = seg_dur / len(words)
            for i, w in enumerate(words):
                w_start = seg_start + i * per_word
                w_end = seg_start + (i + 1) * per_word
                coarse.append({"word": w, "start": w_start, "end": w_end})
        return result, coarse


# assign speaker per word using enrollment embeddings
def assign_speaker_per_word(aligned_words, enroll_embs, encoder, wav_path, sim_threshold=SIM_THRESHOLD):
    y, sr = librosa.load(wav_path, sr=16000)
    out = []
    for w in aligned_words:
        start, end, text = float(w['start']), float(w['end']), w['word']
        pad = 0.02
        s = max(0, start - pad); e = min(len(y)/sr, end + pad)
        start_sample = int(s * sr); end_sample = int(e * sr)
        wav_slice = y[start_sample:end_sample]
        assigned = []
        if len(wav_slice) >= 160:
            try:
                emb = encoder.embed_utterance(wav_slice)
                sims = {sp: float(cosine_similarity(emb.reshape(1,-1), sp_emb.reshape(1,-1))[0,0]) for sp, sp_emb in enroll_embs.items()}
                for sp, sim in sims.items():
                    if sim >= sim_threshold:
                        assigned.append({"speaker": sp, "score": sim})
                if not assigned and sims:
                    top_sp = max(sims.items(), key=lambda x: x[1])
                    assigned.append({"speaker": top_sp[0], "score": float(top_sp[1])})
            except Exception:
                assigned = []
        out.append({"start": start, "end": end, "word": text.strip(), "assigned": assigned})
    return out

# fallback assign by diarization overlap
def assign_by_diarization_words(aligned_words, diar_segs):
    out = []
    for w in aligned_words:
        s, e = float(w['start']), float(w['end'])
        overlapped = []
        for d in diar_segs:
            ov = max(0, min(e, d['end']) - max(s, d['start']))
            if ov > 0:
                overlapped.append({"speaker": d['label'], "overlap": ov})
        if overlapped:
            total = sum(x['overlap'] for x in overlapped)
            assigned = [{"speaker": x['speaker'], "score": x['overlap']/total} for x in overlapped]
        else:
            assigned = []
        out.append({"start": s, "end": e, "word": w['word'], "assigned": assigned})
    return out

# ------------------ Flask routes ------------------

INDEX_HTML = """
<!doctype html>
<title>Local Diarization + ASR (Flask PoC)</title>
<h1>Upload audio and enrollment</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <label>Main audio file, 需辨識的錄音內容 (wav/mp3):</label><br>
  <input type=file name=audio required><br><br>
  <label>Enrollment 說話人聲音檔 (zip of .wav files or multiple wavs):</label><br>
  <input type=file name=enroll multiple><br><br>
  <input type=submit value='Upload & Process'>
</form>
<hr>
<p>Notes: set HUGGINGFACE_TOKEN env var for pyannote. Processing runs locally on the server machine.</p>
"""

RESULT_HTML = """
<!doctype html>
<title>Transcription Result</title>
<h1>Transcription & Speaker Assignment</h1>
<p><a href="/">Upload another</a></p>
<button id="download-json">Download JSON</button>
<table id="transcript" border=1 cellpadding=4 style="border-collapse:collapse">
  <tr><th>#</th><th>Start</th><th>End</th><th>Word</th><th>Assigned Speakers (click to edit)</th></tr>
  {% for w in words %}
  <tr data-idx="{{ loop.index0 }}">
    <td>{{ loop.index }}</td>
    <td>{{'%.2f'|format(w.start)}}</td>
    <td>{{'%.2f'|format(w.end)}}</td>
    <td>{{w.word}}</td>
    <td class="speakers">
      {% for a in w.assigned %}
        <span class="tag">{{a.speaker}} ({{'%.2f'|format(a.score)}})</span>
      {% endfor %}
    </td>
  </tr>
  {% endfor %}
</table>
<script>
// Download edited JSON
document.getElementById('download-json').addEventListener('click', ()=>{
  // collect data from table
  const rows = document.querySelectorAll('#transcript tr[data-idx]');
  const words = [];
  rows.forEach(r=>{
    const idx = r.dataset.idx;
    const start = parseFloat(r.cells[1].innerText);
    const end = parseFloat(r.cells[2].innerText);
    const word = r.cells[3].innerText;
    const tags = Array.from(r.querySelectorAll('.tag')).map(t=>t.innerText);
    words.push({start,end,word,assigned:tags});
  });
  const blob = new Blob([JSON.stringify({words: words}, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'transcript_edited.json'; a.click();
});
</script>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # create a job dir
        job_id = str(uuid.uuid4())
        job_dir = UPLOAD_FOLDER / job_id
        job_dir.mkdir(parents=True)

        # save main audio
        audio_file = request.files.get('audio')
        if not audio_file:
            return "No audio uploaded", 400
        audio_path = job_dir / (secure_filename(audio_file.filename) if hasattr(audio_file, 'filename') else 'audio.wav')
        audio_file.save(str(audio_path))

        # save enroll files (can be multiple wavs or a zip)
        enroll_files = request.files.getlist('enroll')
        enroll_dir = job_dir / 'enroll'
        enroll_dir.mkdir()
        for ef in enroll_files:
            filename = ef.filename
            if filename.lower().endswith('.zip'):
                zpath = job_dir / filename
                ef.save(str(zpath))
                with zipfile.ZipFile(str(zpath), 'r') as z:
                    z.extractall(path=str(enroll_dir))
            else:
                fpath = enroll_dir / filename
                ef.save(str(fpath))

        # convert main audio to 16k mono
        cleaned = job_dir / 'cleaned.wav'
        ensure_mono_16k(str(audio_path), str(cleaned))

        # build enroll embeddings
        encoder = VoiceEncoder()
        enroll_embs = build_enroll_embeddings(enroll_dir, encoder)

        # diarization
        diar_segs = run_pyannote_diarization(str(cleaned))

        # asr & align
        asr_res, aligned_words = run_asr_and_align(str(cleaned))

        # assign by embedding
        words_with_speakers = assign_speaker_per_word(aligned_words, enroll_embs, encoder, str(cleaned))

        # fallback for any with empty assigned
        for i,w in enumerate(words_with_speakers):
            if not w['assigned']:
                fallback = assign_by_diarization_words([aligned_words[i]], diar_segs)[0]
                w['assigned'] = fallback['assigned']

        # save json
        out = {"words": words_with_speakers, "diarization": diar_segs, "asr_raw": asr_res}
        out_path = job_dir / 'result.json'
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # render result page with words
        class W: pass
        words_for_template = []
        for w in words_with_speakers:
            o = W()
            o.start = w['start']; o.end = w['end']; o.word = w['word']; o.assigned = [type('A',(object,),a)() for a in w['assigned']]
            # convert dicts to objects with attributes for Jinja
            for idx, a in enumerate(o.assigned):
                a.speaker = w['assigned'][idx]['speaker']
                a.score = w['assigned'][idx]['score']
            words_for_template.append(o)

        return render_template_string(RESULT_HTML, words=words_for_template)

    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", 500

# small helper to sanitize file names
from werkzeug.utils import secure_filename

if __name__ == '__main__':
    # dev server
    app.run(debug=True)
