#!/usr/bin/env python3
"""
Batch process a folder of audio files.
Structure:
Input_Dir/
  doctorAAA.wav
  doctorBBB.wav
  conversation_XXX.wav
  conversation_YYY.wav

Usage:
  python batch_folder.py --input ./Case/batch --output ./uploads_batch/
"""
import argparse
import os
import shutil
import json
import uuid
from pathlib import Path
from resemblyzer import VoiceEncoder

# Import from flaskv5
# This will trigger model load, which is expected
from flaskv5 import (
    run_asr_and_align, 
    ensure_mono_16k, 
    build_enroll_embeddings, 
    assign_speaker_per_word_simple, 
    compute_peaks,
    HAVE_RESEMBLYZER,
    UPLOAD_FOLDER as FLASK_UPLOADS
)

def is_audio(f: Path):
    return f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing audio files")
    parser.add_argument("--output", "-o", type=str, default="uploads", help="Output directory (default: uploads)")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_base = Path(args.output).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # 1. Scan files
    # Check for Enrollment subdirectory
    enroll_dir = input_dir / "Enrollment"
    enroll_files = []
    if enroll_dir.exists() and enroll_dir.is_dir():
        enroll_files = sorted([f for f in enroll_dir.iterdir() if f.is_file() and is_audio(f)])
    else:
        print(f"Warning: No 'Enrollment' directory found in {input_dir}. Proceeding without enrollment.")
    
    # Conversation files are in the root of input_dir
    conv_files = sorted([f for f in input_dir.iterdir() if f.is_file() and is_audio(f)])

    print(f"Found {len(enroll_files)} enrollment files in {enroll_dir}: {[f.name for f in enroll_files]}")
    print(f"Found {len(conv_files)} conversation files to process.")

    # 2. Prepare Shared Enrollment Embeddings (Optimization)
    # Instead of re-calculating for every job, let's calculate once if possible.
    # However, flaskv5 functions are designed to take a directory. 
    # To reuse them exactly, we can create a temp directory for enrollment, 
    # OR we can build the embeddings dictionary manually here and pass it to assign_speaker...
    # `assign_speaker_per_word_simple` takes `enroll_embs` dict.
    # `build_enroll_embeddings` takes text path.
    
    # We can create a temporary dummy folder with symlinks/copies of enroll files to use `build_enroll_embeddings`
    # OR just reimplement the simple loop. Reimplementing is cleaner than temp dirs.
    
    enroll_embs = {}
    encoder = None
    if HAVE_RESEMBLYZER and enroll_files:
        print("Building enrollment embeddings...")
        try:
            encoder = VoiceEncoder()
            # Reuse logic from flaskv5 broadly
            speakers = {}
            for f in enroll_files:
                # Load audio
                wav, sr = librosa.load(str(f), sr=16000, mono=True)
                emb = encoder.embed_utterance(wav)
                stem = f.stem
                # Extract speaker name (e.g. doctor楊OO -> doctor楊OO)
                # flaskv5 does: speaker = stem.split("_")[0] if "_" in stem else stem
                # We should probably keep the full name if it's unique, or split.
                # User example: doctor楊OO.wav. 
                # If we split by _, it might break if name has _. 
                # Let's use the logic that works for the user's specific naming convention if possible.
                # Or just use the stem.
                speaker = stem 
                speakers.setdefault(speaker, []).append(emb)
            
            for sp, embs in speakers.items():
                enroll_embs[sp] = np.mean(np.vstack(embs), axis=0)
            print(f"Enrollment complete. Speakers: {list(enroll_embs.keys())}")
        except Exception as e:
            print(f"Enrollment failed: {e}")
            enroll_embs = {}

    # 3. Process Conversations
    for i, conv_file in enumerate(conv_files):
        print(f"\n[{i+1}/{len(conv_files)}] Processing {conv_file.name} ...")
        
        # Create Job ID / Directory
        # Use filename as ID or explicit ID? 
        # Flaskv5 usually uses random UUID, but for batch, file stem is nicer.
        job_id = conv_file.stem
        job_dir = output_base / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # 3.1 Copy/Convert Audio
        cleaned_path = job_dir / "cleaned.wav"
        ensure_mono_16k(str(conv_file), str(cleaned_path))
        
        # 3.2 Copy Enrollment Files to separate 'enroll' folder in job dir
        # This keeps the job self-contained and "compatible" with the View page 
        # (which looks for 'enroll' folder to show "Enrolled Speakers: ...")
        job_enroll_dir = job_dir / "enroll"
        job_enroll_dir.mkdir(exist_ok=True)
        for ef in enroll_files:
            shutil.copy(ef, job_enroll_dir / ef.name)

        # 3.3 Run ASR
        print("  Running ASR...")
        asr_res, aligned = run_asr_and_align(str(cleaned_path))
        
        # 3.4 Assign Speakers
        print("  Assigning Speakers...")
        words_with_speakers = []
        if HAVE_RESEMBLYZER and enroll_embs:
            # Note: flaskv5.assign_... takes (aligned, embs, encoder, wav_path, threshold)
            words_with_speakers = assign_speaker_per_word_simple(
                aligned, enroll_embs, encoder, str(cleaned_path), sim_threshold=0.7
            )
        else:
            words_with_speakers = [ {"start": w['start'], "end": w['end'], "word": w['word'], "assigned": [] } for w in aligned ]

        # 3.5 Peaks
        peaks = compute_peaks(str(cleaned_path))

        # 3.6 Save Result
        result = {"asr_raw": asr_res, "words": words_with_speakers, "peaks": peaks}
        with open(job_dir / "result.json", "w", encoding="utf8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"  Done! Output in {job_dir}")

if __name__ == "__main__":
    import librosa
    import numpy as np
    main()
