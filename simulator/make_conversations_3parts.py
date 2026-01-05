#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read JSON/JSONL dataset. TTS in three segments:
- "instruction" -> Voice I
- "input"       -> Voice A
- "output"      -> Voice B
Concatenate into one WAV: conversation_{index}.wav

Install:
  pip install edge-tts pydub

System dependency (pydub needs ffmpeg for mp3 decode):
  - macOS: brew install ffmpeg
  - Ubuntu/Debian: sudo apt-get install ffmpeg
  - Windows: install ffmpeg and add to PATH

Usage:
  python make_conversations_3parts.py --in data.json --out_dir out_wav
  python make_conversations_3parts.py --in data.jsonl --out_dir out_wav --start_index 1

Voices & gaps:
  --voice_i ...  (instruction voice)
  --voice_a ...  (input voice)
  --voice_b ...  (output voice)
  --gap_ia_ms ... silence between instruction and input
  --gap_ab_ms ... silence between input and output

SUPPORTED_VOICES = {
    'Xiaoxiao-晓晓': 'zh-CN-XiaoxiaoNeural',
    'Xiaoyi-晓伊': 'zh-CN-XiaoyiNeural',
    'Yunjian-云健': 'zh-CN-YunjianNeural',
    'Yunxi-云希': 'zh-CN-YunxiNeural',
    'Yunxia-云夏': 'zh-CN-YunxiaNeural',
    'Yunyang-云扬': 'zh-CN-YunyangNeural',
    'liaoning-Xiaobei-晓北辽宁': 'zh-CN-liaoning-XiaobeiNeural',
    'shaanxi-Xiaoni-陕西晓妮': 'zh-CN-shaanxi-XiaoniNeural'


python simulator/make_conversations_3parts.py \
  --in datasets/Sampled_Dataset.json \
  --out_dir simulator/wav_out \
  --voice_i zh-CN-XiaoxiaoNeural \
  --voice_a zh-CN-YunxiNeural \
  --voice_b zh-CN-XiaoyiNeural \
  --gap_ia_ms 500 \
  --gap_ab_ms 800

"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import edge_tts
from pydub import AudioSegment


DEFAULT_VOICE_I = "zh-CN-XiaoxiaoNeural"  # instruction
DEFAULT_VOICE_A = "zh-CN-YunxiNeural"     # input
DEFAULT_VOICE_B = "zh-CN-XiaoyiNeural"    # output


def load_json_or_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() in {".jsonl", ".jl"}:
        items: List[Dict[str, Any]] = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
        return items

    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("Unsupported JSON root type (expect object or array).")


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


async def tts_to_mp3(text: str, voice: str, out_mp3: Path, rate: str = "+0%", volume: str = "+0%") -> bool:
    """
    Returns True if synthesized (non-empty text), False if skipped (empty text).
    """
    text = (text or "").strip()
    if not text:
        return False

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    await communicate.save(str(out_mp3))
    return True


def mp3_to_audiosegment(mp3_path: Path) -> AudioSegment:
    return AudioSegment.from_file(mp3_path, format="mp3")


def normalize_to_dbfs(audio: AudioSegment, target_dbfs: float) -> AudioSegment:
    if audio.duration_seconds <= 0:
        return audio
    # dBFS could be -inf for pure silence; handle that
    if audio.dBFS == float("-inf"):
        return audio
    change = target_dbfs - audio.dBFS
    return audio.apply_gain(change)


def build_3part_conversation_wav(
    seg_i: AudioSegment,
    seg_a: AudioSegment,
    seg_b: AudioSegment,
    out_wav: Path,
    gap_ia_ms: int = 600,
    gap_ab_ms: int = 600,
    normalize_dbfs: float = None,
) -> None:
    gap_ia = AudioSegment.silent(duration=gap_ia_ms)
    gap_ab = AudioSegment.silent(duration=gap_ab_ms)

    combined = seg_i + gap_ia + seg_a + gap_ab + seg_b

    if normalize_dbfs is not None:
        combined = normalize_to_dbfs(combined, normalize_dbfs)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    combined.export(out_wav, format="wav")


async def process_items(
    items: List[Dict[str, Any]],
    out_dir: Path,
    voice_i: str,
    voice_a: str,
    voice_b: str,
    start_index: int,
    gap_ia_ms: int,
    gap_ab_ms: int,
    rate_i: str,
    rate_a: str,
    rate_b: str,
    volume_i: str,
    volume_a: str,
    volume_b: str,
    normalize_dbfs: float = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_tts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items, start=start_index):
        instruction = safe_text(item.get("instruction", ""))
        inp = safe_text(item.get("input", ""))
        out = safe_text(item.get("output", ""))

        mp3_i = tmp_dir / f"{idx:06d}_I.mp3"
        mp3_a = tmp_dir / f"{idx:06d}_A.mp3"
        mp3_b = tmp_dir / f"{idx:06d}_B.mp3"

        ok_i, ok_a, ok_b = await asyncio.gather(
            tts_to_mp3(instruction, voice_i, mp3_i, rate=rate_i, volume=volume_i),
            tts_to_mp3(inp,         voice_a, mp3_a, rate=rate_a, volume=volume_a),
            tts_to_mp3(out,         voice_b, mp3_b, rate=rate_b, volume=volume_b),
        )

        # If any segment text is empty, substitute a short silence so concatenation still works.
        seg_i = mp3_to_audiosegment(mp3_i) if ok_i and mp3_i.exists() else AudioSegment.silent(duration=200)
        seg_a = mp3_to_audiosegment(mp3_a) if ok_a and mp3_a.exists() else AudioSegment.silent(duration=200)
        seg_b = mp3_to_audiosegment(mp3_b) if ok_b and mp3_b.exists() else AudioSegment.silent(duration=200)

        out_wav = out_dir / f"conversation_{idx}.wav"
        build_3part_conversation_wav(
            seg_i=seg_i,
            seg_a=seg_a,
            seg_b=seg_b,
            out_wav=out_wav,
            gap_ia_ms=gap_ia_ms,
            gap_ab_ms=gap_ab_ms,
            normalize_dbfs=normalize_dbfs,
        )

        print(f"[OK] {out_wav.name}")

    # Cleanup intermediate mp3s (comment out if you want to keep them)
    for p in tmp_dir.glob("*.mp3"):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in", dest="in_path", required=True, help="Input .json or .jsonl file path")
    ap.add_argument("--out_dir", default="out_wav", help="Output directory for wav files")
    ap.add_argument("--start_index", type=int, default=1, help="Start index for conversation_{index}.wav")

    # Voices
    ap.add_argument("--voice_i", default=DEFAULT_VOICE_I, help='Voice for "instruction" (I)')
    ap.add_argument("--voice_a", default=DEFAULT_VOICE_A, help='Voice for "input" (A)')
    ap.add_argument("--voice_b", default=DEFAULT_VOICE_B, help='Voice for "output" (B)')

    # Gaps
    ap.add_argument("--gap_ia_ms", type=int, default=600, help="Silence gap between instruction and input (ms)")
    ap.add_argument("--gap_ab_ms", type=int, default=600, help="Silence gap between input and output (ms)")

    # Per-segment rate/volume (optional but often useful)
    ap.add_argument("--rate_i", default="+0%", help="Speech rate for instruction, e.g. +10% or -10%")
    ap.add_argument("--rate_a", default="+0%", help="Speech rate for input, e.g. +10% or -10%")
    ap.add_argument("--rate_b", default="+0%", help="Speech rate for output, e.g. +10% or -10%")

    ap.add_argument("--volume_i", default="+0%", help="Volume for instruction, e.g. +10% or -10%")
    ap.add_argument("--volume_a", default="+0%", help="Volume for input, e.g. +10% or -10%")
    ap.add_argument("--volume_b", default="+0%", help="Volume for output, e.g. +10% or -10%")

    ap.add_argument(
        "--normalize_dbfs",
        type=float,
        default=None,
        help="Normalize final wav to target dBFS (e.g. -16.0). Leave empty to disable.",
    )

    args = ap.parse_args()

    items = load_json_or_jsonl(args.in_path)
    if not items:
        raise SystemExit("No items found in input file.")

    asyncio.run(
        process_items(
            items=items,
            out_dir=Path(args.out_dir),
            voice_i=args.voice_i,
            voice_a=args.voice_a,
            voice_b=args.voice_b,
            start_index=args.start_index,
            gap_ia_ms=args.gap_ia_ms,
            gap_ab_ms=args.gap_ab_ms,
            rate_i=args.rate_i,
            rate_a=args.rate_a,
            rate_b=args.rate_b,
            volume_i=args.volume_i,
            volume_a=args.volume_a,
            volume_b=args.volume_b,
            normalize_dbfs=args.normalize_dbfs,
        )
    )


if __name__ == "__main__":
    main()
