#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a text, synthesize THREE separate WAV files using three voices:

DEFAULT_VOICE_I = "zh-CN-XiaoxiaoNeural"  # instruction
DEFAULT_VOICE_A = "zh-CN-YunxiNeural"     # input
DEFAULT_VOICE_B = "zh-CN-XiaoyiNeural"    # output

Outputs (by default):
  voice_I.wav
  voice_A.wav
  voice_B.wav

Install:
  pip install edge-tts pydub

FFmpeg is required for mp3->wav conversion (pydub uses ffmpeg):
  - macOS: brew install ffmpeg
  - Ubuntu/Debian: sudo apt-get install ffmpeg
  - Windows: install ffmpeg and add to PATH

Usage examples:
  python simulator/tts_three_voices.py --text "对痰疟有以下药方：['药方1:用藜芦末半钱，温齑水调下。引吐为好。又方：藜芦、皂荚（炙）各一两，巴豆二址五枚，熬黄，研成末，加蜜和成丸子，如小豆大。每空心服一丸，未发病时服一丸，临发病时又服一丸。宜暂时禁食。"
  python tts_three_voices.py --text_file sample.txt --out_dir out_wav
  python tts_three_voices.py --text "..." --rate "+5%" --volume "+0%"
"""

import argparse
import asyncio
from pathlib import Path

import edge_tts
from pydub import AudioSegment

DEFAULT_VOICE_I = "zh-CN-XiaoxiaoNeural"
DEFAULT_VOICE_A = "zh-CN-YunxiNeural"
DEFAULT_VOICE_B = "zh-CN-XiaoyiNeural"


async def tts_to_mp3(text: str, voice: str, out_mp3: Path, rate: str, volume: str) -> None:
    text = (text or "").strip()
    if not text:
        raise ValueError("Input text is empty.")
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    await communicate.save(str(out_mp3))


def mp3_to_wav(mp3_path: Path, wav_path: Path, normalize_dbfs: float | None = None) -> None:
    audio = AudioSegment.from_file(mp3_path, format="mp3")

    if normalize_dbfs is not None and audio.dBFS != float("-inf"):
        gain = normalize_dbfs - audio.dBFS
        audio = audio.apply_gain(gain)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(wav_path, format="wav")


async def main_async(args) -> None:
    # Load text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text

    text = (text or "").strip()
    if not text:
        raise SystemExit("Error: --text or --text_file must provide non-empty content.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Temp mp3 files
    mp3_i = out_dir / "_tmp_I.mp3"
    mp3_a = out_dir / "_tmp_A.mp3"
    mp3_b = out_dir / "_tmp_B.mp3"

    # Output wav files
    wav_i = out_dir / args.out_i
    wav_a = out_dir / args.out_a
    wav_b = out_dir / args.out_b

    # Synthesize mp3 in parallel
    await asyncio.gather(
        tts_to_mp3(text, args.voice_i, mp3_i, args.rate, args.volume),
        tts_to_mp3(text, args.voice_a, mp3_a, args.rate, args.volume),
        tts_to_mp3(text, args.voice_b, mp3_b, args.rate, args.volume),
    )

    # Convert to wav
    mp3_to_wav(mp3_i, wav_i, normalize_dbfs=args.normalize_dbfs)
    mp3_to_wav(mp3_a, wav_a, normalize_dbfs=args.normalize_dbfs)
    mp3_to_wav(mp3_b, wav_b, normalize_dbfs=args.normalize_dbfs)

    # Cleanup temp
    for p in (mp3_i, mp3_a, mp3_b):
        try:
            p.unlink()
        except OSError:
            pass

    print("[OK] Generated:")
    print(f" - {wav_i}")
    print(f" - {wav_a}")
    print(f" - {wav_b}")


def parse_args():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to synthesize")
    src.add_argument("--text_file", help="Path to a UTF-8 text file to synthesize")

    ap.add_argument("--out_dir", default="out_wav", help="Output directory")

    ap.add_argument("--voice_i", default=DEFAULT_VOICE_I, help="Voice I (instruction)")
    ap.add_argument("--voice_a", default=DEFAULT_VOICE_A, help="Voice A (input)")
    ap.add_argument("--voice_b", default=DEFAULT_VOICE_B, help="Voice B (output)")

    ap.add_argument("--out_i", default="voice_I.wav", help="Output wav filename for voice I")
    ap.add_argument("--out_a", default="voice_A.wav", help="Output wav filename for voice A")
    ap.add_argument("--out_b", default="voice_B.wav", help="Output wav filename for voice B")

    ap.add_argument("--rate", default="+0%", help="Speech rate, e.g. +10% or -10%")
    ap.add_argument("--volume", default="+0%", help="Volume, e.g. +10% or -10%")

    ap.add_argument(
        "--normalize_dbfs",
        type=float,
        default=None,
        help="Normalize each WAV to target dBFS (e.g. -16.0). Leave empty to disable.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
