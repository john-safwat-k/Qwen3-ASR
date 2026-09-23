#!/usr/bin/env python3
"""
Convert an (audio, transcript) CSV into the JSONL format expected by qwen3_asr_sft.py.

Input CSV (comma or tab separated, auto-detected):
    audio,transcript
    adapt_sample_0_clean,شوفلنا المشوار ده يا حج

Output JSONL (one object per line):
    {"audio": "/abs/path/adapt/adapt_sample_0_clean.wav", "text": "language Arabic<asr_text>شوفلنا المشوار ده يا حج"}

Example (audio files available locally; paths are resolved and verified):
    python csv_to_jsonl.py --csv downloaded_file/train.csv --audio_dir downloaded_file/train --out train.jsonl

Example (audio will only exist on the training server; paths are built, not checked):
    python csv_to_jsonl.py --csv adapt.csv --audio_root /data/downloaded_file/adapt --ext .wav --out eval.jsonl
"""
import argparse
import csv
import json
import os
import sys

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


def index_audio_dir(audio_dir: str):
    """Map file stem -> absolute path for every audio file under audio_dir (recursive)."""
    index = {}
    for root, _, files in os.walk(audio_dir):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() in AUDIO_EXTS:
                index.setdefault(stem, os.path.abspath(os.path.join(root, name)))
    return index


def detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def main():
    p = argparse.ArgumentParser(description="CSV -> Qwen3-ASR fine-tuning JSONL")
    p.add_argument("--csv", required=True, help="Input CSV with audio + transcript columns")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio_dir", help="Local folder containing the audio files (paths are verified)")
    src.add_argument("--audio_root", help="Folder the audio will live in on the server (paths are not checked)")
    p.add_argument("--ext", default=".wav", help="Audio extension used with --audio_root (default: .wav)")
    p.add_argument("--out", required=True, help="Output .jsonl path")
    p.add_argument("--language", default="Arabic", help="Language prefix; use None if unknown")
    p.add_argument("--audio_col", default="audio")
    p.add_argument("--text_col", default="transcript")
    args = p.parse_args()

    audio_index = None
    if args.audio_dir:
        audio_index = index_audio_dir(args.audio_dir)
        if not audio_index:
            sys.exit(f"No audio files found under {args.audio_dir}")
    ext = args.ext if args.ext.startswith(".") else "." + args.ext

    delimiter = detect_delimiter(args.csv)
    written, missing, empty = 0, [], 0

    with open(args.csv, "r", encoding="utf-8-sig", newline="") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter=delimiter)
        for col in (args.audio_col, args.text_col):
            if col not in reader.fieldnames:
                sys.exit(f"Column '{col}' not found. CSV columns: {reader.fieldnames}")

        for row in reader:
            audio_id = (row[args.audio_col] or "").strip()
            text = " ".join((row[args.text_col] or "").split())
            if not audio_id or not text:
                empty += 1
                continue

            # Accept IDs with or without an extension.
            stem = os.path.splitext(os.path.basename(audio_id))[0]
            if audio_index is None:
                path = f"{args.audio_root.rstrip('/')}/{stem}{ext}"
            else:
                path = audio_index.get(stem)
            if path is None:
                missing.append(audio_id)
                continue

            record = {"audio": path, "text": f"language {args.language}<asr_text>{text}"}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} lines to {args.out}")
    if empty:
        print(f"Skipped {empty} rows with empty audio/transcript")
    if missing:
        print(f"Skipped {len(missing)} rows with no matching audio file, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
