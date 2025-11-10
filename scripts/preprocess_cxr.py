# scripts/preprocess_cxr.py
import argparse
from pathlib import Path
import json, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nih", required=True)
    p.add_argument("--pad", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": time.time(),
        "nih_path": str(Path(args.nih).resolve()),
        "pad_path": str(Path(args.pad).resolve()),
        "note": "placeholder preprocess; replace with real transforms later"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "READY.txt").write_text("processed placeholder\n")

if __name__ == "__main__":
    main()
