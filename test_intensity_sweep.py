"""Intensity sweep — generates T1–T5 clips for a single emotion or all emotions.

For each emotion, produces 5 clips (one per tier) using a fixed representative text.
Files are named <emotion>_T1.mp3 … <emotion>_T5.mp3 inside OUT_DIR.

Usage:
    # Single emotion (faster, ~5 clips)
    python test_intensity_sweep.py --url https://<modal-url> --emotion sad

    # All 8 base emotions (~40 clips)
    python test_intensity_sweep.py --url https://<modal-url>

Levels (T3–T5 from the diagnostic sweep, renamed L1–L3):
    L1  weight=1.0  scale=5×   bypass=false  (strong, natural)
    L2  weight=1.0  scale=10×  bypass=false  (intense, theatrical)
    L3  weight=1.0  scale=1×   bypass=true   (raw / maximum)


Start cost: $22.33
No. Audios:
End Cost:
"""
import argparse
import asyncio
import base64
import time
from pathlib import Path

import httpx


VOICE_ID = "af_sky"
TIMEOUT = 300.0
OUT_DIR = Path(__file__).parent / "intensity_sweep"

# Levels: (label, emo_weight, vector_scale, bypass_normalize)
TIERS = [
    ("L1", 1.0, 5,  False),
    ("L2", 1.0, 10, False),
    ("L3", 1.0, 1,  True),
]

# Base vectors: single-dim dominant so intensity differences are cleanest
# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMOTIONS = {
    "sad":         {"vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "text": "She walked away and never looked back. It was over."},
    "angry":       {"vector": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "text": "He lied. He sat there, looked her in the face, and lied."},
    "fearful":     {"vector": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "text": "Nobody knew what was about to happen next. The silence was unbearable."},
    "excited":     {"vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "text": "She got the rose! I cannot believe it — this is insane!"},
    "melancholic": {"vector": [0.0, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0], "text": "She had given everything to this villa, and in the end, it gave her nothing back."},
    "calm":        {"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "text": "And with that, another chapter of Love Island comes to a close."},
    "surprised":   {"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "text": "She got the rose! I cannot believe it — this is insane!"},
    "disgusted":   {"vector": [0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "text": "Blake had been lying from the very beginning."},
}


def _build_cases(emotions: dict[str, dict]) -> list[dict]:
    cases = []
    for emo_name, cfg in emotions.items():
        base_vec = cfg["vector"]
        text = cfg["text"]
        for tier_label, weight, scale, bypass in TIERS:
            scaled = [round(v * scale, 4) for v in base_vec]
            hints: dict = {"emo_mode": 2, "emo_weight": weight, "emo_vector": scaled}
            if bypass:
                hints["bypass_normalize"] = True
                hints["emo_vector"] = base_vec  # bypass uses unscaled raw vector
            cases.append({
                "name": f"{emo_name}_{tier_label}",
                "label": f"{emo_name} {tier_label} (w={weight}, {scale}×{', bypass' if bypass else ''})",
                "payload": {"text": text, "voice_id": VOICE_ID, "emotion_hints": hints},
            })
    return cases


async def run_case(client: httpx.AsyncClient, base_url: str, case: dict, out_dir: Path) -> dict:
    name = case["name"]
    label = case["label"]
    print(f"  → {label} ... ", end="", flush=True)
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{base_url}/synthesize/sync",
            json=case["payload"],
            timeout=TIMEOUT,
        )
        elapsed = time.monotonic() - t0

        if resp.status_code == 503:
            print("SKIP (503 NOT_READY)")
            return {"name": name, "status": "NOT_READY"}

        resp.raise_for_status()
        data = resp.json()

        audio_bytes = base64.b64decode(data["audio_b64"])
        out_path = out_dir / f"{name}.mp3"
        out_path.write_bytes(audio_bytes)

        duration = data.get("duration_seconds", 0.0)
        print(f"OK  {duration:.1f}s  ({elapsed:.1f}s wall)")
        return {"name": name, "status": "OK", "duration_seconds": duration, "file": str(out_path)}
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"FAIL ({exc!r})")
        return {"name": name, "status": "FAIL", "error": str(exc)}


async def main(base_url: str, emotion_filter: str | None):
    if emotion_filter:
        if emotion_filter not in EMOTIONS:
            print(f"Unknown emotion '{emotion_filter}'. Available: {', '.join(EMOTIONS)}")
            return
        emotions = {emotion_filter: EMOTIONS[emotion_filter]}
    else:
        emotions = EMOTIONS

    out_dir = OUT_DIR / (emotion_filter or "all")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = base_url.rstrip("/")

    cases = _build_cases(emotions)
    scope = emotion_filter or f"all {len(emotions)} emotions"
    print(f"\nIntensity sweep [{scope}] → {base_url}")
    print(f"Output: {out_dir}  ({len(cases)} clips)\n")

    results = []
    async with httpx.AsyncClient() as client:
        for case in cases:
            result = await run_case(client, base_url, case, out_dir)
            results.append(result)

    # Summary grouped by emotion
    print("\n── Summary ──────────────────────────────────────────────────────────")
    for emo_name in emotions:
        print(f"\n  {emo_name}:")
        for r in results:
            if r["name"].startswith(emo_name):
                tier = r["name"].split("_")[-1]
                status = r["status"]
                dur = f"{r['duration_seconds']:.1f}s" if "duration_seconds" in r else "—"
                print(f"    {tier}  {status:<8}  {dur}")

    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{ok}/{len(results)} passed  —  files in {out_dir}/")
    print(f"\nListen order for each emotion: T1 → T2 → T3 → T4 → T5")
    print("Note your observations in emotion_intensity_checklist.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Modal endpoint base URL")
    parser.add_argument(
        "--emotion",
        choices=list(EMOTIONS),
        default=None,
        help="Single emotion to sweep (omit for all)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.url, args.emotion))
