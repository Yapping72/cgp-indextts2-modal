import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx


VOICE_ID = "af_sky"
TIMEOUT = 300.0
OUT_DIR = Path(__file__).parent / "deployment_tests"

# Shared texts for all emotions
TEXT_BASELINE = "Welcome to Love Island. Tonight, everything changes."
TEXT_SAD = "She walked away and never looked back. It was over."
TEXT_TENSE = "Nobody knew what was about to happen next. The silence was unbearable."
TEXT_EXCITED = "She got the rose! I cannot believe it — this is insane!"
TEXT_DRAMATIC = (
    "Penelope looked Blake dead in the eyes and said nothing. "
    "That silence said everything."
)
TEXT_CALM = "And with that, another chapter of Love Island comes to a close."
TEXT_MELANCHOLIC = (
    "She had given everything to this villa, and in the end, "
    "it gave her nothing back."
)
TEXT_ANGRY = "He lied. He sat there, looked her in the face, and lied."
TEXT_QWEN = "Blake had been lying from the very beginning."

EMO_CONFIGS = [
    {
        "slug": "baseline_speaker",
        "label": "Mode 0 — SPEAKER (baseline)",
        "text": TEXT_BASELINE,
        "emotion_hints": None,
    },
    {
        "slug": "sad",
        "label": "Mode 2 — VECTOR sad",
        "text": TEXT_SAD,
        "emotion_hints": {
            "emo_mode": 2,
            "emo_weight": 1.0,
            "emo_vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    },
    {
        "slug": "tense",
        "label": "Mode 2 — VECTOR tense (blended)",
        "text": TEXT_TENSE,
        "emotion_hints": {
            "emo_mode": 2,
            "emo_weight": 1.0,
            "emo_vector": [0.0, 0.3, 0.1, 0.5, 0.0, 0.2, 0.0, 0.0],
        },
    },
    {
        "slug": "excited",
        "label": "Mode 2 — VECTOR excited",
        "text": TEXT_EXCITED,
        "emotion_hints": {
            "emo_mode": 2,
            "emo_weight": 1.0,
            "emo_vector": [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
        },
    },
    {
        "slug": "dramatic",
        "label": "Mode 2 — VECTOR dramatic (multi-dim blend)",
        "text": TEXT_DRAMATIC,
        "emotion_hints": {
            "emo_mode": 2,
            "emo_weight": 1.0,
            "emo_vector": [0.0, 0.3, 0.3, 0.0, 0.0, 0.3, 0.3, 0.0],
        },
    },
    {
        "slug": "calm",
        "label": "Mode 2 — VECTOR calm",
        "text": TEXT_CALM,
        "emotion_hints": {
            "emo_mode": 2,
            "emo_weight": 1.0,
            "emo_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    },
    {
        "slug": "melancholic",
        "label": "Mode 2 — VECTOR melancholic",
        "text": TEXT_MELANCHOLIC,
        "emotion_hints": {
            "emo_mode": 2,
            "emo_weight": 1.0,
            "emo_vector": [0.0, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0],
        },
    },
    {
        "slug": "angry",
        "label": "Mode 2 — VECTOR angry",
        "text": TEXT_ANGRY,
        "emotion_hints": {
            "emo_mode": 2,
            "emo_weight": 1.0,
            "emo_vector": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    },
    {
        "slug": "text_mode_qwen",
        "label": "Mode 3 — TEXT (experimental Qwen)",
        "text": TEXT_QWEN,
        "emotion_hints": {
            "emo_mode": 3,
            "emo_weight": 1.0,
            "emo_text": (
                "Deliver this with quiet, controlled disgust — the kind of person "
                "who is deeply unimpressed but trying to stay composed."
            ),
        },
    },
]

TEST_CASES = []

# For every emotion config, create a baseline + emotional pair with identical text.
for cfg in EMO_CONFIGS:
    base_slug = cfg["slug"]

    # Baseline (no emotion_hints)
    TEST_CASES.append(
        {
            "name": f"baseline_{base_slug}",
            "label": f"Baseline — {cfg['label']}",
            "payload": {
                "text": cfg["text"],
                "voice_id": VOICE_ID,
            },
        }
    )

    # If there is an emotion_hints config, add the emotional variant
    if cfg["emotion_hints"] is not None:
        TEST_CASES.append(
            {
                "name": f"emo_{base_slug}",
                "label": cfg["label"],
                "payload": {
                    "text": cfg["text"],
                    "voice_id": VOICE_ID,
                    "emotion_hints": cfg["emotion_hints"],
                },
            }
        )

# ── Diagnostic cases — all use TEXT_SAD to isolate variables ─────────────────
#
# Theory: normalize_emo_vec(apply_bias=True) adds a bias floor to all 8 dims,
# diluting pure emotions. Scaling the raw vector up makes the bias negligible.
#
# [0,0,1,0,...] + bias[0.05,...] → sad ≈ 51% of budget after normalize
# [0,0,5,0,...] + bias[0.05,...] → sad ≈ 83% of budget after normalize
# [0,0,10,0,...] + bias[0.05,...] → sad ≈ 93% of budget after normalize
#
# If 5x and 10x sound noticeably sadder than 1x → bias theory confirmed.
# If all three sound identical → normalization is ratio-only (bias theory wrong).
#
# Also tests emo_weight to confirm auto-override (0.65 vs 1.0 should be identical
# when no emo_audio_prompt is given — IndexTTS2 forces emo_alpha=1.0 in that case).

_DIAG_TEXT = TEXT_SAD  # fixed text so any delta is purely from parameter changes

DIAGNOSTIC_CASES = [
    {
        "name": "diag_sad_1x_w065",
        "label": "Diag: sad 1× weight=0.65 (current default)",
        "payload": {
            "text": _DIAG_TEXT,
            "voice_id": VOICE_ID,
            "emotion_hints": {"emo_mode": 2, "emo_weight": 1.0, "emo_vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        },
    },
    {
        "name": "diag_sad_1x_w100",
        "label": "Diag: sad 1× weight=1.0 (if identical → auto-override confirmed)",
        "payload": {
            "text": _DIAG_TEXT,
            "voice_id": VOICE_ID,
            "emotion_hints": {"emo_mode": 2, "emo_weight": 1.0, "emo_vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        },
    },
    {
        "name": "diag_sad_5x_w065",
        "label": "Diag: sad 5× vector (bias dilution test)",
        "payload": {
            "text": _DIAG_TEXT,
            "voice_id": VOICE_ID,
            "emotion_hints": {"emo_mode": 2, "emo_weight": 1.0, "emo_vector": [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        },
    },
    {
        "name": "diag_sad_10x_w065",
        "label": "Diag: sad 10× vector (near-pure sad after bias)",
        "payload": {
            "text": _DIAG_TEXT,
            "voice_id": VOICE_ID,
            "emotion_hints": {"emo_mode": 2, "emo_weight": 1.0, "emo_vector": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        },
    },
    {
        "name": "diag_sad_bypass",
        "label": "Diag: sad bypass_normalize (raw vector, no bias) — requires Modal redeploy",
        "payload": {
            "text": _DIAG_TEXT,
            "voice_id": VOICE_ID,
            "emotion_hints": {"emo_mode": 2, "emo_weight": 1.0, "emo_vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "bypass_normalize": True},
        },
    },
]

TEST_CASES.extend(DIAGNOSTIC_CASES)


async def run_case(client: httpx.AsyncClient, base_url: str, case: dict) -> dict:
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
            return {"name": name, "label": label, "status": "NOT_READY"}

        resp.raise_for_status()
        data = resp.json()

        audio_bytes = base64.b64decode(data["audio_b64"])
        out_path = OUT_DIR / f"{name}.mp3"
        out_path.write_bytes(audio_bytes)

        duration = data.get("duration_seconds", 0.0)
        words = len(data.get("word_events", []))
        print(f"OK  {duration:.1f}s  {words} words  ({elapsed:.1f}s wall)")
        return {
            "name": name,
            "label": label,
            "status": "OK",
            "duration_seconds": duration,
            "word_count": words,
            "file": str(out_path),
            "wall_seconds": round(elapsed, 1),
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"FAIL ({exc!r})")
        return {
            "name": name,
            "label": label,
            "status": "FAIL",
            "error": str(exc),
        }


async def main(base_url: str, diag_only: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_url = base_url.rstrip("/")
    cases = DIAGNOSTIC_CASES if diag_only else TEST_CASES
    label = "diagnostic" if diag_only else "full"
    print(f"\nIndexTTS2 emotion mode tests [{label}] → {base_url}")
    print(f"Output dir: {OUT_DIR}\n")

    results = []
    async with httpx.AsyncClient() as client:
        for case in cases:
            result = await run_case(client, base_url, case)
            results.append(result)

    print("\n── Results ──────────────────────────────────────────────────────────")
    print(f"{'File':<30} {'Status':<10} {'Duration':>10} {'Words':>6} {'Wall':>7}")
    print("-" * 70)
    for r in results:
        status = r["status"]
        duration = f"{r['duration_seconds']:.1f}s" if "duration_seconds" in r else "—"
        words = str(r.get("word_count", "—"))
        wall = f"{r['wall_seconds']:.1f}s" if "wall_seconds" in r else "—"
        print(f"{r['name']:<30} {status:<10} {duration:>10} {words:>6} {wall:>7}")

    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{ok}/{len(results)} passed  —  files in {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Modal endpoint base URL")
    parser.add_argument("--diag-only", action="store_true", help="Run only the 5 diagnostic cases")
    args = parser.parse_args()
    asyncio.run(main(args.url, diag_only=args.diag_only))
