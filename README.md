# IndexTTS2 Modal Deployment

Expressive text-to-speech service powered by [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) and [OpenAI Whisper](https://github.com/openai/whisper), deployed on [Modal](https://modal.com) with GPU acceleration. Synthesis requests return base64-encoded MP3 audio alongside Whisper-derived word-level timestamps.

***

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Voice System](#voice-system)
  - [voices/ Directory](#voices-directory)
  - [voices.json Manifest](#voicesjson-manifest)
  - [Adding a New Voice](#adding-a-new-voice)
  - [Voice Resolution Priority](#voice-resolution-priority)
- [HTTP API](#http-api)
  - [GET /voices](#get-voices)
  - [POST /synthesize/sync](#post-synthesizesync)
  - [GET /health](#get-health)
  - [GET /ready](#get-ready)
- [Emotion Hints](#emotion-hints)
- [Infrastructure](#infrastructure)
- [Cold Start & Scaling](#cold-start--scaling)
- [Local Development](#local-development)

***

## Architecture Overview

```
cgp-indextts2-modal repo
├── voices/               ← Reference MP3s (committed directly, no LFS)
│   ├── af_nova.mp3
│   └── ...
└── voices.json           ← Voice manifest (id, file, gender, accent)

Modal Image Build
└── git clone → /opt/cgp-voices
    ├── /opt/cgp-voices/voices/        (VOICES_DIR)
    └── /opt/cgp-voices/voices.json    (VOICES_MANIFEST)

Modal Runtime (A10G GPU)
├── IndexTTS-2  (~8 GB VRAM)   ← voice cloning synthesis
└── Whisper base (~0.1 GB)     ← word-level timestamp extraction
```

The entire voice library is baked into the container image at build time — no manual volume uploads or external storage buckets are required.

***

## Voice System

Voices are stored inside this repository and loaded automatically every time the Modal image is built. There are two key artifacts:

### `voices/` Directory

**Path inside the container:** `/opt/cgp-voices/voices/`

This directory holds the reference audio files used for voice cloning. Each file is a short MP3 clip (typically 5–30 seconds) that captures the speaker's timbre and prosody.

**Rules for audio files:**
- Format: MP3 or WAV
- Minimum file size: **> 1 KB** (files ≤ 1 KB are skipped automatically)
- Files must be committed directly to Git — **do not use Git LFS** (the repo's LFS budget is exhausted; the image build uses `GIT_LFS_SKIP_SMUDGE=1`)
- Filename must match the `"file"` field in `voices.json`

> **Tip:** Aim for 10–30 seconds of clean, noise-free audio per voice. Longer clips do not improve quality and increase image build time.

***

### `voices.json` Manifest

**Path inside the container:** `/opt/cgp-voices/voices.json`

A flat JSON array that declares every available voice and its metadata. The deployment reads this file on startup to build the internal voice map.

#### Schema

```json
[
  {
    "id":     "af_nova",
    "file":   "af_nova.mp3",
    "gender": "female",
    "accent": "american"
  }
]
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | ✅ | Unique voice identifier used in API requests (`voice_id`) |
| `file` | `string` | ✅ | Filename relative to `voices/` — e.g. `"af_nova.mp3"` |
| `gender` | `string` | ✅ | `"male"`, `"female"`, or `"neutral"` |
| `accent` | `string` | ✅ | Free-text accent descriptor — e.g. `"american"`, `"british"` |

**Validation at startup:**
- Entries whose `file` does not exist under `voices/` are logged as warnings and skipped.
- Entries whose file is ≤ 1 KB are also skipped (treated as corrupt/placeholder).
- If `voices.json` is missing entirely, the system falls back to scanning `voices/` for `*.mp3` / `*.wav` files and assigns `gender: "neutral"` and `accent: "unknown"` to each.

***

### Adding a New Voice

1. **Add the audio file** to `voices/` in this repository.
2. **Update `voices.json`** — append a new entry with a unique `id`.
3. **Commit both files** and push to `main`.
4. **Redeploy:**

```bash
modal deploy indextts2_app.py
```

The image rebuild clones the latest repo state, picks up the new entry, and makes it available at `GET /voices` immediately after the deploy completes.

> ⚠️ No deploy = no new voice. The voice library is frozen into the container image at build time.

***

### Voice Resolution Priority

When a synthesis request arrives, the reference audio is resolved in this order:

| Priority | Condition | Source |
|----------|-----------|--------|
| 1 | `reference_audio_b64` is provided | Inline base64-encoded audio (voice cloning on-the-fly) |
| 2 | `voice_id` matches an entry in `voices.json` | `voices/` directory inside the container |
| 3 | `voice_id == "default"` | First voice in `voices.json` |
| 4 | Fallback | `DEFAULT_REFERENCE_PATH` in the Modal Volume (manually uploaded) |

If none of the above resolves, the request returns `HTTP 422` with a list of available voice IDs.

***

## HTTP API

Base URL is the Modal-assigned ASGI endpoint (printed by `modal deploy`).

### `GET /voices`

Returns all voices declared in `voices.json` along with a total count.

**Response:**
```json
{
  "voices": [
    { "id": "af_nova", "file": "af_nova.mp3", "gender": "female", "accent": "american" }
  ],
  "count": 1
}
```

***

### `POST /synthesize/sync`

Synthesizes speech and returns audio + word timestamps in a single synchronous response.

**Request body:**
```json
{
  "text": "Hello, world!",
  "voice_id": "af_nova",
  "reference_audio_b64": null,
  "emotion_hints": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | — | Text to synthesize (must not be empty) |
| `voice_id` | `string` | `"default"` | ID from `voices.json`, or `"default"` for the first voice |
| `reference_audio_b64` | `string \| null` | `null` | Base64-encoded audio for inline voice cloning; overrides `voice_id` |
| `emotion_hints` | `object \| null` | `null` | Optional emotion control — see [Emotion Hints](#emotion-hints) |

**Response:**
```json
{
  "audio_b64": "<base64 MP3>",
  "duration_seconds": 3.84,
  "word_events": [
    { "word": "Hello,", "start_time": 0.0, "end_time": 0.42 },
    { "word": "world!", "start_time": 0.44, "end_time": 0.91 }
  ],
  "status": "SUCCESS"
}
```

`word_events` format is intentionally compatible with the Kokoro TTS path so both backends can be swapped transparently downstream.

**Error responses:**

| Code | Condition |
|------|-----------|
| `422` | Empty `text`, or `voice_id` not found and no `reference_audio_b64` |
| `500` | Internal synthesis failure |
| `503` | Container still loading (model not ready) |

***

### `GET /health`

Lightweight liveness probe — always returns `200 {"status": "ok"}` once the container is up.

***

### `GET /ready`

Readiness probe — returns `200 {"status": "READY"}` only after `@modal.enter()` completes (IndexTTS-2 and Whisper fully loaded). Returns `503 {"status": "NOT_READY"}` during warm-up.

***

## Emotion Hints

Pass an `emotion_hints` object in the synthesis request to control expressive style.

| `emo_mode` | Mode name | Required fields | Description |
|------------|-----------|-----------------|-------------|
| `0` | **SPEAKER** (default) | — | Style cloned directly from reference audio; no explicit emotion |
| `2` | **VECTOR** | `emo_vector` | Numeric emotion vector; normalized + bias-applied by default |
| `3` | **TEXT** | `emo_text` (optional) | Qwen-based text-conditioned emotion; reads `emo_text` or the spoken text |

**Additional fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `emo_weight` | `float` | `0.65` | Blend weight between speaker style and emotion target |
| `emo_vector` | `list[float]` | — | Raw emotion vector (used in mode `2`) |
| `bypass_normalize` | `bool` | `false` | Skip normalization of `emo_vector` — use for diagnostics/max intensity only |
| `emo_text` | `string` | — | Emotion descriptor string for mode `3` |

> Mode `1` (AUDIO emotion) is reserved for future extension and not wired up yet.

***

## Infrastructure

| Resource | Value |
|----------|-------|
| GPU | NVIDIA A10G (24 GB VRAM) |
| IndexTTS-2 VRAM | ~8 GB |
| Whisper `base` VRAM | ~0.1 GB |
| Modal Volumes | `indextts2-model` (model weights), `indextts2-outputs` (audio outputs) |
| HF model | `IndexTeam/IndexTTS-2` |
| Container timeout | 10 minutes |
| Scaledown window | 5 minutes idle |
| HF Token secret | `HF_TOKEN` (Modal Secret named `HF_TOKEN`) |

***

## Cold Start & Scaling

- **First run:** ~90–120 seconds — downloads `IndexTeam/IndexTTS-2` from Hugging Face and caches it in the `indextts2-model` Modal Volume.
- **Subsequent cold starts:** ~30–60 seconds — weights are loaded from the cached Volume.
- **Warm requests:** < 5 seconds for typical short utterances.
- **Scaledown:** Container shuts down after 5 minutes of idle time.

Poll `GET /ready` before sending synthesis requests to a freshly deployed or restarted container.

***

## Local Development

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Deploy (builds image + starts app)
modal deploy indextts2_app.py

# Run a quick synthesis test
curl -s -X POST <MODAL_URL>/synthesize/sync \
  -H "Content-Type: application/json" \
  -d '{"text": "Testing IndexTTS2.", "voice_id": "af_nova"}' \
  | python3 -c "
import sys, json, base64
r = json.load(sys.stdin)
open('out.mp3','wb').write(base64.b64decode(r['audio_b64']))
print(f'Duration: {r[\"duration_seconds\"]:.2f}s — {len(r[\"word_events\"])} word events')
"
```












# Configuration Steps
1. Logout of modal.com
2. ``modal setup`` 
    * Login using desired account
3. Create new secret in Modal: 
    * HF_TOKEN = see .env file
4. ``cat ~/.modal.toml``
    * Retrieve MODAL_TOKEN_ID,  
5. ``modal deploy indextts2_modal.py``