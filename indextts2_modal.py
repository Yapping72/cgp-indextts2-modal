"""IndexTTS2 Modal deployment — expressive TTS with Whisper word timestamps.

Exposes HTTP endpoints:
    GET  /voices           → list all voices from the manifest
    POST /synthesize/sync  → synthesize text, return base64 MP3 + word events

Request:  {"text": str, "voice_id": "af_nova", "reference_audio_b64": str|null}
Response: {"audio_b64": str, "duration_seconds": float, "word_events": [...]}

word_events format (matches the Kokoro path):
    [{"word": str, "start_time": float, "end_time": float}]

── Voice system ──────────────────────────────────────────────────────────────
Voices are defined in voices.json inside the cgp-indextts2-modal repo.
The repo is cloned into /opt/voices during the image build — no manual volume
uploads required. To add a voice: commit the MP3 + update voices.json, then
redeploy with `modal deploy`.

voices.json schema:
    [{"id": str, "file": str, "gender": "male|female|neutral", "accent": str}, ...]

── Modal GPU ─────────────────────────────────────────────────────────────────
GPU: A10G (24 GB VRAM — IndexTTS2 ~8 GB, Whisper base ~0.1 GB).
Cold start: ~90–120 s (model download cached in Volume after first run).
Scaledown: 5 minutes idle.
"""
import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID = "IndexTeam/IndexTTS-2"
APP_NAME = "indextts2-expressive-tts"
VOICES_REPO = Path("/opt/cgp-voices")       # git clone root
VOICES_DIR = VOICES_REPO / "voices"         # MP3s live here
VOICES_MANIFEST = VOICES_REPO / "voices.json"  # manifest at repo root

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .uv_pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "accelerate>=1.6.0",
        "transformers==4.52.1",
        "openai-whisper>=20231117",
        "huggingface-hub>=0.23.0",
        "fastapi[standard]",
        "pydantic>=2.0",
        "omegaconf",
        "einops",
        "matplotlib",
        "scipy",
        "librosa",
        "soundfile",
        "inflect",
        "pypinyin",
        "cn2an",
        "jieba",
        "WeTextProcessing",
        "num2words",
        "tqdm",
        "vector-quantize-pytorch",
        "encodec",
        "sentencepiece",
        "tokenizers",
        "safetensors",
        "g2p-en",
        "descript-audiotools",
        "ffmpeg-python",
        "munch",
        "json5",
        "textstat",
        "modelscope",
    )
    .run_commands(
        # IndexTTS code — skip LFS (repo LFS budget is exhausted; we only need Python)
        "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git /opt/indextts",
        "pip install --no-deps -e /opt/indextts",
        # Voice repo — MP3s committed directly (no LFS), driven by voices.json manifest.
        # To add a voice: commit MP3 + update voices.json → redeploy.
        # Clone into /opt/cgp-voices; voices.json and voices/ subdir land there.
        # VOICES_DIR (/opt/voices) points at the voices/ subdir inside the repo.
        "git clone https://github.com/Yapping72/cgp-indextts2-modal.git /opt/cgp-voices",
    )
    .env({"HF_HOME": "/models", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App(APP_NAME, secrets=[modal.Secret.from_name("HF_TOKEN")])

outputs = modal.Volume.from_name("indextts2-outputs", create_if_missing=True)
model_vol = modal.Volume.from_name("indextts2-model", create_if_missing=True)

MODEL_PATH = Path("/models")
MODEL_DIR = MODEL_PATH / "indextts2"
DEFAULT_REFERENCE_PATH = MODEL_DIR / "default_reference.wav"

MINUTES = 60


# ── Helpers (module-level, no GPU required) ────────────────────────────────────

def _get_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    return None


def _load_voice_manifest() -> list[dict]:
    """Load voices.json from the cloned voice repo.

    Each entry: {"id": str, "file": str, "gender": str, "accent": str}
    Falls back to scanning VOICES_DIR for MP3/WAV files if manifest is missing.
    """
    if VOICES_MANIFEST.exists():
        entries = json.loads(VOICES_MANIFEST.read_text())
        # Resolve each entry's file path; drop entries whose file is missing
        resolved = []
        for entry in entries:
            audio_path = VOICES_DIR / entry["file"]
            if audio_path.exists() and audio_path.stat().st_size > 1024:
                resolved.append({**entry, "_path": audio_path})
            else:
                print(f"[IndexTTS2] voices.json: '{entry['file']}' not found or too small, skipping")
        return resolved

    # Fallback: filesystem scan (no manifest)
    print("[IndexTTS2] voices.json not found — falling back to filesystem scan")
    results = []
    for ext in ("*.mp3", "*.wav"):
        for f in sorted(VOICES_DIR.glob(ext)):
            if f.stat().st_size < 1024:
                continue
            results.append({"id": f.stem, "file": f.name, "gender": "neutral", "accent": "unknown", "_path": f})
    return results


def _measure_audio_duration(audio_path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
         "-c:a", "libmp3lame", "-q:a", "2", str(mp3_path)],
        check=True,
    )


# ── Modal class ────────────────────────────────────────────────────────────────

@app.cls(
    image=image,
    volumes={Path("/outputs"): outputs, MODEL_PATH: model_vol},
    gpu="A10G",
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
)
class IndexTTS2:

    @modal.enter()
    def load_model(self):
        from huggingface_hub import snapshot_download
        import whisper

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(MODEL_PATH),
            local_dir=str(MODEL_DIR),
            token=_get_hf_token(),
        )

        # ── Load voice manifest ───────────────────────────────────────────────
        entries = _load_voice_manifest()
        # voice_map: id → Path
        self.voice_map: dict[str, Path] = {e["id"]: e["_path"] for e in entries}
        # voice_meta: id → public-facing metadata (no _path)
        self.voice_meta: list[dict] = [{k: v for k, v in e.items() if k != "_path"} for e in entries]

        print(f"[IndexTTS2] {len(self.voice_map)} voice(s) loaded: {list(self.voice_map)}")
        if not self.voice_map:
            print(
                "[IndexTTS2] No voices found. Add MP3s + update voices.json in "
                "cgp-indextts2-modal and redeploy."
            )

        # ── Load IndexTTS2 ────────────────────────────────────────────────────
        from indextts.infer_v2 import IndexTTS2 as _IndexTTS2
        cfg_files = sorted(MODEL_DIR.rglob("config.yaml"))
        if not cfg_files:
            raise RuntimeError(
                f"config.yaml not found under {MODEL_DIR}. "
                f"Files: {[str(p.relative_to(MODEL_DIR)) for p in sorted(MODEL_DIR.rglob('*'))[:30]]}"
            )
        self.tts = _IndexTTS2(model_dir=str(MODEL_DIR), cfg_path=str(cfg_files[0]))

        # ── Load Whisper ──────────────────────────────────────────────────────
        self.whisper_model = whisper.load_model("base")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_reference(self, voice_id: str, tmp_path: Path, reference_audio_b64: str | None) -> Path:
        """Resolve voice_id to a reference audio path.

        Priority:
          1. Inline reference_audio_b64 (base64-encoded audio bytes)
          2. Named voice from manifest (voice_id matches an entry's "id")
          3. "default" → first voice in the manifest
          4. DEFAULT_REFERENCE_PATH in the Modal Volume (manually uploaded fallback)
        """
        if reference_audio_b64:
            ref = tmp_path / "reference.wav"
            ref.write_bytes(base64.b64decode(reference_audio_b64))
            return ref

        if voice_id in self.voice_map:
            return self.voice_map[voice_id]

        if voice_id == "default":
            if self.voice_map:
                first = next(iter(self.voice_map.values()))
                print(f"[IndexTTS2] voice_id='default' → using '{first.stem}'")
                return first
            if DEFAULT_REFERENCE_PATH.exists():
                return DEFAULT_REFERENCE_PATH

        raise ValueError(
            f"Voice '{voice_id}' not found. "
            f"Available: {list(self.voice_map) or '(none)'}. "
            "Pass reference_audio_b64 for inline cloning, or add the voice to voices.json."
        )

    def _ensure_wav(self, audio_path: Path, tmp_path: Path) -> Path:
        """Convert any audio to 16-bit mono 22050 Hz PCM WAV for soundfile/librosa."""
        out = tmp_path / f"{audio_path.stem}_pcm.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
             "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", str(out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            import sys
            print(f"[IndexTTS2] _ensure_wav failed for {audio_path}: {result.stderr!r}", file=sys.stderr)
            raise RuntimeError(f"ffmpeg conversion failed for {audio_path.name}: {result.stderr.strip()}")
        return out

    def _do_list_voices(self) -> list[dict]:
        return self.voice_meta

    def _do_synthesize(
        self,
        text: str,
        voice_id: str = "default",
        reference_audio_b64: str | None = None,
        emotion_hints: dict | None = None,
    ) -> dict:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            ref_path = self._resolve_reference(voice_id, tmp_path, reference_audio_b64)
            ref_wav = self._ensure_wav(ref_path, tmp_path)

            # Decode emotion_hints → infer() kwargs (mirrors gen_single() in webui.py)
            hints = emotion_hints or {}
            emo_mode = int(hints.get("emo_mode", 0))
            emo_weight = float(hints.get("emo_weight", 0.65))
            emo_vector = hints.get("emo_vector")   # pre-normalization list or None
            emo_text = hints.get("emo_text") or None

            bypass_normalize: bool = bool(hints.get("bypass_normalize", False))

            # Build infer() kwargs from mode
            infer_kwargs: dict = {}
            if emo_mode == 0 or not hints:
                # SPEAKER mode: no explicit emotion params — infer uses spk_audio_prompt only
                pass
            elif emo_mode == 2 and emo_vector is not None:
                # VECTOR mode: normalize (default) or bypass for diagnostic/max-intensity use
                if bypass_normalize:
                    # Pass raw vector directly — no bias, no cap. Use for testing or max intensity.
                    infer_kwargs["emo_vector"] = emo_vector
                else:
                    infer_kwargs["emo_vector"] = self.tts.normalize_emo_vec(emo_vector, apply_bias=True)
                infer_kwargs["emo_alpha"] = emo_weight
            elif emo_mode == 3:
                # TEXT mode (experimental): Qwen reads emo_text or spoken text
                infer_kwargs["use_emo_text"] = True
                infer_kwargs["emo_text"] = emo_text
                infer_kwargs["emo_alpha"] = emo_weight
            # Mode 1 (AUDIO) left for future extension — not wired yet

            wav_out = tmp_path / f"output_{int(time.time())}.wav"
            self.tts.infer(spk_audio_prompt=str(ref_wav), text=text, output_path=str(wav_out), **infer_kwargs)

            mp3_out = tmp_path / "output.mp3"
            _convert_wav_to_mp3(wav_out, mp3_out)
            duration = _measure_audio_duration(mp3_out)
            audio_bytes = mp3_out.read_bytes()

            word_events: list[dict] = []
            try:
                result = self.whisper_model.transcribe(str(mp3_out), word_timestamps=True, fp16=True)
                for seg in result.get("segments", []):
                    for w in seg.get("words", []):
                        word = w.get("word", "").strip()
                        if word:
                            word_events.append({
                                "word": word,
                                "start_time": float(w["start"]),
                                "end_time": float(w["end"]),
                            })
            except Exception as exc:
                import sys
                print(f"[IndexTTS2] Whisper failed: {exc}", file=sys.stderr)

        return {
            "audio_b64": base64.b64encode(audio_bytes).decode(),
            "duration_seconds": duration,
            "word_events": word_events,
        }

    # ── Modal Python API ───────────────────────────────────────────────────────

    @modal.method()
    def list_voices(self) -> list[dict]:
        return self._do_list_voices()

    @modal.method()
    def synthesize(self, text: str, voice_id: str = "default", reference_audio_b64: str | None = None) -> dict:
        return self._do_synthesize(text, voice_id, reference_audio_b64)

    # ── HTTP API (runs on the same GPU container — no cross-container spawning) ─

    @modal.asgi_app(requires_proxy_auth=False)
    def api(self):
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel as PydanticBaseModel
        from typing import Optional

        fast_app = FastAPI(title="IndexTTS2 Expressive TTS")

        @fast_app.get("/health")
        async def health():
            return {"status": "ok"}

        @fast_app.get("/ready")
        async def ready():
            """Machine-readable readiness probe — returns READY only after @modal.enter() completes."""
            from fastapi.responses import JSONResponse as _JSONResponse
            if hasattr(self, "tts") and hasattr(self, "whisper_model"):
                return {"status": "READY"}
            return _JSONResponse(status_code=503, content={"status": "NOT_READY"})

        @fast_app.get("/voices")
        async def list_voices_endpoint():
            """List all voices declared in voices.json with their metadata."""
            return {"voices": self._do_list_voices(), "count": len(self.voice_map)}

        class SynthesizeRequest(PydanticBaseModel):
            text: str
            voice_id: str = "default"
            reference_audio_b64: Optional[str] = None
            emotion_hints: Optional[dict] = None  # decoded by _do_synthesize to emo_vector/use_emo_text/etc.

        @fast_app.post("/synthesize/sync")
        async def synthesize_sync(body: SynthesizeRequest):
            from fastapi.responses import JSONResponse as _JSONResponse
            if not hasattr(self, "tts") or not hasattr(self, "whisper_model"):
                return _JSONResponse(status_code=503, content={"status": "NOT_READY"})
            if not body.text.strip():
                raise HTTPException(status_code=422, detail="text must not be empty")
            try:
                result = self._do_synthesize(body.text, body.voice_id, body.reference_audio_b64, body.emotion_hints)
                return {**result, "status": "SUCCESS"}
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

        return fast_app


