"""IndexTTS2 Modal deployment — expressive TTS with Whisper word timestamps.

Exposes a single HTTP endpoint:
    POST /synthesize/sync
        Request:  {"text": str, "voice_id": "default", "reference_audio_b64": str|null}
        Response: {"audio_b64": str, "duration_seconds": float, "word_events": [...]}

The word_events list format matches the Kokoro path:
    [{"word": str, "start_time": float, "end_time": float}]

The FastAPI web server runs ON the same A10G GPU container as the model
(via @modal.asgi_app() as a class method). Every request hits an already-warm
container — no cross-container spawning, no per-request cold starts.

── Reference voice setup ─────────────────────────────────────────────────────
The "default" voice_id uses the first bundled voice found at startup.
IndexTTS-2 ships with voice_01–voice_12 and emo_sad/emo_hate in examples/.

Per-character voice cloning: pass reference_audio_b64 (base64 WAV) in the
request; the endpoint will use it instead of the bundled reference.

── Modal GPU ─────────────────────────────────────────────────────────────────
GPU: A10G (24 GB VRAM — IndexTTS2 requires ~8 GB; Whisper base ~0.1 GB).
Torch: 2.7.0 (Modal resolves CUDA wheels; transformers==4.52.1 for OffloadedCache).
Cold start: ~90–120 s (model download on first start, cached in Volume).
Per-synthesis: ~20–40 s for a typical 5–10 second utterance.
Scaledown: 5 minutes idle → container shuts down.
Credit burn: A10G ~$0.20/hr → ~$0.017/pipeline (10 chunks, ~5 min total).
At $30/account/month free credits: ≈ 1,700 pipelines per account.
"""
import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

# ── Model identifiers ──────────────────────────────────────────────────────────
MODEL_ID = "IndexTeam/IndexTTS-2"
APP_NAME = "indextts2-expressive-tts"

# ── Container image ────────────────────────────────────────────────────────────
# IndexTTS2 requires cloning index-tts/index-tts and using indextts.infer_v2
# (IndexTTS2 class). The PyPI indextts package and infer.py are the v1 code;
# they do not have emo_condition_module support required by the HF model's
# config.yaml. We use --no-deps to keep torch at the pinned version.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    # Single uv_pip_install so uv resolves the entire dep graph together.
    # transformers==4.52.1 is pinned exactly (indextts2's requirement) — the pin
    # wins over any looser constraint from WeTextProcessing/openai-whisper.
    # WeTextProcessing provides the `tn` module used by indextts/utils/front.py.
    .uv_pip_install(
        # Core ML stack — pinned to match ltx2 pattern; Modal resolves CUDA wheels
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "accelerate>=1.6.0",
        "transformers==4.52.1",     # exact version required by indextts2
        # Inference / API
        "openai-whisper>=20231117",
        "huggingface-hub>=0.23.0",
        "fastapi[standard]",
        "pydantic>=2.0",
        # indextts2 deps — all explicit because we use --no-deps below
        "omegaconf",
        "einops",
        "matplotlib",
        "scipy",
        "librosa",
        "soundfile",
        "inflect",              # English number-to-words
        "pypinyin",             # Chinese pinyin
        "cn2an",                # Chinese number conversion
        "jieba",                # Chinese tokeniser
        "WeTextProcessing",     # provides the `tn` module used by front.py TextNormalizer
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
        "modelscope",           # required by infer_v2.py for Qwen emotion model loading
    )
    # Clone IndexTTS repo (code only — skip LFS audio files) and install the
    # package with --no-deps so we don't downgrade the pinned torch/transformers.
    # GIT_LFS_SKIP_SMUDGE=1: the repo stores example voices via Git LFS and
    # the public LFS budget is frequently exhausted; we only need the Python code.
    .run_commands(
        "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git /opt/indextts",
        "pip install --no-deps -e /opt/indextts",
    )
    .env({"HF_HOME": "/models", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App(APP_NAME, secrets=[modal.Secret.from_name("HF_TOKEN")])

VOLUME_NAME = "indextts2-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")

MODEL_VOLUME_NAME = "indextts2-model"
model_vol = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")

MINUTES = 60  # seconds

DEFAULT_REFERENCE_PATH = MODEL_PATH / "indextts2" / "default_reference.wav"
MODEL_DIR = MODEL_PATH / "indextts2"


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


def _measure_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-c:a", "libmp3lame", "-q:a", "2",
            str(mp3_path),
        ],
        check=True,
    )


@app.cls(
    image=image,
    volumes={OUTPUTS_PATH: outputs, MODEL_PATH: model_vol},
    gpu="A10G",
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
)
class IndexTTS2:
    @modal.enter()
    def load_model(self):
        """Download and load IndexTTS2 + Whisper on container startup.

        Models are cached in the Modal Volume so subsequent cold starts skip
        the download step.
        """
        from huggingface_hub import snapshot_download
        import whisper

        token = _get_hf_token()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Download IndexTTS2 model weights (cached in Volume after first run)
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(MODEL_PATH),
            local_dir=str(MODEL_DIR),
            token=token,
        )

        # ── Discover reference voices ─────────────────────────────────────────
        # The GitHub repo example voices are NOT available (LFS budget exhausted
        # — GIT_LFS_SKIP_SMUDGE=1 means only LFS pointer stubs were checked out).
        # We scan the HF model download dir for any audio files there, and also
        # pick up voices manually uploaded to the Volume at MODEL_DIR.
        # Skip tiny files (< 1 KB) — LFS pointer stubs are ~130 bytes.
        _WEIGHT_DIRS = {"checkpoints", "ckpt", "weights", "model", "bigvgan", ".cache"}
        voice_map: dict[str, Path] = {}
        for ext in ("*.wav", "*.mp3"):
            for f in sorted(MODEL_DIR.rglob(ext)):
                rel = f.relative_to(MODEL_DIR)
                if rel.parts[0].lower() in _WEIGHT_DIRS:
                    continue
                if f.stat().st_size < 1024:
                    continue
                vid = f.stem
                if vid not in voice_map:
                    voice_map[vid] = f
        self.voice_map: dict[str, Path] = voice_map
        voice_ids = list(voice_map.keys())
        print(f"[IndexTTS2] {len(voice_map)} reference voice(s) found: {voice_ids or '(none)'}")
        if not voice_map:
            print(
                f"[IndexTTS2] No reference voices found. Upload a PCM WAV to "
                f"the '{MODEL_VOLUME_NAME}' Volume at {DEFAULT_REFERENCE_PATH} "
                "or pass reference_audio_b64 in synthesis requests."
            )

        # ── Load IndexTTS2 ────────────────────────────────────────────────────
        # Use infer_v2 (IndexTTS2 class) — infer.py is the v1 code and does not
        # support emo_condition_module which the HF model's config.yaml requires.
        from indextts.infer_v2 import IndexTTS2 as _IndexTTS2
        cfg_files = sorted(MODEL_DIR.rglob("config.yaml"))
        if not cfg_files:
            raise RuntimeError(
                f"config.yaml not found under {MODEL_DIR}. "
                f"Files present: {[str(p.relative_to(MODEL_DIR)) for p in sorted(MODEL_DIR.rglob('*'))[:30]]}"
            )
        self.tts = _IndexTTS2(
            model_dir=str(MODEL_DIR),
            cfg_path=str(cfg_files[0]),
        )

        # ── Load Whisper for word-level timestamps ────────────────────────────
        self.whisper_model = whisper.load_model("base")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_reference(
        self,
        voice_id: str,
        tmp_path: Path,
        reference_audio_b64: str | None,
    ) -> Path:
        """Return the reference audio path to use for synthesis.

        Priority:
          1. Caller-supplied reference_audio_b64 (inline voice cloning)
          2. Named voice from bundled voice_map (IndexTTS2 ships with samples)
          3. "default" → first bundled voice available
          4. Manually uploaded default at DEFAULT_REFERENCE_PATH
        """
        if reference_audio_b64:
            ref_path = tmp_path / "reference.wav"
            ref_path.write_bytes(base64.b64decode(reference_audio_b64))
            return ref_path

        if voice_id in self.voice_map:
            return self.voice_map[voice_id]

        if voice_id == "default":
            if self.voice_map:
                return next(iter(self.voice_map.values()))
            if DEFAULT_REFERENCE_PATH.exists():
                return DEFAULT_REFERENCE_PATH

        available = list(self.voice_map.keys())
        raise ValueError(
            f"Voice '{voice_id}' not found. "
            f"Available bundled voices: {available or '(none)'}. "
            "Pass reference_audio_b64 for inline voice cloning, or upload a WAV "
            f"to '{MODEL_VOLUME_NAME}' Volume at {DEFAULT_REFERENCE_PATH}."
        )

    def _ensure_wav(self, audio_path: Path, tmp_path: Path) -> Path:
        """Convert any audio file to 16-bit mono PCM WAV that librosa can read.

        The bundled example voices may be in non-standard formats (e.g. MP3
        with a .wav extension) that soundfile cannot open directly. ffmpeg
        normalises them to a format librosa can always load.
        """
        out = tmp_path / f"{audio_path.stem}_pcm.wav"
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(audio_path),
                "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            import sys
            print(
                f"[IndexTTS2] _ensure_wav failed for {audio_path}: "
                f"stderr={result.stderr!r}",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"ffmpeg conversion failed for {audio_path.name}: {result.stderr.strip()}"
            )
        return out

    def _do_list_voices(self) -> list[dict]:
        """Return metadata for all bundled reference voices."""
        _ROOTS = [MODEL_DIR, Path("/opt/indextts")]
        result = []
        for vid, p in sorted(self.voice_map.items()):
            rel = str(p)
            for root in _ROOTS:
                try:
                    rel = str(p.relative_to(root))
                    break
                except ValueError:
                    pass
            result.append({"voice_id": vid, "filename": p.name, "path": rel})
        return result

    def _do_synthesize(
        self,
        text: str,
        voice_id: str = "default",
        reference_audio_b64: str | None = None,
    ) -> dict:
        """Synthesize text and return MP3 audio + Whisper word events.

        Returns:
            {
                "audio_b64": str,           # base64-encoded MP3
                "duration_seconds": float,
                "word_events": [{"word": str, "start_time": float, "end_time": float}]
            }
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            ref_path = self._resolve_reference(voice_id, tmp_path, reference_audio_b64)

            # Convert reference to standard PCM WAV — bundled example voices may
            # be MP3 with a .wav extension which soundfile/librosa cannot open.
            ref_wav = self._ensure_wav(ref_path, tmp_path)

            # ── Synthesize ────────────────────────────────────────────────────
            # infer_v2 uses spk_audio_prompt (not audio_prompt) for the speaker
            # reference; emo_audio_prompt defaults to spk_audio_prompt when None.
            wav_out = tmp_path / f"output_{int(time.time())}.wav"
            self.tts.infer(
                spk_audio_prompt=str(ref_wav),
                text=text,
                output_path=str(wav_out),
            )

            # ── WAV → MP3 ─────────────────────────────────────────────────────
            mp3_out = tmp_path / "output.mp3"
            _convert_wav_to_mp3(wav_out, mp3_out)
            duration = _measure_audio_duration(mp3_out)
            audio_bytes = mp3_out.read_bytes()

            # ── Whisper word timestamps (best-effort) ─────────────────────────
            word_events: list[dict] = []
            try:
                result = self.whisper_model.transcribe(
                    str(mp3_out),
                    word_timestamps=True,
                    fp16=True,
                )
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

    # ── External Modal API (callable from Python / other Modal functions) ──────

    @modal.method()
    def list_voices(self) -> list[dict]:
        return self._do_list_voices()

    @modal.method()
    def synthesize(
        self,
        text: str,
        voice_id: str = "default",
        reference_audio_b64: str | None = None,
    ) -> dict:
        return self._do_synthesize(text, voice_id, reference_audio_b64)

    # ── Web API — runs on the SAME GPU container as the model ─────────────────
    # Using @modal.asgi_app() as a class method means the FastAPI server runs
    # inside the IndexTTS2 container. Handlers call self._do_*() directly —
    # no cross-container spawning, no extra cold starts.

    @modal.asgi_app(requires_proxy_auth=False)
    def api(self):
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel as PydanticBaseModel
        from typing import Optional

        fast_app = FastAPI(title="IndexTTS2 Expressive TTS")

        @fast_app.get("/health")
        async def health():
            return {"status": "ok"}

        @fast_app.get("/voices")
        async def list_voices_endpoint():
            """List all bundled reference voices available for synthesis."""
            try:
                voices = self._do_list_voices()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to list voices: {exc}") from exc
            return {"voices": voices, "count": len(voices)}

        class SynthesizeRequest(PydanticBaseModel):
            text: str
            voice_id: str = "default"
            reference_audio_b64: Optional[str] = None  # base64 WAV for voice cloning

        @fast_app.post("/synthesize/sync")
        async def synthesize_sync(body: SynthesizeRequest):
            """Synthesize text and return JSON with base64 audio + word events."""
            if not body.text.strip():
                raise HTTPException(status_code=422, detail="text must not be empty")
            try:
                return self._do_synthesize(body.text, body.voice_id, body.reference_audio_b64)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

        return fast_app
