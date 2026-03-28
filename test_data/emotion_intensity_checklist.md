# IndexTTS2 Emotion Intensity Checklist

Use this to map LLM_SCRIPT mood vocabulary to specific intensity levels.
Run `test_intensity_sweep.py --url <modal-url> --emotion <slug>` to generate clips.

## Intensity Levels

T3–T5 from the diagnostic sweep, renamed L1–L3 for production use.

| Level | emo_weight | vector_scale | bypass_normalize | Expected use |
|-------|-----------|--------------|-----------------|--------------|
| L1    | 1.0       | 5×           | false           | Strong, natural — default for most moods |
| L2    | 1.0       | 10×          | false           | Intense, theatrical — high-energy / extreme moods |
| L3    | 1.0       | 1×           | true (raw vec)  | Maximum / unfiltered — reserved for future extreme vocab |

---

## sad
Base vector: `[0, 0, 1, 0, 0, 0, 0, 0]`
Text: *"She walked away and never looked back. It was over."*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `sad_L1.mp3` | | |
| L2 (10×)    | `sad_L2.mp3` | | |
| L3 (bypass) | `sad_L3.mp3` | | |

---

## angry
Base vector: `[0, 1, 0, 0, 0, 0, 0, 0]`
Text: *"He lied. He sat there, looked her in the face, and lied."*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `angry_L1.mp3` | | |
| L2 (10×)    | `angry_L2.mp3` | | |
| L3 (bypass) | `angry_L3.mp3` | | |

---

## fearful
Base vector: `[0, 0, 0, 1, 0, 0, 0, 0]`
Text: *"Nobody knew what was about to happen next. The silence was unbearable."*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `fearful_L1.mp3` | | |
| L2 (10×)    | `fearful_L2.mp3` | | |
| L3 (bypass) | `fearful_L3.mp3` | | |

---

## excited
Base vector: `[1, 0, 0, 0, 0, 0, 0, 0]`
Text: *"She got the rose! I cannot believe it — this is insane!"*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `excited_L1.mp3` | | |
| L2 (10×)    | `excited_L2.mp3` | | |
| L3 (bypass) | `excited_L3.mp3` | | |

---

## melancholic
Base vector: `[0, 0, 0.3, 0, 0, 1, 0, 0]`
Text: *"She had given everything to this villa, and in the end, it gave her nothing back."*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `melancholic_L1.mp3` | | |
| L2 (10×)    | `melancholic_L2.mp3` | | |
| L3 (bypass) | `melancholic_L3.mp3` | | |

---

## calm
Base vector: `[0, 0, 0, 0, 0, 0, 0, 1]`
Text: *"And with that, another chapter of Love Island comes to a close."*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `calm_L1.mp3` | | |
| L2 (10×)    | `calm_L2.mp3` | | |
| L3 (bypass) | `calm_L3.mp3` | | |

---

## surprised
Base vector: `[0, 0, 0, 0, 0, 0, 1, 0]`
Text: *"She got the rose! I cannot believe it — this is insane!"*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `surprised_L1.mp3` | | |
| L2 (10×)    | `surprised_L2.mp3` | | |
| L3 (bypass) | `surprised_L3.mp3` | | |

---

## disgusted
Base vector: `[0, 0.2, 0, 0, 1, 0, 0, 0]`
Text: *"Blake had been lying from the very beginning."*

| Level | File | Notes | Suggested LLM vocab |
|-------|------|-------|---------------------|
| L1 (5×)     | `disgusted_L1.mp3` | | |
| L2 (10×)    | `disgusted_L2.mp3` | | |
| L3 (bypass) | `disgusted_L3.mp3` | | |

---

## Intensity Vocabulary Map

Current assignments in `emotion_adder.py` (`_MOOD_PRESETS`). Update after listening.

| LLM mood word | Base emotion | Level | Notes |
|---------------|-------------|-------|-------|
| `cheerful`    | excited     | L1    | |
| `upbeat`      | excited     | L1    | |
| `comedic`     | excited     | L1    | |
| `romantic`    | excited     | L1    | gentle — L1 intentional |
| `excited`     | excited     | L2    | |
| `triumphant`  | excited     | L2    | |
| `intense`     | angry       | L1    | |
| `angry`       | angry       | L2    | |
| `dramatic`    | blended     | L2    | sad+angry+melancholic+surprised |
| `chaotic`     | blended     | L3    | peak energy — loss of control |
| `desperate`   | blended     | L3    | peak angry+fear blend |
| `solemn`      | melancholic | L1    | |
| `melancholic` | melancholic | L1    | |
| `sad`         | sad         | L2    | |
| `tragic`      | sad         | L3    | peak sadness |
| `mysterious`  | fearful     | L1    | subtle — L1 intentional |
| `eerie`       | fearful     | L1    | |
| `haunting`    | fearful     | L1    | |
| `tense`       | fearful     | L1    | |
| `fearful`     | fearful     | L2    | |
| `disgusted`   | disgusted   | L2    | |
| `surprised`   | surprised   | L1    | |
| `shocking`    | surprised   | L3    | peak surprise — jaw-drop |
| `surreal`     | blended     | L1    | subtle — L1 intentional |
| `calm`        | calm        | L1    | |
| `neutral`     | —           | —     | SPEAKER mode, no vector |

---

## Notes
- L3 (bypass) skips `normalize_emo_vec` entirely — raw float values go directly to `infer()`. May be unstable for blended multi-dim vectors. Recommended only for pure single-dim emotions.
- `calm` and `neutral` at L2+ may sound robotic — L1 is the ceiling for those.
- Blended moods (`dramatic`, `chaotic`, `tense`) use their actual multi-dim base vectors from `_MOOD_PRESETS`, not the pure single-dim vectors tested in this sweep. Sweep results are a guide, not a direct match.
- Once all emotions are swept, revisit the vocabulary map — some assignments may need moving up/down a level based on listening results.
