# Copilot instructions for Advanced Height Measurement

This repository is a small, single-script Python project that uses OpenCV and MediaPipe to estimate a person's height from webcam video. Keep changes minimal and explicit; rely on the main script for concrete examples.

- **Entrypoint**: [advanced height mesurement.py](advanced%20hight%20measurement/advanced%20height%20mesurement.py)
- **Run (Windows / general)**: `python "advanced height mesurement.py"` from the repository root.

Key patterns and touchpoints
- The project is procedural and single-file. Avoid large refactors without asking the maintainer.
- Camera discovery: `init_camera()` tries multiple indices and platform-specific OpenCV backends (`cv.CAP_DSHOW`, `cv.CAP_MSMF`, `cv.CAP_V4L2`, `cv.CAP_AVFOUNDATION`). See `init_camera()` for how cameras are probed and properties are set (frame size, FPS, buffer).
- Calibration: The script uses a constant `FACE_HEIGHT_CM = 20` as a reference and maintains a `px2cm_history` deque (maxlen=30). Calibration is considered complete after 15 good samples. See the robust averaging (IQR) code path for outlier handling.
- Pose & face detection: MediaPipe `FaceMesh` and `Pose` are initialized near the top. Pose landmarks used: shoulders (11/12) and hips (23/24). Minimum visibility threshold: `min_visibility = 0.6`.
- Height estimation: Uses `nose_to_shoulder` and `shoulder_to_hip` converted using the calibrated px→cm ratio and then combined with hardcoded anthropometric multipliers. Recent estimates are smoothed via `height_history` (deque, median used).
- Controls: Press `q` to quit, `r` to reset calibration. TTS is provided via `pyttsx3` with a fallback `print()` if initialization fails.

Dependencies (discoverable in the script)
- `opencv-python` (`cv2`)
- `mediapipe`
- `numpy`
- `pyttsx3`

Suggested agent behaviour when editing
- Preserve the single-file layout unless asked to modularize. If splitting, keep behavior identical and update README and run instructions.
- When changing numeric thresholds (visibility, face height, multipliers), document why and include before/after sample outputs or metrics.
- Camera handling must remain robust across platforms; add tests or try/catch where hardware access is modified.
- Keep user-facing text and key instructions (stand 1-2 meters, lighting) unchanged unless improving clarity.

Examples to reference in edits
- Camera probing and backend list: `init_camera()` in the main file.
- Calibration buffering and IQR filtering: `px2cm_history` usage and filtering logic.
- Pose-based measurement and landmarks: landmarks indices 4 (nose), 11/12, 23/24 and `min_visibility` check.

If anything in this document is unclear or you'd like more details (for example, expected failure modes on specific webcams or suggested tests for calibration stability), tell me which part to expand and I'll update this file.
