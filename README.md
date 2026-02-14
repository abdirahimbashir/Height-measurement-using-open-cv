# Advanced Height Measurement

Small single-file Python script that uses OpenCV and MediaPipe to estimate a person's height from webcam video.

Quick start (Windows):

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the script from repository root:

```powershell
python "advanced height mesurement.py"
```

Notes:
- Stand 1–2 meters from the camera and ensure good lighting during calibration.
- Press `r` to reset calibration and `q` to quit.
