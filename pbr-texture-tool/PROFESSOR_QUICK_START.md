# Quick Start for Local Evaluation

## Recommended steps

1. Extract the project ZIP to a local folder.
2. Make sure the model files are already inside:
   - `models/deeppbr/`
   - `models/cyclegan/`
3. Double-click `setup_env.bat` and wait until installation finishes.
4. Double-click `run_app.bat`.
5. A local Streamlit address should open in the browser.
6. Upload an image and run the pipeline.

## Important notes

- Recommended operating system: **Windows 10/11**
- Recommended Python version: **Python 3.12.x**
- CPU execution is supported
- The first run may take longer because dependencies must be installed
- Very high-resolution images may take more time to process

## Minimal folder check

Expected key files and folders:

```text
app/
models/deeppbr/
models/cyclegan/
app.py
requirements.txt
setup_env.bat
run_app.bat
```

## If the app does not start

Check the following:

- Python 3.12 is installed
- The model files are placed in the correct folders
- `setup_env.bat` finished without errors
- The folder path does not contain unusual permission restrictions
