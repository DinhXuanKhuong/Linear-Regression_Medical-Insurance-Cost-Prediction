# Medical Insurance Cost Prediction

This repository contains a linear / polynomial regression pipeline and a small Streamlit app that predicts medical insurance charges from user inputs. The trained pipeline is saved as `poly_regression_pipeline.joblib` and the Streamlit front-end is implemented in `app.py`.

## What you'll find

- `app.py` — Streamlit app for interactive predictions.
- `poly_regression_pipeline.joblib` — Trained scikit-learn pipeline used by the app.
- `finally_code.ipynb` — Notebook used for EDA, model training and experiments.
- `data/medical_insurance.csv` — Original dataset used for training (optional for running the app).

## Requirements

- Python 3.8+
- The project uses these Python packages: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

I included a `requirements.txt` in the repository so you can install the dependencies with pip.

## Setup (recommended)

1. Create and activate a virtual environment (Linux / macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Streamlit app

Start the app with:

```bash
streamlit run app.py
```

This will open a local web page where you can enter age, sex, BMI, number of children, smoker status and region. The app uses the pipeline saved in `poly_regression_pipeline.joblib` to estimate insurance charges.

If you prefer to run without Streamlit (for debugging), you can import `app.py` or run small snippets manually, but the Streamlit UI is the recommended way to use this project.

## Notes & troubleshooting

- Ensure `poly_regression_pipeline.joblib` is present in the repository root. The app loads this file at startup.
- If you get ModuleNotFoundError for a package, confirm your virtual environment is active and run:

```bash
pip install <missing-package>
```

- If the app raises an error when predicting, check that the pipeline file was created with compatible scikit-learn versions. If you retrain the model locally using `finally_code.ipynb`, save the pipeline again with `joblib.dump(...)`.

## Development / Re-training

If you want to retrain or modify the model, open `finally_code.ipynb`. After training, export the pipeline with:

```python
from joblib import dump
dump(pipeline, "poly_regression_pipeline.joblib")
```

Place the file in the repository root so `app.py` can load it.


## License

See `LICENSE` in the repository root.


