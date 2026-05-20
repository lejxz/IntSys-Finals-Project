# Email Security Project (Naive Bayes)

This project detects fraud/phishing and prompt injection in email content using a Naive Bayes classifier implemented in pure Python.

## Project Structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── model/
│   └── classifier.py
└── data/
    └── dummy_dataset.csv
```

Expected outcome: a single dataset and two Python files (`streamlit_app.py` and `model/classifier.py`).

## Quick Start

1. Create and activate a virtual environment (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit (typically http://localhost:8501).
