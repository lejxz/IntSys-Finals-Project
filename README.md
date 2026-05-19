# Email Security Project (Naive Bayes)

This project detects fraud/phishing and prompt injection in email content using a theory-first Naive Bayes classifier implemented in pure Python.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── model/
│   ├── classifier.py
├── data/
│   └── dummy_dataset.csv
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## VS Code Setup

1. Open this repository in VS Code.
2. Open a terminal in VS Code.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Start the web app:

```powershell
python app.py
```

6. Open http://127.0.0.1:5000 in your browser.

