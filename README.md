# Personalized Disease Risk and Lifestyle Recommendation System

A Django web application that compares a user's health and lifestyle answers with patterns in the public
_Personal Key Indicators of Heart Disease_ (CDC BRFSS 2020) dataset, and gives explained, rule-based lifestyle tips.

> **Educational project. Not a medical device, not a diagnosis.** The model is trained on US survey data,
> has not been clinically validated and may be less accurate for people in India (see `docs/VIVA.md`).

## What is inside

| Path                        | Purpose                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `ml/`                       | Offline ML: `inspect_dataset.py`, `train_model.py`, `evaluate_model.py`, shared feature schema `config.py`, `predict.py` (inference) |
| `accounts/`                 | Registration, login/logout, profile, account deletion                                                                                |
| `health/`                   | Assessment form, assessment history, dashboard, database model `Assessment`                                                          |
| `prediction/`               | `services.py` (glue between Django and `ml/`), "About the model" page                                                                |
| `recommendations/engine.py` | Rule-based, explained lifestyle recommendations (independent of the ML model)                                                        |
| `templates/`, `static/`     | Bootstrap 5 UI, Chart.js dashboard                                                                                                   |
| `config/`                   | Django settings (env-based), URLs, WSGI                                                                                              |
| `data/`                     | Put the Kaggle CSV here (git-ignored)                                                                                                |
| `model_artifacts/`          | Trained `model.joblib` + `model_meta.json` are written here (git-ignored)                                                            |
| `docs/`                     | Architecture, report outline, viva Q&A, deployment guide                                                                             |

## 1. Setup (Windows PowerShell)

Requires **Python 3.11 or newer** (`python --version`).

```powershell
cd path\to\disease-risk-system
python -m venv venv
.\venv\Scripts\Activate.ps1
# if activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and replace `SECRET_KEY` with your own random value:

```powershell
python -c "from django.core.management.utils import get_random_secret_key as g; print(g())"
```

## 2. Get the dataset

1. Download from https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
2. Extract and copy **`heart_2020_cleaned.csv`** into the `data\` folder.

This project expects the 2020 file's columns (`HeartDisease, BMI, Smoking, ...`). The Kaggle page also has a
newer 2022 file with _different_ columns. The scripts stop with a clear message if the columns do not match.

Inspect it first:

```powershell
python ml\inspect_dataset.py data\heart_2020_cleaned.csv
```

## 3. Train the model

```powershell
python ml\train_model.py --quick     # fast first try (subsamples)
python ml\train_model.py             # full run (several minutes: it trains 6 configurations)
python ml\evaluate_model.py          # re-check the saved model on the held-out test split
```

It cross-validates Logistic Regression, Random Forest and Gradient Boosting (each with and without class
weighting), selects by mean cross-validated ROC-AUC, chooses recall-oriented thresholds from out-of-fold
scores, evaluates once on the untouched test set, and saves `model_artifacts/model.joblib`
and `model_meta.json`. All numbers are computed from your data; none are hard-coded.

## 4. Run the app

```powershell
python manage.py migrate
python manage.py createsuperuser     # optional, for /admin (read-only for health data)
python manage.py runserver
```

Open http://127.0.0.1:8000/ , register, and create an assessment.
If you skipped step 3 the form shows a clear "model not trained" setup message and saves nothing.

Inspect the database: `python manage.py dbshell` (SQLite), or `/admin/`, or "DB Browser for SQLite" on `db.sqlite3`.

## 5. Tests

```powershell
python manage.py test               # everything
python manage.py test health        # one app
python manage.py test ml            # ML tests
```

Tests use a tiny model built from **synthetic** data (`ml/testing_utils.py`) so they run without the Kaggle
file. The running app never uses synthetic data.

## 6. GitHub and deployment

See `docs/DEPLOYMENT.md` (GitHub setup, Render, PostgreSQL, WhiteNoise, model artifact handling, debugging).

## Design decisions (short)

- **One schema file** (`ml/config.py`) drives training, the form, validation and inference, so they cannot drift apart.
- **Preprocessing lives inside the scikit-learn `Pipeline`**: fitted on training folds only, and the same transform is applied at inference.
- **No percentage is shown.** The app shows a category (lower / borderline / elevated) and an _uncalibrated score_, because the score is not a validated personal risk.
- **`Race` is deliberately not used** (US-specific, ethically problematic, irrelevant for India).
- **Every health query is scoped to `request.user`**; other users' records return 404.
- **Data minimisation and deletion**: profile fields are optional; users can delete assessments or their entire account.
