# Deployment guide (GitHub + Render)

Check Render's current plan limits and prices before you start (free-tier rules, such as sleeping
services, database expiry and persistent-disk availability, change over time).

## 1. GitHub

```powershell
git init
git add .
git status          # CHECK: .env, data/*.csv, db.sqlite3, venv/ must NOT be listed
git commit -m "Initial commit"
# create an empty repo on github.com first, then:
git branch -M main
git remote add origin https://github.com/<you>/disease-risk-system.git
git push -u origin main
```

Never commit `.env`, the dataset or credentials. If you ever do, rotate the secret.

## 2. Model artifact: the important decision

Render's web-service filesystem is **ephemeral**: files created at runtime vanish on redeploy, and the
`data/` folder is not in Git, so the server cannot train the model itself. Choose one:

**Option A (simplest, good for a college demo): commit the trained model.**
Train locally, check the size (`Get-ChildItem model_artifacts`), then force-add the two files:
```powershell
git add -f model_artifacts/model.joblib model_artifacts/model_meta.json
git commit -m "Add trained model"
git push
```
Keep it small (Logistic Regression / the default RF settings are small). Make sure the scikit-learn
version in `requirements.txt` matches the one you trained with (the app refuses a mismatched model and
tells you to retrain).

**Option B: persistent disk** (paid on Render). Mount a disk at `/var/data`, set env var
`MODEL_DIR=/var/data/models`, and upload `model.joblib` and `model_meta.json` there (e.g. via Render Shell / SSH).

Larger models: store them in object storage and download at build time (not covered here).

## 3. Create the PostgreSQL database
Render dashboard -> **New -> PostgreSQL** -> create. Copy the **Internal Database URL**.

## 4. Create the web service
Render -> **New -> Web Service** -> connect the GitHub repo.

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `bash build.sh` |
| Start command | `gunicorn config.wsgi:application` |

Environment variables:

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.12.3` (or any 3.11+) |
| `DEBUG` | `False` |
| `SECRET_KEY` | long random string (generate with the command in README) |
| `DATABASE_URL` | the Internal Database URL from step 3 |
| `MODEL_DIR` | only for Option B |

`RENDER_EXTERNAL_HOSTNAME` is set automatically by Render and is added to `ALLOWED_HOSTS` and
`CSRF_TRUSTED_ORIGINS` by `config/settings.py`. For a custom domain add `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS`.

Alternatively use the included `render.yaml` as a Blueprint.

`build.sh` installs dependencies, runs `collectstatic` (WhiteNoise serves the static files) and `migrate`.

## 5. Create an admin user (optional)
Render -> your service -> **Shell**: `python manage.py createsuperuser`

## 6. Test the deployed site
1. Open the `https://<name>.onrender.com` URL (HTTP redirects to HTTPS).
2. Register, submit an assessment, open dashboard and history.
3. Open "About the model": it should show your model version and metrics (if not, the artifact is missing).
4. Register a second user and confirm they cannot see the first user's assessments.

## 7. Debugging
* **Build fails**: read the Render build log. Typical causes: Python version too old for pinned packages; a wrong build command (use `bash build.sh`).
* **`DisallowedHost` / 400**: add the hostname to `ALLOWED_HOSTS`.
* **CSRF 403 on login/forms**: add `CSRF_TRUSTED_ORIGINS=https://your-host`.
* **`SECRET_KEY environment variable is required`**: set it in Render.
* **Unstyled pages**: `collectstatic` did not run or the build command is wrong.
* **"model has not been trained yet"**: artifacts missing in `MODEL_DIR` (see step 2).
* **"trained with scikit-learn X but this server runs Y"**: retrain locally with the pinned versions and re-commit.
* **Redirect loops**: only set `SECURE_SSL_REDIRECT=False` temporarily while diagnosing.
