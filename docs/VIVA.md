# Viva preparation

**Q: What does your system do?**
It collects health/lifestyle answers, runs a trained classifier that measures how closely they resemble people who reported heart disease in a US survey, and shows rule-based, explained lifestyle tips. It is educational decision support, not diagnosis.

**Q: Why not just report accuracy?**
Heart disease is a minority class. A model that always says "no disease" gets high accuracy but misses every patient. So we track recall (sensitivity), the false-negative rate, precision, F1 and ROC-AUC, and read the confusion matrix.

**Q: How did you handle class imbalance without leakage?**
Class weighting (cost-sensitive learning) inside each model; no synthetic rows are created, so nothing from a validation fold can leak. Every preprocessing step is inside a scikit-learn `Pipeline`, so it is re-fitted on each training fold only.

**Q: How did you choose the model?**
Mean 5-fold cross-validated ROC-AUC on the training data across 3 algorithms x 2 imbalance strategies. The test set was used once, at the end. Then decision thresholds were set from out-of-fold predictions to reach target recall (80% "elevated", 95% "borderline").

**Q: Why not use the test set to pick the threshold?**
That would leak test information into the design and make the final test numbers optimistic.

**Q: Is the output a probability of getting heart disease?**
No. The target is "reported heart disease now" in cross-sectional survey data, so it measures similarity to existing reports, not future risk. Class weighting shifts scores upward, so they are not calibrated probabilities. We therefore show categories and a labelled score instead of a percentage. Calibration means predicted 0.2 corresponds to ~20% observed; `train_model.py` prints a calibration table so this can be checked, and even good calibration on US data would not make it valid for an individual in India.

**Q: Limitations for Indian users?**
Different population (genetics, diet, body composition, baseline rates), US race categories do not apply (so `Race` is excluded), BMI cut-offs differ (South Asians have higher cardiometabolic risk at lower BMI; our recommendation engine uses the 23 / 27.5 Asian action points), self-reported data, no labs (blood pressure, cholesterol).

**Q: Why not merge BRFSS, the Kaggle file and NHANES?**
They differ in population, years, question wording and measurement (NHANES has measured labs and exams; BRFSS is self-report). Merged features would mean different things in different rows and add hidden source bias.

**Q: Is feature importance proof of causation?**
No. It shows what the model uses to separate groups in this dataset, not what causes disease. The app never claims that changing a habit changes risk.

**Q: Why rule-based recommendations?**
They are transparent, auditable and safe, and can use factors absent from the training data (diet) without pretending the model uses them. Every tip carries its reason and none prescribes medicine.

**Q: How do you protect privacy?**
Login required; every query filters by `request.user` (other users' record IDs return 404); CSRF; server-side validation; hashed passwords; secrets in env vars; HTTPS/HSTS/secure cookies in production; read-only admin for health data; users can delete assessments or their whole account; no data is sent to external AI services; minimal optional profile.

**Q: How does Django use the model without retraining?**
`ml/artifacts.py` loads `model.joblib` once and caches it (keyed by file modification time). Training happens offline in `ml/train_model.py`.

**Q: What happens if the model file is missing?**
A `ModelNotAvailableError` is raised and the user sees a setup message; nothing is predicted or saved (tested).

**Q: What would you do next?**
Validate on Indian cohort data, add clinically measured features, calibrate and externally validate before any real use, and get clinical review.
