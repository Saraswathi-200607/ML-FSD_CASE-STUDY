# Project report outline

1. **Title page, certificate, acknowledgements, abstract**
2. **Introduction**: lifestyle diseases, motivation, objectives, scope, what the system is *not* (not a diagnosis)
3. **Literature survey**: ML for cardiovascular risk (e.g. Framingham score as the clinical benchmark), BRFSS/NHANES-based studies, fairness and population-shift issues
4. **Problem statement and requirements**: functional (auth, assessment, prediction, recommendations, history) and non-functional (security, privacy, usability)
5. **System design**: architecture diagram (`docs/ARCHITECTURE.md`), module diagram, ER diagram (User, Profile, Assessment), request-flow/sequence diagram, UI wireframes
6. **Dataset**: source, size, columns and target (take from *your* `inspect_dataset.py` output), class imbalance, missing values, duplicates, why `Race` was excluded, why datasets were not merged, limitations for Indian users
7. **Machine-learning methodology**: leakage-safe preprocessing in a Pipeline, stratified split, cross-validation, algorithms (LR, RF, GB), class-weighting, metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), why recall/false negatives matter, threshold selection from out-of-fold scores, calibration check
8. **Results**: the comparison table from `train_model.py` (your real output), selected model and justification, confusion matrix, calibration table, error analysis
9. **Recommendation engine**: rules table (factor -> rule -> reason), safety principles
10. **Implementation**: technology stack, key code listings (settings, models, forms, views, predict), screenshots
11. **Security and privacy**: authentication, CSRF, validation, per-user scoping, secrets, HTTPS, data minimisation, retention, deletion
12. **Testing**: test plan, test cases and results (`python manage.py test`), access-control tests
13. **Deployment**: Render, PostgreSQL, WhiteNoise, model-artifact handling
14. **Limitations and ethical considerations**: US data, self-report, cross-sectional, no clinical validation, uncalibrated scores
15. **Future work**: Indian datasets, clinical validation, calibration, NHANES lab features (separate model), explainability (SHAP) with causation caveats
16. **Conclusion, references, appendix** (full code, user manual)
