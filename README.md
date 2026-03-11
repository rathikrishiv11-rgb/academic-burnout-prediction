# Academic Burnout & Performance Risk Prediction

## Overview
This project predicts which students are at risk of academic performance decline using behavioral and academic indicators from the UCI Student Performance Dataset. The model identifies students who may need early intervention based on attendance, study habits, and past academic history.

**Note on framing:** The term "burnout" in this project refers to grade-based academic risk (failure or consistent decline), not clinically validated burnout. This is a dataset constraint, not a modeling choice.

---

## Dataset
**Source:** UCI Student Performance Dataset (Math subject)  
**Size:** 395 students, 30+ features  
**Link:** https://archive.ics.uci.edu/ml/datasets/student+performance

Key features used:
- Absences, past failures, study time
- Family support, school support
- Lifestyle indicators (alcohol consumption)

**Dropped features:** G1, G2, G3 — these directly encode the target variable and would cause data leakage if kept as features.

---

## Problem Definition
A student is labeled **at_risk = 1** if:
- Final grade G3 < 10 (failing threshold), OR
- Consistent decline across all grading periods: G1 > G2 > G3

This definition was chosen to capture both outright failure and progressive disengagement.

---

## Methodology

### Feature Engineering
- `total_alcohol` — combined weekday + weekend alcohol consumption
- `high_absence_flag` — 1 if absences exceed dataset median
- `low_study_flag` — 1 if study time ≤ 2 hours/week

### Modeling
Three models were trained and compared:
- Logistic Regression (interpretable baseline)
- Decision Tree (non-linear patterns)
- Random Forest (ensemble method)

All models used `class_weight="balanced"` to handle class imbalance.  
Evaluation used 5-fold Stratified Cross-Validation in addition to a held-out test set.

---

## Results

| Model | Test Accuracy | Test Recall (at-risk) | ROC-AUC | CV Recall (mean ± std) |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |

> **Replace the — values with your actual printed output after running the script.**

**Primary metric: Recall for at-risk students.**  
In this problem, missing a student who is actually at risk (false negative) is more costly than a false alarm. Recall is therefore prioritized over accuracy.

---

## Limitations
1. **Label validity:** `at_risk` is derived from grades, not a validated burnout measure. The model predicts academic performance risk, not psychological burnout.
2. **Dataset size:** 395 rows is small. Cross-validation variance is high — results are indicative, not reliable for real deployment.
3. **Leakage prevention:** G1 and G2 were explicitly dropped to prevent target leakage, since they are used in the label definition.
4. **Generalizability:** Data is from Portuguese secondary schools. Results should not be generalized to other populations without re-validation.
5. **Threshold:** Default 0.5 prediction threshold is used. In real deployment, this would be tuned based on the cost tradeoff between false negatives and false positives.

---

## Technologies
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Pipeline, StratifiedKFold, RandomForestClassifier)

---

## What I Would Improve With More Time
- Experiment with XGBoost and compare against Random Forest
- Tune the prediction threshold based on false negative cost
- Collect a larger, more diverse dataset
- Replace grade-proxy label with a validated academic stress survey
