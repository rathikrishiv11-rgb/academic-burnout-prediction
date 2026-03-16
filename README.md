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

## Results

| Model               | Test Accuracy | Test Recall (At-Risk) | ROC-AUC | CV Recall Mean | CV Recall Std |
|---------------------|--------------|----------------------|---------|----------------|---------------|
| Logistic Regression | 65.8%        | 51.9%                | 0.714   | 54.6%          | ±11.3%        |
| Decision Tree       | 73.4%        | 59.3%                | 0.664   | 45.7%          | ±14.2%        |
| Random Forest       | 70.9%        | 29.6%                | 0.703   | 25.5%          | ±6.9%         |

**Selected model: Logistic Regression**

Logistic Regression was selected as the most reliable model based on:
- Highest ROC-AUC (0.714) — best overall discrimination ability
- Highest cross-validated recall (54.6%) — most consistent on unseen data
- Lowest CV variance (±11.3%) — most stable across folds

Decision Tree achieved higher single test-split recall (59.3%) but showed
high cross-validation variance (±14.2%), suggesting overfitting to the 
specific test split rather than genuine generalisation.

Random Forest underperformed on recall (29.6%) despite class_weight="balanced",
likely due to poor probability calibration on the small dataset 
(395 rows, 27 at-risk students in test set).

**Key lesson:** Accuracy is a misleading metric for imbalanced classification.
Random Forest achieved 70.9% accuracy while missing 70% of at-risk students —
failing the core purpose of the system. Cross-validated recall on the minority
class is the only metric that matters here.

---

## Technologies
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Pipeline, StratifiedKFold, RandomForestClassifier)

---

## Future Work
- SMOTE-based oversampling to address class imbalance more aggressively
- XGBoost with hyperparameter tuning as an additional ensemble baseline
- Prediction threshold optimisation based on false negative cost analysis
- Validation on a larger, multi-institution dataset
- Replacement of grade-proxy label with a validated academic stress instrument
