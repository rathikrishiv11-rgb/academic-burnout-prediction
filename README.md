# Early Academic Burnout & Performance Decline Prediction

## Overview
This project focuses on early identification of students at risk of academic
burnout and performance decline using behavioral and academic indicators.
The objective is to build a machine learning model that can help educators
and institutions proactively identify students who may require early
academic intervention.

---

## Dataset
The project uses the **UCI Student Performance Dataset (Math subject)**, which
contains academic, behavioral, and demographic information of secondary
school students.

Key attributes include:
- Academic performance (G1, G2)
- Absences and past failures
- Study habits and lifestyle indicators
- Family and school support factors

---

## Problem Formulation
A student is classified as **at-risk** if:
- The final grade is below a passing threshold, or
- There is a consistent decline in academic performance across grading periods

This definition was carefully designed to reduce label noise and ensure that
only meaningful academic risk patterns are captured.

---

## Methodology

### Exploratory Data Analysis (EDA)
EDA revealed that:
- Higher absenteeism is strongly associated with burnout risk
- Lower study engagement correlates with increased academic decline
- Past academic failures significantly increase burnout probability
- Lifestyle factors such as alcohol consumption show weaker predictive power

### Feature Engineering
Key engineered features include:
- Combined alcohol consumption indicator
- High absence flag based on dataset median
- Low study engagement flag

These features help capture behavioral patterns rather than relying only on
raw academic scores.

---

## Modeling Approach
The following models were evaluated:
- Logistic Regression (baseline, interpretable)
- Decision Tree (non-linear modeling)
- Random Forest (ensemble method)

Models were compared using:
- Accuracy
- Recall (for at-risk students)
- Precision and F1-score

---

## Results
Random Forest achieved the highest overall performance, including the best
accuracy and recall for identifying at-risk students. Logistic Regression
also performed strongly and served as a reliable and interpretable baseline.

The results highlight the importance of clean target definition and feature
engineering in improving model performance.

---

## Conclusion
This project demonstrates how machine learning can be applied to educational
data to support early academic intervention strategies. By identifying
students at risk of burnout at an early stage, institutions can take
proactive steps to improve student outcomes.

---

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn