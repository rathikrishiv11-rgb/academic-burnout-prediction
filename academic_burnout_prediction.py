import pandas as pd

df = pd.read_csv("student-mat.csv", sep=';')

# print(df.head())
# print(df.shape) 
# print(df[["G1", "G2", "G3"]].head())

df["at_risk"] = ((df.G3<10) | ((df.G3<df.G2) & (df.G2<df.G1))).astype(int) 
# print(df["at_risk"].value_counts)

df.drop(columns=["G3"], inplace=True)
# print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x="at_risk", y="absences", data=df)
plt.title("Absences vs Academic Burnout Risk")
plt.show()
#Students flagged as at-risk tend to have higher absences, indicating disengagement.

sns.countplot(x="studytime", hue="at_risk", data=df)
plt.title("Study Time Distribution by Burnout Risk")
plt.show()
#Students classified as at-risk are more concentrated in lower and moderate study time categories, suggesting that reduced academic engagement is associated with a higher likelihood of burnout or performance decline.

sns.countplot(x="failures", hue="at_risk", data=df)
plt.title("Past Failures vs Academic Burnout Risk")
plt.show()
#Students with previous academic failures show a significantly higher likelihood of being classified as at-risk.

sns.countplot(x="famsup", hue="at_risk", data=df)
plt.title("Family Support vs Academic Burnout Risk")
plt.show()
#Although a higher number of at-risk students report having family support, this is likely due to the overall dominance of this category in the dataset rather than a direct causal relationship. Proportional analysis would be required for deeper insight.

df["total_alcohol"] = df["Dalc"] + df["Walc"]
sns.boxplot(x="at_risk", y="total_alcohol", data=df)
plt.title("Alcohol Consumption vs Academic Burnout Risk")
plt.show()
#Alcohol consumption shows similar distributions across both at-risk and non–at-risk groups, indicating it may not be a strong standalone predictor of academic burnout in this dataset.

# print(df["at_risk"].value_counts()) 

df["high_absence_flag"] = (df["absences"] > df["absences"].median()).astype(int)
df["low_study_flag"] = (df["studytime"] <= 2).astype(int)

# print(df[["total_alcohol", "high_absence_flag", "low_study_flag"]].head())
#Feature Engineering:
#total_alcohol captures combined lifestyle behavior
#high_absence_flag identifies frequent absenteeism
#low_study_flag represents reduced academic engagement
df.drop(columns=["Dalc", "Walc"], inplace=True)

from sklearn.model_selection import train_test_split
X = df.drop(columns=["at_risk"])
Y = df["at_risk"]

X.select_dtypes(include="object").columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42, stratify=Y)
# print(X_train.shape)
# print(X_test.shape)

# print(Y_train.value_counts(normalize=True)) 
# print(Y_test.value_counts(normalize=True)) 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

log_reg.fit(X_train, Y_train)
Y_pred_lr = log_reg.predict(X_test)

print("Logistic Regression Accuracy : ", accuracy_score(Y_test, Y_pred_lr))
print(confusion_matrix(Y_test, Y_pred_lr))
print(classification_report(Y_test, Y_pred_lr))
# Observation:
# Logistic Regression demonstrates strong baseline performance with high accuracy and balanced precision–recall. While it is not the top-performing model, its interpretability and stability make it a reliable reference model for academic burnout prediction.

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(Y_test, Y_pred_dt))
print(confusion_matrix(Y_test, Y_pred_dt))
print(classification_report(Y_test, Y_pred_dt))
# Observation:
# Decision Tree improves recall for at-risk students compared to Logistic Regression but shows lower overall accuracy and precision. While it captures non-linear relationships, it introduces more false positives, reducing its reliability as a standalone model.

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(Y_test, Y_pred_rf))
print(confusion_matrix(Y_test, Y_pred_rf))
print(classification_report(Y_test, Y_pred_rf))
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nTop 10 Feature Importances (Random Forest):")
print(importances.head(10))

# Observation:

# Absences, past failures, and study engagement-related features contribute most significantly to burnout risk prediction.
# Random Forest achieves the highest accuracy and recall for at-risk students, indicating strong performance in identifying potential burnout cases. This suggests that ensemble methods are effective when the target variable is carefully defined with minimal label noise.

results = {
    "Model" : ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy" : [
        accuracy_score(Y_test, Y_pred_lr),
        accuracy_score(Y_test, Y_pred_dt),
        accuracy_score(Y_test, Y_pred_rf)
    ]
}

results_df = pd.DataFrame(results)
print(results_df)

from sklearn.metrics import recall_score
print("\nRecall for 'at risk' students (class = 1) : ")
print("Logistic Regression : ", recall_score(Y_test, Y_pred_lr))
print("Decision Tree : ", recall_score(Y_test, Y_pred_dt))
print("Random Forest : ", recall_score(Y_test, Y_pred_rf))
# Observation:
# Random Forest achieves the highest recall for at-risk students, while Logistic Regression remains competitive with slightly lower recall but higher interpretability. Model choice depends on whether sensitivity or simplicity is prioritized.

# Final Model Selection:
# Random Forest was selected as the final model as it achieved the highest overall accuracy and recall for at-risk students. Logistic Regression also performed strongly and remains a highly interpretable baseline. The results demonstrate that ensemble methods can better capture complex behavioral patterns associated with academic burnout when the target variable is carefully defined and label noise is minimized.

# CONCLUSION
# This project focused on early identification of academic burnout and performance decline using behavioral and academic indicators. Exploratory data analysis revealed that high absenteeism, low study engagement, and past academic failures are strongly associated with burnout risk.

# Multiple machine learning models were evaluated, including Logistic Regression, Decision Tree, and Random Forest. Random Forest emerged as the best-performing model, achieving the highest accuracy and recall for at-risk students, while Logistic Regression provided a strong and interpretable baseline.

# The study highlights the importance of careful target definition and feature engineering in supervised learning tasks and demonstrates how data-driven approaches can support proactive academic intervention strategies.

# The risk definition was carefully designed to balance sensitivity and realism, ensuring that only consistent performance decline or academic failure was classified as burnout risk.

# The proposed approach demonstrates how data-driven insights can support timely academic interventions, enabling educators and institutions to proactively identify and support students at risk of burnout.Improved performance after refining the risk definition highlights the importance of careful target engineering in supervised learning tasks.