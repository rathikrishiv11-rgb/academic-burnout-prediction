import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

# 1. LOAD DATA

df = pd.read_csv("student-mat.csv", sep=';')
print("Dataset shape:", df.shape)
print(df.head())

# 2. TARGET VARIABLE DEFINITION

df["at_risk"] = (
    (df["G3"] < 10) |
    ((df["G3"] < df["G2"]) & (df["G2"] < df["G1"]))
).astype(int)

print("\nClass Distribution ")
print(df["at_risk"].value_counts())
print(df["at_risk"].value_counts(normalize=True).round(3))

df.drop(columns=["G3"], inplace=True)

# 3. EXPLORATORY DATA ANALYSIS (EDA)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("EDA: Behavioral Indicators vs At-Risk Label", fontsize=14)

sns.boxplot(x="at_risk", y="absences", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Absences vs At-Risk")

sns.countplot(x="studytime", hue="at_risk", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Study Time vs At-Risk")

sns.countplot(x="failures", hue="at_risk", data=df, ax=axes[0, 2])
axes[0, 2].set_title("Past Failures vs At-Risk")

sns.countplot(x="famsup", hue="at_risk", data=df, ax=axes[1, 0])
axes[1, 0].set_title("Family Support vs At-Risk")

df["total_alcohol"] = df["Dalc"] + df["Walc"]
sns.boxplot(x="at_risk", y="total_alcohol", data=df, ax=axes[1, 1])
axes[1, 1].set_title("Total Alcohol vs At-Risk")

sns.scatterplot(x="G1", y="G2", hue="at_risk", data=df, ax=axes[1, 2], alpha=0.6)
axes[1, 2].set_title("G1 vs G2 by At-Risk")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()
print("EDA plots saved.")

# 4. FEATURE ENGINEERING

df["high_absence_flag"] = (df["absences"] > df["absences"].median()).astype(int)
df["low_study_flag"] = (df["studytime"] <= 2).astype(int)

df.drop(columns=["Dalc", "Walc", "G1", "G2"], inplace=True)

print("\nFinal feature set shape after engineering:", df.drop(columns=["at_risk"]).shape)

# 5. PREPROCESSING

X = df.drop(columns=["at_risk"])
y = df["at_risk"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
print("Test class balance:\n", y_test.value_counts(normalize=True).round(3))

# 6. MODEL TRAINING + EVALUATION

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression 
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
cv_lr = cross_val_score(log_reg, X, y, cv=cv, scoring="recall")

print("\n Logistic Regression")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Test ROC-AUC  : {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]):.4f}")
print(f"CV Recall (5-fold): {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(classification_report(y_test, y_pred_lr))
ConfusionMatrixDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig("cm_logreg.png", dpi=150)
plt.show()

# Decision Tree 
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    class_weight="balanced" 
)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
cv_dt = cross_val_score(dt, X, y, cv=cv, scoring="recall")

print("\nDecision Tree ")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Test ROC-AUC  : {roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]):.4f}")
print(f"CV Recall (5-fold): {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")
print(classification_report(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced" 
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
cv_rf = cross_val_score(rf, X, y, cv=cv, scoring="recall")

print("\n Random Forest ")
print(f"Test Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Test ROC-AUC  : {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")
print(f"CV Recall (5-fold): {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
print(classification_report(y_test, y_pred_rf))

# Feature importances
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Feature Importances (Random Forest):")
print(importances.head(10))

plt.figure(figsize=(10, 5))
importances.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()

# 7. ROC CURVE COMPARISON
fig, ax = plt.subplots(figsize=(8, 6))
for model, name in [(log_reg, "Logistic Regression"), (dt, "Decision Tree"), (rf, "Random Forest")]:
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.set_title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()

# 8. FINAL COMPARISON TABLE

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Test Accuracy": [
        round(accuracy_score(y_test, y_pred_lr), 4),
        round(accuracy_score(y_test, y_pred_dt), 4),
        round(accuracy_score(y_test, y_pred_rf), 4)
    ],
    "Test Recall (at-risk)": [
        round(recall_score(y_test, y_pred_lr), 4),
        round(recall_score(y_test, y_pred_dt), 4),
        round(recall_score(y_test, y_pred_rf), 4)
    ],
    "ROC-AUC": [
        round(roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]), 4),
        round(roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]), 4),
        round(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]), 4)
    ],
    "CV Recall Mean": [
        round(cv_lr.mean(), 4),
        round(cv_dt.mean(), 4),
        round(cv_rf.mean(), 4)
    ],
    "CV Recall Std": [
        round(cv_lr.std(), 4),
        round(cv_dt.std(), 4),
        round(cv_rf.std(), 4)
    ]
})

print("\nFinal Model Comparison")
print(results.to_string(index=False))

# 1. Label validity: "at_risk" is derived from grades, not validated burnout measures.
#    The model predicts grade-based academic risk, not burnout in the clinical sense.
# 2. Dataset size: 395 rows is very small. Cross-validation variance will be high.
#    Results are indicative, not reliable for real deployment.
# 3. Leakage risk: G1 and G2 were dropped to prevent label leakage. If kept,
#    model performance would inflate artificially.
# 4. Generalizability: Data is from Portuguese secondary schools. 
#    Do not generalize to other populations without re-validation.
# 5. Threshold: Default 0.5 prediction threshold is used.
