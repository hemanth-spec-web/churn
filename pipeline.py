"""
Predictive Modeling and Risk Scoring for Bank Customer Churn
End-to-end pipeline: preprocessing -> feature engineering -> modeling -> evaluation -> explainability
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib
import shap

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
OUT = "artifacts"
import os
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------------------------------------
df = pd.read_csv("Churn_Modelling.csv")
df.columns = [c.strip() for c in df.columns]

# Drop non-informative identifiers
df_model = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# ---------------------------------------------------------------------------
# 2. EDA figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
df["Exited"].value_counts().rename({0: "Retained", 1: "Churned"}).plot(
    kind="bar", color=["#2E7D32", "#C62828"], ax=ax
)
ax.set_title("Churn Class Distribution")
ax.set_ylabel("Customers")
plt.tight_layout()
plt.savefig(f"{OUT}/figures/class_distribution.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
num_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure", "NumOfProducts"]
for ax, col in zip(axes.flat, num_cols):
    sns.kdeplot(data=df, x=col, hue="Exited", fill=True, common_norm=False, ax=ax, palette=["#2E7D32", "#C62828"])
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/numeric_distributions.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
churn_geo = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
churn_geo.plot(kind="bar", ax=axes[0], color="#455A64")
axes[0].set_title("Churn Rate by Geography")
axes[0].set_ylabel("Churn Rate")
churn_gender = df.groupby("Gender")["Exited"].mean()
churn_gender.plot(kind="bar", ax=axes[1], color="#455A64")
axes[1].set_title("Churn Rate by Gender")
axes[1].set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(f"{OUT}/figures/churn_by_category.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
churn_prod = df.groupby("NumOfProducts")["Exited"].mean()
churn_prod.plot(kind="bar", ax=ax, color="#6A1B9A")
ax.set_title("Churn Rate by Number of Products")
ax.set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(f"{OUT}/figures/churn_by_products.png", dpi=150)
plt.close()

corr = df_model.select_dtypes(include=np.number).corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/figures/correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
fe = df_model.copy()
fe["BalanceSalaryRatio"] = fe["Balance"] / fe["EstimatedSalary"].replace(0, 1)
fe["ProductDensity"] = fe["NumOfProducts"] / (fe["Tenure"] + 1)
fe["EngagementProductInteraction"] = fe["IsActiveMember"] * fe["NumOfProducts"]
fe["AgeTenureInteraction"] = fe["Age"] * fe["Tenure"]
fe["IsZeroBalance"] = (fe["Balance"] == 0).astype(int)
fe["CreditScoreBand"] = pd.cut(
    fe["CreditScore"], bins=[0, 580, 670, 740, 800, 900],
    labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"]
)

# One-hot encode categoricals
fe = pd.get_dummies(fe, columns=["Geography", "Gender", "CreditScoreBand"], drop_first=True)

feature_cols = [c for c in fe.columns if c != "Exited"]
X = fe[feature_cols]
y = fe["Exited"]

# ---------------------------------------------------------------------------
# 4. Stratified train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scale_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
              "EstimatedSalary", "BalanceSalaryRatio", "ProductDensity",
              "EngagementProductInteraction", "AgeTenureInteraction"]
scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_s[scale_cols] = scaler.transform(X_test[scale_cols])

# ---------------------------------------------------------------------------
# 5. Model development
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE), True),
    "Decision Tree": (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE), False),
    "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), False),
    "Gradient Boosting": (GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE), False),
    "XGBoost": (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, eval_metric="logloss",
                               scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                               random_state=RANDOM_STATE, n_jobs=-1), False),
}

results = []
roc_data = {}
fitted = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_s if needs_scaling else X_train
    Xte = X_test_s if needs_scaling else X_test

    model.fit(Xtr, y_train)
    fitted[name] = model

    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    cv_scores = cross_val_score(model, Xtr, y_train, cv=skf, scoring="roc_auc")

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "CV ROC-AUC (mean)": cv_scores.mean(),
        "CV ROC-AUC (std)": cv_scores.std(),
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_proba))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted[best_model_name]
best_needs_scaling = models[best_model_name][1]
print(f"\nBest model: {best_model_name}")

# ROC curve comparison plot
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/roc_comparison.png", dpi=150)
plt.close()

# Confusion matrix for best model
Xte_best = X_test_s if best_needs_scaling else X_test
y_pred_best = best_model.predict(Xte_best)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/figures/confusion_matrix.png", dpi=150)
plt.close()

with open(f"{OUT}/classification_report.txt", "w") as f:
    f.write(f"Best model: {best_model_name}\n\n")
    f.write(classification_report(y_test, y_pred_best, target_names=["Retained", "Churned"]))

# ---------------------------------------------------------------------------
# 6. Explainability — feature importance + SHAP (tree-based best model preferred)
# ---------------------------------------------------------------------------
explain_name = "XGBoost" if "XGBoost" in fitted else best_model_name
explain_model = fitted[explain_name]
Xte_explain = X_test_s if models[explain_name][1] else X_test

if hasattr(explain_model, "feature_importances_"):
    fi = pd.Series(explain_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    fi.head(12)[::-1].plot(kind="barh", ax=ax, color="#00695C")
    ax.set_title(f"Feature Importance — {explain_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/feature_importance.png", dpi=150)
    plt.close()
    fi.to_csv(f"{OUT}/feature_importance.csv", header=["importance"])

# SHAP
explainer = shap.TreeExplainer(explain_model)
shap_values = explainer.shap_values(Xte_explain.iloc[:1000])
if isinstance(shap_values, list):
    shap_values = shap_values[1]

fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, Xte_explain.iloc[:1000], show=False, plot_size=(8, 6))
plt.tight_layout()
plt.savefig(f"{OUT}/figures/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, Xte_explain.iloc[:1000], plot_type="bar", show=False, plot_size=(8, 5))
plt.tight_layout()
plt.savefig(f"{OUT}/figures/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Partial dependence-style plot (manual, for Age and NumOfProducts) using best model
from sklearn.inspection import PartialDependenceDisplay
pdp_features = [c for c in ["Age", "NumOfProducts", "IsActiveMember", "Balance"] if c in Xte_explain.columns]
fig, ax = plt.subplots(figsize=(11, 8))
PartialDependenceDisplay.from_estimator(explain_model, Xte_explain.astype(float), pdp_features, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/partial_dependence.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. Persist artifacts for the Streamlit app
# ---------------------------------------------------------------------------
joblib.dump(best_model, f"{OUT}/best_model.joblib")
joblib.dump(scaler, f"{OUT}/scaler.joblib")
joblib.dump(fitted, f"{OUT}/all_models.joblib")

metadata = {
    "feature_cols": list(X.columns),
    "scale_cols": scale_cols,
    "best_model_name": best_model_name,
    "best_needs_scaling": best_needs_scaling,
    "explain_model_name": explain_name,
    "explain_needs_scaling": models[explain_name][1],
    "geography_categories": sorted(df["Geography"].unique().tolist()),
    "gender_categories": sorted(df["Gender"].unique().tolist()),
    "credit_band_bins": [0, 580, 670, 740, 800, 900],
    "credit_band_labels": ["Poor", "Fair", "Good", "VeryGood", "Excellent"],
}
with open(f"{OUT}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nArtifacts written to", OUT)
print(json.dumps(metadata, indent=2))
