import copy
from pathlib import Path

import nbformat as nbf


source = Path("Assignments/Assignment 1/Example 3.ipynb")
target = Path("Assignments/Assignment 1/Example 3 V2.ipynb")
nb = nbf.read(source, as_version=4)

# Make the V2 notebook self-contained, while retaining the V1 data-audit work.
nb.cells[1].source = nb.cells[1].source.replace(
    "import warnings\n", "import warnings\nimport time\nimport pickle\n"
).replace(
    "from sklearn.model_selection import train_test_split",
    "from sklearn.base import clone\nfrom sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, LeaveOneGroupOut"
).replace(
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.preprocessing import StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC"
)

nb.cells[30].source = """# V2: model comparison and drift-aware evaluation

This V2 retains V1's reproducible parsing and audit, then evaluates five classifiers using **only the 128 sensor features**. `batch`, `class_id`, and `concentration_ppmv` remain metadata throughout: they are never supplied to a classifier.

The workflow uses a stratified hold-out set for a conventional benchmark, cross-validation inside tuning, and leave-one-batch-out testing to examine sensor drift. The latter is deliberately harder and estimates transfer to an unseen collection batch."""

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

nb.cells.extend([
md("""## 16. Experimental guardrails

`sensor_features` is explicitly defined from the V1 schema. The assertions below make accidental inclusion of batch or concentration fail loudly. All learned transformations (imputation, scaling and PCA) live inside pipelines, so they are fit on training folds only."""),
code("""# 16. Explicitly enforce the no-metadata predictor rule
sensor_features = [f"feature_{i}" for i in range(1, 129)]
excluded_metadata = {"batch", "class_id", "concentration_ppmv", "gas"}

assert sensor_features == feature_cols
assert not (set(sensor_features) & excluded_metadata)
assert X.columns.tolist() == sensor_features

class_labels = sorted(y.unique())
print(f"Predictors ({len(sensor_features)}): feature_1 ... feature_128")
print("Excluded from every classifier:", ", ".join(sorted(excluded_metadata)))
"""),
md("""## 17. Baseline models and timing

The five requested classifiers are benchmarked on the same stratified split. Macro F1 is the primary ranking metric. Fit time, prediction time and serialised model size provide practical computational-cost indicators; they are machine-dependent and should be interpreted comparatively."""),
code("""# 17. Pipelines keep preprocessing leakage-free
def make_original_feature_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "K-Nearest Neighbours": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=1000)),
        ]),
    }


def score_predictions(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0),
    }


def fit_and_measure(name, estimator, X_fit, y_fit, X_eval, y_eval):
    fit_start = time.perf_counter()
    estimator.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    predictions = estimator.predict(X_eval)
    predict_seconds = time.perf_counter() - predict_start
    row = {
        "Model": name,
        **score_predictions(y_eval, predictions),
        "Fit seconds": fit_seconds,
        "Predict seconds": predict_seconds,
        "Model size (MB)": len(pickle.dumps(estimator)) / 1024**2,
    }
    return estimator, predictions, row
"""),
code("""# 17a. Standard stratified hold-out benchmark (128 original sensor features)
benchmark_rows, benchmark_predictions, fitted_baselines = [], {}, {}
for name, estimator in make_original_feature_models().items():
    fitted, predictions, row = fit_and_measure(name, estimator, X_train, y_train, X_test, y_test)
    fitted_baselines[name] = fitted
    benchmark_predictions[name] = predictions
    benchmark_rows.append(row)

benchmark_results = (
    pd.DataFrame(benchmark_rows)
    .sort_values("Macro F1", ascending=False)
    .reset_index(drop=True)
)
display(benchmark_results.style.format({col: "{:.4f}" for col in benchmark_results.columns if col != "Model"}))

plt.figure(figsize=(9, 4))
sns.barplot(data=benchmark_results, x="Macro F1", y="Model", hue="Model", legend=False, palette="viridis")
plt.xlim(0, 1)
plt.title("Standard Hold-out Performance: Original 128 Features")
plt.tight_layout()
plt.show()
"""),
md("""## 18. Original features versus PCA

For a fair comparison, each PCA pipeline learns enough components to explain 95% of training-fold variance. It does not reuse the EDA PCA fitted above, which would leak test information."""),
code("""# 18. Replace feature preprocessing with training-only PCA (95% explained variance)
def make_pca_feature_models():
    models = make_original_feature_models()
    pca_models = {}
    for name, pipeline in models.items():
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                 ("pca", PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE))]
        steps.append(("model", pipeline.named_steps["model"]))
        pca_models[name] = Pipeline(steps)
    return pca_models

pca_rows, fitted_pca = [], {}
for name, estimator in make_pca_feature_models().items():
    fitted, predictions, row = fit_and_measure(name, estimator, X_train, y_train, X_test, y_test)
    row["Representation"] = "PCA (95% training variance)"
    row["PCA components"] = fitted.named_steps["pca"].n_components_
    fitted_pca[name] = fitted
    pca_rows.append(row)

original_rows = benchmark_results.copy()
original_rows["Representation"] = "Original 128 features"
original_rows["PCA components"] = 128
representation_results = pd.concat([original_rows, pd.DataFrame(pca_rows)], ignore_index=True)
representation_results = representation_results.sort_values(["Model", "Representation"]).reset_index(drop=True)
display(representation_results.style.format({col: "{:.4f}" for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Fit seconds", "Predict seconds", "Model size (MB)"]}))

plt.figure(figsize=(10, 5))
sns.barplot(data=representation_results, x="Model", y="Macro F1", hue="Representation")
plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.title("Original 128 Features vs Training-only PCA")
plt.tight_layout()
plt.show()
"""),
md("""## 19. Hyperparameter tuning

Randomised search controls runtime while exploring model-specific choices. The search uses only the training partition with stratified 3-fold cross-validation, optimising macro F1. The untouched test set is evaluated once after each search."""),
code("""# 19. Compact, reproducible tuning searches on original sensor features only
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
param_distributions = {
    "Logistic Regression": {
        "model__C": np.logspace(-3, 2, 12),
        "model__class_weight": [None, "balanced"],
    },
    "K-Nearest Neighbours": {
        "model__n_neighbors": list(range(3, 32, 2)),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "Decision Tree": {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [None, 5, 10, 20, 35],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
    },
    "Random Forest": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 10, 20, 35],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "Support Vector Machine": {
        "model__C": np.logspace(-2, 2, 12),
        "model__gamma": ["scale", "auto", *np.logspace(-4, -1, 8)],
    },
}

tuning_rows, tuned_estimators, tuned_predictions = [], {}, {}
for name, estimator in make_original_feature_models().items():
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions[name],
        n_iter=10,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search_start = time.perf_counter()
    search.fit(X_train, y_train)
    search_seconds = time.perf_counter() - search_start
    test_start = time.perf_counter()
    predictions = search.predict(X_test)
    test_seconds = time.perf_counter() - test_start
    tuned_estimators[name] = search.best_estimator_
    tuned_predictions[name] = predictions
    tuning_rows.append({
        "Model": name,
        "CV Macro F1": search.best_score_,
        **score_predictions(y_test, predictions),
        "Search seconds": search_seconds,
        "Test prediction seconds": test_seconds,
        "Best parameters": search.best_params_,
    })

tuning_results = pd.DataFrame(tuning_rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
display(tuning_results.style.format({col: "{:.4f}" for col in ["CV Macro F1", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Search seconds", "Test prediction seconds"]}))
"""),
md("""## 20. Per-class performance and confusion matrices

The tuned model with the highest test macro F1 is reported in detail. Per-class recall highlights gases that the overall score can conceal, and the normalised confusion matrix makes error patterns comparable across classes."""),
code("""# 20. Detailed diagnostics for the best tuned model
champion_name = tuning_results.iloc[0]["Model"]
champion = tuned_estimators[champion_name]
champion_predictions = tuned_predictions[champion_name]
print("Champion selected by held-out Macro F1:", champion_name)

per_class = pd.DataFrame(
    classification_report(y_test, champion_predictions, labels=class_labels, output_dict=True, zero_division=0)
).T.loc[class_labels, ["precision", "recall", "f1-score", "support"]]
display(per_class.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}))

cm = confusion_matrix(y_test, champion_predictions, labels=class_labels)
cm_normalised = confusion_matrix(y_test, champion_predictions, labels=class_labels, normalize="true")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
axes[0].set(title=f"{champion_name}: counts", xlabel="Predicted", ylabel="Actual")
sns.heatmap(cm_normalised, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
axes[1].set(title=f"{champion_name}: row-normalised", xlabel="Predicted", ylabel="Actual")
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.show()
"""),
md("""## 21. Batch-aware evaluation for sensor drift

Leave-one-batch-out validation trains on all other batches and tests on the held-out batch. `batch` identifies the split only and is not a model input. This result should be compared with the random stratified benchmark: a substantial decline indicates sensitivity to batch-to-batch drift."""),
code("""# 21. Sensor-drift evaluation: hold out one entire batch at a time
logo = LeaveOneGroupOut()
batch_rows = []
for train_idx, test_idx in logo.split(X, y, groups=metadata["batch"]):
    held_out_batch = int(metadata.iloc[test_idx]["batch"].iloc[0])
    batch_model = clone(champion)
    fit_start = time.perf_counter()
    batch_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_seconds = time.perf_counter() - fit_start
    predictions = batch_model.predict(X.iloc[test_idx])
    batch_rows.append({
        "Held-out batch": held_out_batch,
        "Test rows": len(test_idx),
        **score_predictions(y.iloc[test_idx], predictions),
        "Fit seconds": fit_seconds,
    })

batch_results = pd.DataFrame(batch_rows).sort_values("Held-out batch").reset_index(drop=True)
display(batch_results.style.format({col: "{:.4f}" for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Fit seconds"]}))

print("Batch-aware mean Macro F1:", f"{batch_results['Macro F1'].mean():.4f}")
print("Random hold-out champion Macro F1:", f"{tuning_results.iloc[0]['Macro F1']:.4f}")

plt.figure(figsize=(10, 4))
sns.lineplot(data=batch_results, x="Held-out batch", y="Macro F1", marker="o")
plt.axhline(tuning_results.iloc[0]["Macro F1"], color="tab:red", linestyle="--", label="Random hold-out champion")
plt.ylim(0, 1)
plt.title(f"Batch-aware performance for {champion_name}")
plt.legend()
plt.tight_layout()
plt.show()
"""),
md("""## 22. Conclusions and reporting checklist

Use the generated tables to report the selected model, its held-out macro F1, the PCA trade-off, per-class weaknesses, and the gap between random and batch-aware evaluation. Do not claim batch-aware performance as a deployment estimate without considering whether future batches also differ in concentration, environment or acquisition protocol.

**Feature-policy confirmation:** every estimator in this notebook receives `feature_1` through `feature_128` only. `batch` is used solely for grouping in Section 21; `concentration_ppmv` is never used for fitting, selection or prediction."""),
])

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

nb.metadata["title"] = "AI Application Example III: Multi-Class Gas Classification (V2)"
nbf.write(nb, target)
print(f"Wrote {target}")
