# ====================================
# IMPORTING REQUIRED LIBRARIES
# ====================================

# Standard Libraries
import io
import json
import time
from typing import Tuple, Optional

# Data Libraries
import numpy as np
import pandas as pd

# Scikit-learn Libraries (for machine learning)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Model Persistence
import joblib

# Visualization
import matplotlib.pyplot as plt

# App Framework
import streamlit as st


# ====================================
# STREAMLIT PAGE SETUP
# ====================================

st.set_page_config(page_title="AI-Powered Dashboard", layout="wide")
st.title("AI-Powered Dashboard with Machine Learning")
st.caption("Upload your CSV file to start the model training process.")


# ====================================
# HELPERS
# ====================================

def get_ohe_feature_names(preprocessor, cat_cols, num_cols):
    """
    Retrieve final feature names after ColumnTransformer (num + OHE cat).
    """
    names = []
    # numeric block
    names.extend(list(num_cols))
    # categorical block
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_out = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            cat_out = []
        names.extend(cat_out)
    return names


def download_json_button(data: dict, filename: str, label: str):
    buf = io.StringIO()
    json.dump(data, buf, indent=2, default=str)
    buf.seek(0)
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="application/json")


# ====================================
# FILE UPLOAD
# ====================================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# Cached CSV read (helps when re-running app with the same file)
@st.cache_data(show_spinner=False)
def _read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

file_bytes = uploaded_file.getvalue()
try:
    df = _read_csv_cached(file_bytes)
except Exception as e:
    st.error(f"Error reading the CSV file: {e}")
    st.stop()

# Coerce obvious numeric strings to numeric (non-destructive for true strings)
df = df.apply(pd.to_numeric, errors="ignore")

# Ensure row_id
if "row_id" not in df.columns:
    df.insert(0, "row_id", np.arange(len(df)))

# Preview
st.subheader("Data Preview")
st.dataframe(df.head(50))
with st.expander("DataFrame Information", expanded=False):
    st.write(f"Rows: **{len(df)}**, Columns: **{len(df.columns)}**")
    st.write("Column Data Types:")
    st.write(df.dtypes.astype(str))

# ====================================
# MODEL SELECTION
# ====================================

if len(df.columns) < 2:
    st.error("The CSV file must contain at least two columns for training.")
    st.stop()

st.markdown("---")
st.subheader("AI Training Model Selection")

target_column = st.selectbox("Select Target Column", options=list(df.columns))
feature_columns = [c for c in df.columns if c not in [target_column, "row_id"]]

# Guard: no usable features
if len(feature_columns) == 0:
    st.error("No feature columns found (only target/row_id present). Please include at least one feature.")
    st.stop()

target_series = df[target_column]
numeric_target = pd.api.types.is_numeric_dtype(target_series)
unique_target = target_series.nunique(dropna=True)
algorithm_suggestion = "classification" if (not numeric_target or unique_target <= 20) else "regression"

algorithm_type = st.radio(
    "Select Machine Learning Algorithm Type",
    options=["Classification", "Regression"],
    index=0 if algorithm_suggestion == "classification" else 1,
    help="The suggested algorithm is recommended but you can override it if you wish",
)
task_type = algorithm_type.lower()

st.caption(
    f"Selected target: **{target_column}** • "
    f"Feature columns: **{len(feature_columns)}** • "
    f"Suggested: **{algorithm_suggestion}** • "
    f"Selected: **{task_type}**"
)

st.markdown("---")
st.subheader("How AI Training is Conducted")

if task_type == "classification":
    model_selection = st.selectbox(
        "Select Classification Machine Learning Algorithm",
        ["Logistic Regression", "Random Forest Classifier"],
    )
else:
    model_selection = st.selectbox(
        "Select Regression Machine Learning Algorithm",
        ["Linear Regression", "Random Forest Regressor"],
    )

cols = st.columns(3)
with cols[0]:
    test_size_data = st.slider(
        "Test Data Size (%)", min_value=10, max_value=50, value=20, step=5,
        help="Percentage of data reserved for testing."
    )
with cols[1]:
    random_state_data = st.number_input(
        "Random Seed", min_value=0, max_value=10000, value=42, step=1
    )
with cols[2]:
    use_balancing = False
    if task_type == "classification":
        use_balancing = st.checkbox("Handle class imbalance (class_weight='balanced')", value=True)

st.caption(
    f"Selected Model: **{model_selection}** • "
    f"Test Size: **{test_size_data}%** • "
    f"Random State: **{random_state_data}**"
)

# Warn: High-cardinality categoricals (OHE blow-up risk)
# (check on full df to reflect user expectations)
temp_X = df[feature_columns]
numeric_mask = temp_X.select_dtypes(include=[np.number]).columns
categorical_features_all = [c for c in temp_X.columns if c not in numeric_mask]
high_card = [c for c in categorical_features_all if df[c].nunique(dropna=True) > 100]
if high_card:
    st.warning(
        f"High-cardinality categorical columns detected (>100 unique values): {high_card}. "
        f"Consider dropping them or using alternative encoding to avoid very wide matrices."
    )

# Leakage check: feature identical to target
leaky = []
for c in feature_columns:
    try:
        if df[c].equals(df[target_column]):
            leaky.append(c)
    except Exception:
        pass
if leaky:
    st.warning(f"Potential leakage detected: {leaky} appear identical to the target. Consider removing them.")

# ====================================
# START TRAINING
# ====================================

st.markdown("---")
st.subheader("Start AI Training")
start_training = st.button("Start Training")

if start_training:
    # Drop NaNs in target
    training_df = df.dropna(subset=[target_column]).copy()
    if training_df.empty:
        st.error("The target column contains only NaN values. Please check your data.")
        st.stop()

    X = training_df[feature_columns]
    y = training_df[target_column]

    # Split setup
    stratifier = None
    test_size_float = test_size_data / 100.0
    if task_type == "classification" and y.nunique(dropna=True) > 1:
        class_counts = y.value_counts(dropna=False)
        min_class_count = class_counts.min()
        can_stratify = (
            (min_class_count >= 2) and
            all((c * test_size_float) >= 1 and (c * (1 - test_size_float)) >= 1 for c in class_counts)
        )
        if can_stratify:
            stratifier = y
        else:
            st.info("Stratified split skipped due to small class size.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size_float,
        random_state=random_state_data,
        stratify=stratifier
    )

    # Identify types on training set
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X_train.columns if col not in numeric_features]

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

    # Model
    if task_type == "classification":
        if model_selection == "Logistic Regression":
            model = LogisticRegression(
                max_iter=2000,
                class_weight=("balanced" if use_balancing else None)
            )
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=random_state_data,
                n_jobs=-1,
                class_weight=("balanced" if use_balancing else None)
            )
    else:
        if model_selection == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(
                n_estimators=300,
                random_state=random_state_data,
                n_jobs=-1
            )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    with st.spinner("Training model..."):
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

    # Predict
    y_pred = pipeline.predict(X_test)

    # ====================================
    # METRICS & VISUALS
    # ====================================

    if task_type == "classification":
        # Metrics (side-by-side)
        cols = st.columns(3)
        with cols[0]:
            st.metric("Accuracy (test)", f"{accuracy_score(y_test, y_pred):.4f}")

        # Safer ROC-AUC (binary; fallback to decision_function; multiclass macro if proba)
        labels_unique = np.sort(np.unique(y_test))
        auc_displayed = False
        try:
            if len(labels_unique) == 2:
                # Binary
                scores = None
                if hasattr(pipeline, "predict_proba"):
                    scores = pipeline.predict_proba(X_test)[:, 1]
                elif hasattr(pipeline, "decision_function"):
                    scores = pipeline.decision_function(X_test)
                if scores is not None:
                    with cols[1]:
                        st.metric("ROC AUC (binary)", f"{roc_auc_score(y_test, scores):.4f}")
                        auc_displayed = True
            elif len(labels_unique) > 2 and hasattr(pipeline, "predict_proba"):
                Y_test_b = label_binarize(y_test, classes=labels_unique)
                proba = pipeline.predict_proba(X_test)
                # When using RandomForestClassifier, predict_proba is (n_samples, n_classes)
                auc_macro = roc_auc_score(Y_test_b, proba, average="macro", multi_class="ovr")
                with cols[1]:
                    st.metric("ROC AUC (multiclass macro)", f"{auc_macro:.4f}")
                    auc_displayed = True
        except Exception:
            pass

        # Classification report
        st.text("Classification report:\n" + classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix (heatmap)
        labels_sorted = labels_unique
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        st.write("Confusion Matrix:")
        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm, interpolation='nearest')
        ax_cm.set_xticks(range(len(labels_sorted)))
        ax_cm.set_yticks(range(len(labels_sorted)))
        ax_cm.set_xticklabels(labels_sorted, rotation=45, ha="right")
        ax_cm.set_yticklabels(labels_sorted)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig_cm)

        # Raw confusion matrix table
        st.dataframe(
            pd.DataFrame(cm,
                         index=[f"True {l}" for l in labels_sorted],
                         columns=[f"Pred {l}" for l in labels_sorted])
        )

        # Predictions CSVs
        out_clf = training_df.loc[y_test.index, ["row_id"] + feature_columns + [target_column]].copy()
        pred_col_clf = f"prediction_{target_column}"
        out_clf[pred_col_clf] = pd.Series(y_pred, index=y_test.index)
        out_clf["correct"] = (out_clf[pred_col_clf] == out_clf[target_column]).astype(int)
        out_clf = out_clf.sort_values("row_id")

        buf_clf = io.BytesIO()
        out_clf.to_csv(buf_clf, index=False)
        buf_clf.seek(0)
        st.download_button("Download Classification Predictions (test set) as CSV",
                           buf_clf, "predictions_classification.csv", "text/csv")

        incorrect_clf = out_clf[out_clf["correct"] == 0].copy()
        buf_incorrect = io.BytesIO()
        incorrect_clf.to_csv(buf_incorrect, index=False)
        buf_incorrect.seek(0)
        st.download_button("Download Incorrect Predictions (classification) as CSV",
                           buf_incorrect, "incorrect_predictions.csv", "text/csv")

        # Feature importance (RF) OR coefficients (LogReg)
        try:
            if "Random Forest" in model_selection:
                final_feature_names = get_ohe_feature_names(
                    pipeline.named_steps["preprocessor"],
                    categorical_features,
                    numeric_features
                )
                importances = pipeline.named_steps["model"].feature_importances_
                fi = pd.DataFrame({
                    "feature": final_feature_names,
                    "importance": importances
                }).sort_values("importance", ascending=False).head(25)
                st.subheader("Top Feature Importances")
                st.dataframe(fi.reset_index(drop=True))
            elif model_selection == "Logistic Regression":
                final_feature_names = get_ohe_feature_names(
                    pipeline.named_steps["preprocessor"],
                    categorical_features,
                    numeric_features
                )
                coefs = pipeline.named_steps["model"].coef_
                if coefs.ndim == 1:
                    coef_df = pd.DataFrame({"feature": final_feature_names, "coef": coefs}) \
                                .sort_values("coef", key=np.abs, ascending=False).head(25)
                else:
                    # multiclass: aggregate by max abs across classes
                    coef_df = pd.DataFrame({"feature": final_feature_names,
                                            "coef_max_abs": np.max(np.abs(coefs), axis=0)}) \
                                .sort_values("coef_max_abs", ascending=False).head(25)
                st.subheader("Top Coefficients")
                st.dataframe(coef_df.reset_index(drop=True))
        except Exception:
            pass

    else:
        # Regression metrics side-by-side
        cols = st.columns(3)
        mae_val = mean_absolute_error(y_test, y_pred)
        rmse_val = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_val = r2_score(y_test, y_pred)
        with cols[0]:
            st.metric("MAE (test)", f"{mae_val:.4f}")
        with cols[1]:
            st.metric("RMSE (test)", f"{rmse_val:.4f}")
        with cols[2]:
            st.metric("R² (test)", f"{r2_val:.4f}")

        # Predictions & errors
        pred_df = pd.DataFrame({"Actual": y_test.reset_index(drop=True),
                                "Predicted": pd.Series(y_pred)})
        pred_df["Error"] = pred_df["Predicted"] - pred_df["Actual"]
        pred_df["AbsError"] = pred_df["Error"].abs()
        st.dataframe(pred_df.head(20))
        st.dataframe(pred_df.nlargest(10, "AbsError"))

        # Residual plot
        st.write("Residual Plot (Predicted vs. Residual):")
        fig_res, ax_res = plt.subplots()
        ax_res.scatter(pred_df["Predicted"], pred_df["Error"], alpha=0.6)
        ax_res.axhline(0, linestyle="--")
        ax_res.set_xlabel("Predicted")
        ax_res.set_ylabel("Residual (Predicted - Actual)")
        st.pyplot(fig_res)

        # Predictions download
        out = training_df.loc[y_test.index, ["row_id"] + feature_columns + [target_column]].copy()
        pred_col = f"prediction_{target_column}"
        out[pred_col] = pd.Series(y_pred, index=y_test.index)
        out["error"] = out[pred_col] - out[target_column]
        out["abs_error"] = out["error"].abs()
        out = out.sort_values("row_id")
        buf = io.BytesIO()
        out.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Download Predictions (test set) as CSV", buf, "predictions.csv", "text/csv")

        # Linear Regression coefficients
        try:
            if model_selection == "Linear Regression":
                final_feature_names = get_ohe_feature_names(
                    pipeline.named_steps["preprocessor"],
                    categorical_features,
                    numeric_features
                )
                coefs = pipeline.named_steps["model"].coef_
                if coefs.ndim == 1:
                    coef_df = pd.DataFrame({"feature": final_feature_names, "coef": coefs}) \
                                .sort_values("coef", key=np.abs, ascending=False).head(25)
                    st.subheader("Top Coefficients")
                    st.dataframe(coef_df.reset_index(drop=True))
        except Exception:
            pass

    # ====================================
    # SAVE TRAINED MODEL
    # ====================================

    st.session_state["trained_pipeline"] = pipeline
    st.session_state["X_columns"] = feature_columns
    st.session_state["numeric_features"] = numeric_features
    st.session_state["categorical_features"] = categorical_features

    model_buf = io.BytesIO()
    joblib.dump(pipeline, model_buf)
    model_buf.seek(0)
    st.download_button("Download Trained Model (.joblib)", model_buf, "trained_pipeline.joblib", "application/octet-stream")

    # Run summary JSON (handy for audit trails)
    run_summary = {
        "task_type": task_type,
        "model": model_selection,
        "random_state": random_state_data,
        "test_size_pct": test_size_data,
        "n_rows": int(len(df)),
        "n_features_used": int(len(feature_columns)),
        "features": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_time_sec": round(train_time, 3),
    }
    if task_type == "classification":
        run_summary["metrics"] = {
            "accuracy": float(accuracy_score(y_test, y_pred))
        }
        # (Optionally could include AUC if computed; keeping minimal)
    else:
        run_summary["metrics"] = {
            "mae": float(mae_val),
            "rmse": float(rmse_val),
            "r2": float(r2_val),
        }
    download_json_button(run_summary, "run_summary.json", "Download Run Summary (JSON)")

    # ====================================
    # SCORE NEW DATA
    # ====================================

    st.markdown("---")
    st.subheader("Use the trained model to score a new CSV")
    score_file = st.file_uploader("Upload CSV to score", type=["csv"], key="score_csv")

    if score_file and "trained_pipeline" in st.session_state:
        try:
            new_df = pd.read_csv(score_file)
        except Exception as e:
            st.error(f"Error reading scoring CSV: {e}")
            new_df = None

        if new_df is not None:
            X_cols = st.session_state["X_columns"]
            missing = [c for c in X_cols if c not in new_df.columns]
            extra = [c for c in new_df.columns if c not in X_cols]
            if missing:
                st.error(f"Missing columns required by the model: {missing}")
            else:
                preds = st.session_state["trained_pipeline"].predict(new_df[X_cols])
                scored = new_df.copy()
                scored[f"prediction_{target_column}"] = preds

                # Optional: prediction confidence for classification (max prob)
                if task_type == "classification" and hasattr(st.session_state["trained_pipeline"], "predict_proba"):
                    try:
                        proba = st.session_state["trained_pipeline"].predict_proba(new_df[X_cols])
                        scored["prediction_confidence"] = proba.max(axis=1)
                    except Exception:
                        pass

                buf_score = io.BytesIO()
                scored.to_csv(buf_score, index=False)
                buf_score.seek(0)
                st.download_button("Download Scored CSV", buf_score, "scored.csv", "text/csv")
