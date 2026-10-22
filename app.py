import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

MODEL_PATH = "iris_bloom_model.pkl"
DEFAULT_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
DEFAULT_TARGETS = ["setosa", "versicolor", "virginica"]
INPUT_BOUNDS = {
    "sepal length (cm)": (4.3, 7.9),
    "sepal width (cm)": (2.0, 4.4),
    "petal length (cm)": (1.0, 6.9),
    "petal width (cm)": (0.1, 2.5),
}


st.set_page_config(page_title="Iris Bloom", page_icon="🌸", layout="centered")

st.markdown(
    """
    <style>
        :root {
            --bg: #fff8fb;
            --card: #fffdfd;
            --pink: #f9d5e5;
            --lavender: #e9d8ff;
            --mint: #dff7ee;
            --soft-text: #4b4351;
            --accent: #a66de0;
            --muted: #6d6974;
        }
        .stApp {
            background: linear-gradient(135deg, #fffafc 0%, #f9f3ff 45%, #f3fff9 100%);
            color: var(--soft-text);
        }
        div.block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 12px;
        }
        .card {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(166,109,224,0.15);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(141, 118, 175, 0.08);
        }
        h1, h2, h3 {
            color: #4d3b5c;
        }
        .sidebar .stSidebar {
            background: linear-gradient(180deg, #fff5fa 0%, #f4f0ff 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_model_container(obj):
    if isinstance(obj, dict):
        if "model" in obj:
            return obj
        for key in ("model_data", "metadata", "artifact"):
            if key in obj and isinstance(obj[key], dict):
                nested = find_model_container(obj[key])
                if nested is not None:
                    return nested
    return None


def normalize_names(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def load_model_bundle():
    try:
        payload = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    except Exception as exc:  # pragma: no cover - handled in UI
        raise RuntimeError(f"Could not load the model file: {exc}")

    model_container = find_model_container(payload)
    model_data = model_container if model_container is not None else {}

    model = model_data.get("model")
    if model is None:
        if isinstance(payload, dict):
            for key in ("classifier", "estimator", "rf_model", "model_obj"):
                if key in payload:
                    model = payload[key]
                    break
        if model is None:
            model = payload

    feature_names = model_data.get("feature_names")
    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    target_names = model_data.get("target_names")
    if target_names is None and hasattr(model, "classes_"):
        target_names = [str(item) for item in list(model.classes_)]
    if target_names is None:
        target_names = DEFAULT_TARGETS

    return model, list(feature_names), list(target_names)


try:
    model, feature_names, target_names = load_model_bundle()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

if len(feature_names) != 4:
    st.warning(
        "This model does not match the usual 4-feature Iris format. "
        "The UI will still try to use the model's feature order as loaded."
    )

# Keep the input order exactly aligned with the model's training feature order.
feature_order = list(feature_names)
if len(feature_order) < 4:
    feature_order = DEFAULT_FEATURES

if len(feature_order) != 4:
    feature_order = DEFAULT_FEATURES

# Match the common Iris dataset bounds while allowing the app to adapt to the model order.
input_map = {}
for name in feature_order:
    if name in INPUT_BOUNDS:
        input_map[name] = INPUT_BOUNDS[name]
    else:
        input_map[name] = (0.0, 10.0)


with st.sidebar:
    st.markdown("### About Iris Bloom")
    st.markdown(
        """
        Iris Bloom is a beginner-friendly Streamlit app that uses a saved Random Forest model to classify
        Iris flowers. The model was trained on the classic Iris dataset and saved in this folder before this app was built.

        This project keeps the trained model and its feature order intact, then predicts based on the measurements you enter.
        """
    )
    st.caption("Model estimates are probabilities, not guarantees.")


st.title("Iris Bloom")
st.caption("A soft and friendly flower-inspired Iris flower classifier")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("iris_form"):
        col1, col2 = st.columns(2)
        with col1:
            values = {}
            for idx, feature in enumerate(feature_order):
                min_value, max_value = input_map[feature]
                if idx % 2 == 0:
                    values[feature] = st.number_input(
                        feature,
                        min_value=min_value,
                        max_value=max_value,
                        value=float((min_value + max_value) / 2),
                        step=0.1,
                        format="%.2f",
                    )
        with col2:
            for idx, feature in enumerate(feature_order):
                min_value, max_value = input_map[feature]
                if idx % 2 == 1:
                    values[feature] = st.number_input(
                        feature,
                        min_value=min_value,
                        max_value=max_value,
                        value=float((min_value + max_value) / 2),
                        step=0.1,
                        format="%.2f",
                    )

        submitted = st.form_submit_button("Predict species")

    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    ordered_values = [float(values[feature]) for feature in feature_order]
    sample = pd.DataFrame([ordered_values], columns=feature_order)

    try:
        probabilities = model.predict_proba(sample)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = target_names[predicted_index]
        prediction_text = f"Predicted species: {predicted_label}"
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    probs_df = pd.DataFrame(
        {
            "Species": target_names,
            "Estimated probability": probabilities,
        }
    ).sort_values("Estimated probability", ascending=False)

    st.subheader(prediction_text)
    st.write("Estimated class probabilities are model outputs based on the trained Random Forest and should be treated as estimates, not guarantees.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    pastel_colors = ["#f7c8d9", "#d9c7f2", "#c6f0dd"]
    bars = ax.bar(
        probs_df["Species"],
        probs_df["Estimated probability"],
        color=[pastel_colors[i % len(pastel_colors)] for i in range(len(probs_df))],
        edgecolor="#7957a8",
        linewidth=1.2,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Model-estimated class probabilities")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, probs_df["Estimated probability"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4d3b5c",
        )
    fig.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        probs_df.style.format({"Estimated probability": "{:.1%}"}),
        use_container_width=True,
    )

    st.caption("The probability values here are model estimates from the saved Iris model. They are not guaranteed future outcomes.")

else:
    st.info("Enter the flower measurements and click Predict species to get a result.")
