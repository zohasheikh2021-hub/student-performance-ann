"""Streamlit app: enter a student's details and see the ANN's predicted performance category.

This app only LOADS the saved model, scaler and class mapping (via predictor.py).
It never trains anything. Start it with:  streamlit run app.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from predictor import (DISCLAIMER, FEATURE_COLUMNS, FEATURE_INFO, INTEGER_FEATURES,
                       ModelNotAvailableError, Predictor, missing_artifacts)

st.set_page_config(page_title="Student Performance Risk Predictor", page_icon="🎓", layout="centered")

CLASS_COLORS = {"Low": "#e4572e", "Medium": "#f3a712", "High": "#2e9e5b"}

st.markdown(
    """
    <style>
      .result-card {padding: 1.1rem 1.3rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);
                    background: rgba(128,128,128,.07); margin-bottom: .5rem;}
      .result-card h2 {margin: 0; font-size: 2rem;}
      .result-card p {margin: .2rem 0 0 0; opacity: .8;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the saved model...")
def load_predictor() -> Predictor:
    """Load the saved ANN + scaler once and reuse them for every prediction."""
    return Predictor()


def probability_chart(probabilities: dict):
    names = list(probabilities)
    values = [probabilities[n] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(6, 2.4))
    bars = ax.barh(names, values, color=[CLASS_COLORS[n] for n in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Model probability (%)")
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(min(v + 1.5, 88), bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


st.title("🎓 Student Performance Risk Predictor")
st.write(
    "A small **Artificial Neural Network** (Keras/TensorFlow) trained on student data estimates whether a student's "
    "performance category is **Low**, **Medium** or **High** from 10 academic and lifestyle features."
)
st.info(DISCLAIMER)

# ---- load the model (clear message if the files are missing) -------------------------------
missing = missing_artifacts()
if missing:
    st.error(
        f"Model files are missing: {', '.join(missing)}. Run the training notebook first "
        "(`jupyter nbconvert --to notebook --execute --inplace notebooks/ann_training.ipynb`)."
    )
    st.stop()
try:
    predictor = load_predictor()
except ModelNotAvailableError as exc:
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

# ---- input form ------------------------------------------------------------------------------
st.subheader("Student details")
st.caption("The starting values are only an illustrative example, not a real student.")

with st.form("student_form"):
    values = {}
    col_left, col_right = st.columns(2)
    for i, name in enumerate(FEATURE_COLUMNS):
        info = FEATURE_INFO[name]
        is_int = name in INTEGER_FEATURES
        cast = int if is_int else float
        with (col_left if i % 2 == 0 else col_right):
            values[name] = st.number_input(
                info["label"],
                min_value=cast(info["min"]),
                max_value=cast(info["max"]),
                value=cast(info["default"]),
                step=cast(info["step"]),
                help=info["help"],
                key=name,
            )
    submitted = st.form_submit_button("Predict performance", type="primary", use_container_width=True)

# ---- prediction --------------------------------------------------------------------------------
if submitted:
    try:
        result = predictor.predict(values)
        explanation = predictor.explain(values, top_k=3)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()
    except Exception as exc:  # unexpected problem: show the real reason
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
        st.stop()

    label = result["predicted_class"]
    st.subheader("Result")
    st.markdown(
        f"""<div class="result-card"><p>Predicted performance category</p>
        <h2 style="color:{CLASS_COLORS[label]}">{label}</h2>
        <p>Model probability for this class: <b>{result['confidence'] * 100:.1f}%</b></p></div>""",
        unsafe_allow_html=True,
    )
    for w in result["warnings"]:
        st.warning(w)

    st.pyplot(probability_chart(result["probabilities"]), clear_figure=True)
    st.caption(
        "These are scores from the model, shown as percentages. They are not guaranteed real-world certainty: "
        "a 70% score does not mean 70 out of 100 such students are correctly classified."
    )

    st.subheader("Why this prediction?")
    st.write(
        f"For each input, we replaced the student's value with the **training average** and checked how the model's "
        f"probability for **{label}** changed. The inputs that moved it most:"
    )
    for row in explanation:
        direction = "pushes towards" if row["effect"] > 0 else "pushes away from"
        st.markdown(
            f"- **{row['label']}** = {row['value']:g} (training average {row['training_average']:.1f}) "
            f"{direction} **{label}** (effect {row['effect'] * 100:+.1f} percentage points)."
        )
    st.caption("This describes how the model behaves for this input. It does not prove what causes a student's performance.")

st.divider()
st.caption(DISCLAIMER)
