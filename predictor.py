"""Shared inference code for the Student Performance Risk Predictor.

Both ``app.py`` (Streamlit) and ``api.py`` (FastAPI) import this module, so the
feature order, the saved scaler and the saved ANN are guaranteed to be the same
in both places. Nothing in this file trains a model: it only LOADS the artifacts
that ``notebooks/ann_training.ipynb`` saved into ``models/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "student_performance.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
MAPPING_PATH = MODEL_DIR / "class_mapping.json"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

# The ORDER of this list is the order of the columns the scaler and ANN were
# trained on. student_id is deliberately NOT here (it is only an identifier).
FEATURE_COLUMNS = [
    "age",
    "attendance_pct",
    "study_hours_per_week",
    "previous_score",
    "assignment_completion_pct",
    "internal_assessment_score",
    "participation_score",
    "sleep_hours_per_day",
    "extracurricular_hours_per_week",
    "failed_subjects",
]

# Hard limits = what is physically / logically possible (e.g. a percentage cannot
# exceed 100). These are used to REJECT impossible inputs. Separately, inputs that
# fall outside the range seen in the training data only produce a warning, because
# the ANN has never seen such students and its output is less trustworthy there.
FEATURE_INFO = {
    "age": dict(label="Age (years)", min=15, max=40, default=19, step=1,
                help="Students in the dataset are aged 17-24."),
    "attendance_pct": dict(label="Attendance (%)", min=0.0, max=100.0, default=82.0, step=0.5,
                           help="Percentage of classes attended."),
    "study_hours_per_week": dict(label="Study hours per week", min=0.0, max=80.0, default=12.5, step=0.5,
                                 help="Self-study hours per WEEK in this dataset."),
    "previous_score": dict(label="Previous score (0-100)", min=0.0, max=100.0, default=71.0, step=0.5,
                           help="Score in the previous assessment/exam."),
    "assignment_completion_pct": dict(label="Assignment completion (%)", min=0.0, max=100.0, default=80.0,
                                      step=0.5, help="Percentage of assignments completed."),
    "internal_assessment_score": dict(label="Internal assessment score (0-100)", min=0.0, max=100.0,
                                      default=68.0, step=0.5, help="Marks in internal assessments."),
    "participation_score": dict(label="Participation score (0-10)", min=0.0, max=10.0, default=7.0, step=0.1,
                                help="Class participation rating out of 10."),
    "sleep_hours_per_day": dict(label="Sleep hours per day", min=0.0, max=24.0, default=7.0, step=0.1,
                                help="Average hours of sleep per DAY."),
    "extracurricular_hours_per_week": dict(label="Extracurricular hours per week", min=0.0, max=60.0,
                                           default=4.0, step=0.5, help="Sports, clubs, etc. per WEEK."),
    "failed_subjects": dict(label="Number of failed subjects", min=0, max=20, default=0, step=1,
                            help="Count of subjects the student has failed."),
}
INTEGER_FEATURES = {"age", "failed_subjects"}

DISCLAIMER = (
    "Educational prediction tool only. The output is a statistical estimate learned from a "
    "dataset; it is not a definitive judgement of any student's ability or future, and the "
    "probabilities are not guarantees."
)


class ModelNotAvailableError(RuntimeError):
    """Raised when the saved model artifacts are missing or cannot be loaded."""


def missing_artifacts(model_dir: Path = MODEL_DIR) -> list[str]:
    """Return the names of required artifact files that do not exist."""
    required = ["student_performance.keras", "scaler.joblib", "class_mapping.json", "training_metadata.json"]
    return [name for name in required if not (Path(model_dir) / name).exists()]


def validate_features(features: dict) -> dict:
    """Check that all 10 features are present, numeric and within the hard limits.

    Returns a clean dict (floats / ints). Raises ValueError with a readable message.
    """
    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary")
    unknown = set(features) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing feature(s): {missing}")

    clean = {}
    for name in FEATURE_COLUMNS:
        info = FEATURE_INFO[name]
        try:
            value = float(features[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {features[name]!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if not info["min"] <= value <= info["max"]:
            raise ValueError(f"{name} must be between {info['min']} and {info['max']}, got {value}")
        if name in INTEGER_FEATURES:
            if value != int(value):
                raise ValueError(f"{name} must be a whole number, got {value}")
            value = int(value)
        clean[name] = value
    return clean


class Predictor:
    """Loads the saved ANN + scaler + class mapping ONCE and serves predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        model_dir = Path(model_dir)
        missing = missing_artifacts(model_dir)
        if missing:
            raise ModelNotAvailableError(
                f"Missing model file(s) in {model_dir}: {missing}. "
                "Run the training notebook first (see README.md)."
            )
        try:
            # Imported here so that simply importing this module stays fast.
            from tensorflow import keras

            self.model = keras.models.load_model(model_dir / "student_performance.keras")
            self.scaler = joblib.load(model_dir / "scaler.joblib")
            with open(model_dir / "class_mapping.json", encoding="utf-8") as f:
                mapping = json.load(f)
            with open(model_dir / "training_metadata.json", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as exc:  # report the real cause instead of hiding it
            raise ModelNotAvailableError(f"Could not load model artifacts: {type(exc).__name__}: {exc}") from exc

        # index (0,1,2) -> label ("Low","Medium","High"), sorted by index
        self.index_to_label = {int(k): v for k, v in mapping["index_to_label"].items()}
        self.class_names = [self.index_to_label[i] for i in sorted(self.index_to_label)]

        # Consistency checks: the artifacts must agree with each other.
        if list(self.scaler.feature_names_in_) != FEATURE_COLUMNS:
            raise ModelNotAvailableError("scaler feature order does not match FEATURE_COLUMNS")
        if self.metadata["feature_columns"] != FEATURE_COLUMNS:
            raise ModelNotAvailableError("training metadata feature order does not match FEATURE_COLUMNS")
        if self.model.input_shape[-1] != len(FEATURE_COLUMNS):
            raise ModelNotAvailableError("ANN input size does not match the number of features")
        if self.model.output_shape[-1] != len(self.class_names):
            raise ModelNotAvailableError("ANN output size does not match the class mapping")
        self.training_ranges = self.metadata["training_feature_ranges"]

    # ------------------------------------------------------------------
    def _scale(self, clean: dict) -> np.ndarray:
        """Build one row in the training column order and apply the SAVED scaler."""
        row = pd.DataFrame([clean], columns=FEATURE_COLUMNS)
        return self.scaler.transform(row)

    def predict(self, features: dict) -> dict:
        """Predict the performance category for one student."""
        clean = validate_features(features)
        x_scaled = self._scale(clean)
        probs = np.asarray(self.model.predict(x_scaled, verbose=0))[0]
        best = int(np.argmax(probs))
        warnings = []
        for name, value in clean.items():
            lo, hi = self.training_ranges[name]["min"], self.training_ranges[name]["max"]
            if value < lo or value > hi:
                warnings.append(
                    f"{name}={value} is outside the range seen during training ({lo} to {hi}); "
                    "the prediction may be less reliable."
                )
        return {
            "predicted_class": self.index_to_label[best],
            "confidence": float(probs[best]),
            "probabilities": {self.index_to_label[i]: float(probs[i]) for i in sorted(self.index_to_label)},
            "warnings": warnings,
        }

    def explain(self, features: dict, top_k: int = 3) -> list[dict]:
        """Simple 'what-if' explanation.

        For each feature we replace its value with the TRAINING AVERAGE (which is 0 after
        scaling) and see how much the probability of the predicted class changes.
        A positive effect means: this student's value for that feature pushes the model
        TOWARDS the predicted class compared with an average student.
        This describes the model's behaviour, it does not prove cause and effect.
        """
        clean = validate_features(features)
        x_scaled = self._scale(clean)[0]
        base_probs = np.asarray(self.model.predict(x_scaled.reshape(1, -1), verbose=0))[0]
        best = int(np.argmax(base_probs))

        variants = np.tile(x_scaled, (len(FEATURE_COLUMNS), 1))
        for i in range(len(FEATURE_COLUMNS)):
            variants[i, i] = 0.0  # 0 in scaled space == training mean
        alt_probs = np.asarray(self.model.predict(variants, verbose=0))[:, best]

        rows = []
        for i, name in enumerate(FEATURE_COLUMNS):
            rows.append({
                "feature": name,
                "label": FEATURE_INFO[name]["label"],
                "value": clean[name],
                "training_average": float(self.scaler.mean_[i]),
                "effect": float(base_probs[best] - alt_probs[i]),
            })
        rows.sort(key=lambda r: abs(r["effect"]), reverse=True)
        return rows[:top_k]
