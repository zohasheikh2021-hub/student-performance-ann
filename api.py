"""FastAPI service: GET /health and POST /predict.

The saved ANN, scaler and class mapping are loaded ONCE when the API starts (via predictor.py,
the same module the Streamlit app uses). Nothing is trained here.

Start with:  uvicorn api:app --reload
Docs:        http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from predictor import (DISCLAIMER, FEATURE_INFO, ModelNotAvailableError, Predictor,
                       missing_artifacts)


def _limits(name: str) -> dict:
    """Pydantic bounds for a feature, taken from the single source of truth in predictor.py."""
    return {"ge": FEATURE_INFO[name]["min"], "le": FEATURE_INFO[name]["max"]}


class StudentFeatures(BaseModel):
    """The 10 input features. Unknown or missing fields are rejected (HTTP 422)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "age": 19, "attendance_pct": 82.0, "study_hours_per_week": 12.5, "previous_score": 71.0,
            "assignment_completion_pct": 80.0, "internal_assessment_score": 68.0, "participation_score": 7.0,
            "sleep_hours_per_day": 7.0, "extracurricular_hours_per_week": 4.0, "failed_subjects": 0}},
    )

    age: int = Field(..., description="Age in years", **_limits("age"))
    attendance_pct: float = Field(..., description="Attendance percentage", **_limits("attendance_pct"))
    study_hours_per_week: float = Field(..., description="Study hours per week", **_limits("study_hours_per_week"))
    previous_score: float = Field(..., description="Previous score (0-100)", **_limits("previous_score"))
    assignment_completion_pct: float = Field(..., description="Assignment completion percentage",
                                             **_limits("assignment_completion_pct"))
    internal_assessment_score: float = Field(..., description="Internal assessment score (0-100)",
                                             **_limits("internal_assessment_score"))
    participation_score: float = Field(..., description="Participation score (0-10)", **_limits("participation_score"))
    sleep_hours_per_day: float = Field(..., description="Sleep hours per day", **_limits("sleep_hours_per_day"))
    extracurricular_hours_per_week: float = Field(..., description="Extracurricular hours per week",
                                                  **_limits("extracurricular_hours_per_week"))
    failed_subjects: int = Field(..., description="Number of failed subjects", **_limits("failed_subjects"))


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(..., description="Model probability of the predicted class (0-1)")
    probabilities: dict[str, float]
    warnings: list[str] = []
    note: str = DISCLAIMER


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved artifacts once at startup. If they are missing the API still starts,
    /health reports the problem and /predict answers 503."""
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = Predictor()
    except ModelNotAvailableError as exc:
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="Student Performance Risk Predictor API",
    description="Predicts a student's performance category (Low / Medium / High) with a saved Keras ANN. "
                "Educational tool, not a definitive judgement of any student.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Is the API running, and are the model artifacts available and loaded?"""
    missing = missing_artifacts()
    loaded = app.state.predictor is not None
    return {
        "status": "ok" if loaded else "degraded",
        "api_running": True,
        "model_artifacts_available": not missing,
        "missing_artifacts": missing,
        "model_loaded": loaded,
        "classes": app.state.predictor.class_names if loaded else None,
        "error": app.state.load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    """Predict the performance category for one student."""
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model not available: {app.state.load_error}")
    try:
        return predictor.predict(student.model_dump())
    except ValueError as exc:  # defensive: Pydantic already validated the ranges
        raise HTTPException(status_code=422, detail=str(exc))
