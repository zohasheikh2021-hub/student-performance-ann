# AI-Based Student Performance Risk Predictor

A beginner-friendly **Artificial Neural Network (ANN)** project (Month 3, AIML internship). A Keras/TensorFlow network learns from student data and predicts a performance category — **Low, Medium or High** — which is served through a **Streamlit app** and a **FastAPI endpoint**.

> ⚠️ **Educational tool.** Predictions are statistical estimates learned from one dataset. They are not a judgement of any student's ability or future, and the probabilities are model scores, not guarantees.

---

## 1. Problem statement

Schools record attendance, study time, previous scores, assignment completion and more. Can a neural network use these to estimate which performance category a student belongs to? The goal is **not** a complex deep-learning system, but to understand how an ANN is designed, trained, evaluated, saved and used for predictions.

## 2. Objectives

1. Inspect and prepare the supplied dataset without leaking test information.
2. Build a **traditional ML baseline** (Logistic Regression) as a reference point.
3. Build and train a **feed-forward ANN** in Keras.
4. Evaluate both on the *same* unseen test set (accuracy, macro precision/recall/F1, class-wise metrics, confusion matrix).
5. Save the trained model and the fitted scaler, then **load** them (never retrain) in a Streamlit app and a FastAPI service.

## 3. Dataset

`data/student_data.csv` — the file supplied with the assignment, unmodified.

| Fact (measured in the notebook) | Value |
|---|---|
| Rows × columns | 15,000 × 12 |
| Missing values / duplicate rows | 0 / 0 |
| `student_id` values | 15,000, all unique (identifier only) |
| Class counts | Low 4,950 · Medium 5,100 · High 4,950 |

**Target:** `performance_category` → Low / Medium / High.
**Mapping used everywhere:** Low = 0, Medium = 1, High = 2 (natural order, saved in `models/class_mapping.json`).

### Features (10 inputs; `student_id` is *not* used)

| Feature | Meaning | Range in the data |
|---|---|---|
| `age` | Age in years | 17 – 24 |
| `attendance_pct` | Attendance percentage | 45 – 100 |
| `study_hours_per_week` | Self-study hours per **week** | 2 – 40 |
| `previous_score` | Previous exam/assessment score | 25 – 100 |
| `assignment_completion_pct` | Assignments completed (%) | 20 – 100 |
| `internal_assessment_score` | Internal assessment marks | 20 – 100 |
| `participation_score` | Class participation (out of 10) | 1 – 10 |
| `sleep_hours_per_day` | Average sleep per **day** | 4 – 10 |
| `extracurricular_hours_per_week` | Sports/clubs per week | 0 – 18 |
| `failed_subjects` | Number of failed subjects | 0 – 5 |

Leakage check (training rows only): the best *single* feature (`study_hours_per_week`) reaches only 0.564 accuracy with a small decision tree, and the numeric part of `student_id` stays at chance level (0.334 vs. 0.340). No obvious leaking column was found.

## 4. Technologies

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Streamlit · FastAPI (+ Pydantic, Uvicorn) · Jupyter Notebook · joblib.

## 5. Project structure

```
student-performance-ann/
├── data/
│   └── student_data.csv              # supplied dataset (unchanged)
├── notebooks/
│   └── ann_training.ipynb            # THE training workflow (EDA → baseline → ANN → evaluation → saving)
├── models/
│   ├── student_performance.keras     # trained ANN
│   ├── scaler.joblib                 # StandardScaler fitted on training rows
│   ├── class_mapping.json            # 0/1/2 -> Low/Medium/High
│   └── training_metadata.json        # extra: feature order, training ranges, library versions
├── reports/
│   ├── evaluation_report.md          # generated from the actual run
│   ├── results.json                  # all real metrics
│   ├── confusion_matrix.png          # ANN and baseline, from real test predictions
│   └── training_curves.png           # extra: loss/accuracy curves used in the report
├── predictor.py                      # extra: shared inference code (see below)
├── app.py                            # Streamlit app
├── api.py                            # FastAPI service
├── requirements.txt
├── README.md
└── .gitignore
```

**Why the two extra files?** `predictor.py` holds the feature order, input limits, scaler + model loading and the prediction function **once**, so the Streamlit app and the API cannot drift apart or preprocess differently. `training_metadata.json` stores the feature order and the ranges seen in training (used for "outside training range" warnings and consistency checks). `training_curves.png` is referenced by the report.

## 6. Training vs. inference (important)

| | Training | Inference |
|---|---|---|
| Where | `notebooks/ann_training.ipynb` | `app.py`, `api.py` (via `predictor.py`) |
| What happens | Learns weights from data, writes `models/` and `reports/` | **Loads** the saved model + scaler and calls `predict` |
| When | Once (or whenever you want to retrain) | Every user request |

**The Streamlit app and API never train anything on startup.** They load `models/student_performance.keras`, `models/scaler.joblib` and `models/class_mapping.json` (the model is loaded once and cached) and, if a file is missing, show a clear message instead of silently retraining.

## 7. Data preprocessing

1. `student_id` and the target are excluded from the features.
2. **Stratified** splits with a fixed seed (42): 80 % train-pool / 20 % **test**; the pool is split again 85 % / 15 % into **train (10,200)**, **validation (1,800)** and **test (3,000)** rows. The test set is used only once, for the final comparison.
3. **`StandardScaler`** is fitted on the training rows only; validation/test/new inputs are transformed with that same fitted scaler (`models/scaler.joblib`).

## 8. Baseline model

* **Majority-class dummy** — always guesses the most common class (no-skill floor).
* **Logistic Regression** (`StandardScaler` → `LogisticRegression`), trained on the same 10,200 training rows as the ANN and evaluated on the same test rows.

## 9. ANN architecture

```
Input (10 features) → Dense(16, ReLU) → Dense(3, Softmax)      227 trainable parameters
```

(10×16 + 16 = 176 parameters in the hidden layer, 16×3 + 3 = 51 in the output layer.)

**Why this size?** The data is small and tabular (10 inputs, ~10,200 training rows, 3 classes). The notebook compared four small candidates using **validation loss only**:

| Candidate | Parameters | Best-epoch validation loss | Validation accuracy |
|---|---|---|---|
| A: 16 | 227 | 0.6403 | 0.6933 |
| B: 32 → 16 | 931 | 0.6492 | 0.6894 |
| C: 64 → 32 | 2,883 | 0.6529 | 0.6894 |
| D: 64 → 32 → 16 | 3,363 | 0.6506 | 0.6889 |

The selection rule was fixed in advance: lowest validation loss, preferring a simpler model if it is within 1 %. Larger networks did not help, so the simplest one (A) was chosen. This is a *data-based* justification, but all differences here are small.

## 10. Training process

* Loss: `sparse_categorical_crossentropy` · Optimizer: Adam (learning rate 0.001) · Metric: accuracy
* Batch size 64 · up to 100 epochs · **early stopping** (patience 10, restore best weights) · **model checkpoint** on best validation loss
* Seed 42 (Python, NumPy, TensorFlow; deterministic ops enabled)
* Actual run: 91 epochs executed, best epoch 81.

## 11. Evaluation — actual results (test set, 3,000 students)

All numbers come from `reports/results.json` (produced by the notebook run, not typed by hand).

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Majority-class dummy | 0.3400 | 0.1133 | 0.3333 | 0.1692 |
| Logistic Regression | 0.6950 | 0.6956 | 0.6964 | 0.6958 |
| **ANN (Keras)** | 0.6947 | 0.6948 | 0.6961 | 0.6953 |

Class-wise F1 (ANN / Logistic Regression): Low 0.763 / 0.761 · Medium 0.562 / 0.561 · High 0.761 / 0.765.

ANN confusion matrix (rows = actual, columns = predicted):

| | Pred Low | Pred Medium | Pred High |
|---|---|---|---|
| **Actual Low** | 772 | 205 | 13 |
| **Actual Medium** | 242 | 567 | 211 |
| **Actual High** | 19 | 226 | 745 |

**Honest reading of these results**

* **The ANN did not beat the simple baseline.** The two models are practically tied (macro-F1 difference −0.0005; the accuracy margin of error on 3,000 test students is about ±1.65 points). The extra complexity of the ANN was not rewarded on this dataset.
* Both are far above the no-skill floor (0.34), so the features do carry signal — but ~69.5 % accuracy means roughly 3 in 10 test students are misclassified.
* **Medium is the hardest class** (F1 ≈ 0.56); Low and High are mostly confused with Medium, and almost never with each other (13 and 19 cases).
* ANN accuracy is similar on train (0.6919), validation (0.6933) and test (0.6947), so there is no strong sign of overfitting. The classes seem to overlap heavily in these features; this is an interpretation, not something the numbers prove.
* Limits: one dataset, one split, one seed. See `reports/evaluation_report.md` for the full report.

## 12. Setup and usage

Tested on **Python 3.12.3 (Linux)** in a fresh virtual environment created from `requirements.txt`. Other Python versions and Windows/macOS were **not** tested here.

Tested versions: TensorFlow 2.21.0 (Keras 3.15.1), scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.8, Seaborn 0.13.2, joblib 1.5.3, Streamlit 1.66.0, FastAPI 0.143.3, Uvicorn 0.54.0, notebook 7.6.3, ipykernel 7.4.0.

### Install

```bash
cd student-performance-ann
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Train the model (only needed to retrain; trained files are already in `models/`)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/ann_training.ipynb
```

This runs the whole notebook (about 2–4 minutes on a CPU), then rewrites everything in `models/` and `reports/`. You can also run `jupyter notebook` and choose *Run All*. With the fixed seed, two runs in our test environment gave identical metrics; other machines may differ slightly.

### Launch the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown (normally http://localhost:8501), enter student details and press **Predict performance**. You get the predicted category, probabilities for Low/Medium/High (with a chart), and a short "what-if" explanation.

### Launch the API

```bash
uvicorn api:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

| Endpoint | Purpose |
|---|---|
| `GET /health` | Is the API running, are the model files present and loaded? |
| `POST /predict` | Send the 10 features as JSON, get the prediction |

**Example request** (an illustrative input, *not* a real dataset record):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":19,"attendance_pct":82.0,"study_hours_per_week":12.5,"previous_score":71.0,"assignment_completion_pct":80.0,"internal_assessment_score":68.0,"participation_score":7.0,"sleep_hours_per_day":7.0,"extracurricular_hours_per_week":4.0,"failed_subjects":0}'
```

**Response produced by the trained model in our test run** (will differ slightly if you retrain; this is a model output for an illustrative input, not a verified outcome):

```json
{
  "predicted_class": "Medium",
  "confidence": 0.5163,
  "probabilities": {"Low": 0.0999, "Medium": 0.5163, "High": 0.3838},
  "warnings": [],
  "note": "Educational prediction tool only. ..."
}
```

(The project PDF shows an example output of "High, 89.4 %"; that was an illustration of the output format, not what this model returns for those inputs.)

Invalid input (e.g. `attendance_pct` = 150, a missing field, a text value, an unknown field) returns **HTTP 422** with the reason. If the model files are missing, `/health` reports it and `/predict` returns **HTTP 503**. Values that are possible but outside the range seen in training (e.g. age 35) are accepted but come with a `warnings` entry.

`curl` on Windows PowerShell: use `curl.exe`, or open `/docs` and use *Try it out*.

## 13. Understanding the ANN (for your viva)

**Perceptron vs. multilayer ANN.** A *perceptron* is a single neuron: it adds up weighted inputs and fires (or not). It can only draw a straight dividing line. A *multilayer ANN* stacks layers of neurons, so later layers combine the outputs of earlier ones and can learn curved, more complex patterns. Ours has one hidden layer of 16 neurons and an output layer of 3.

**Neuron, weights and bias.** A neuron calculates `z = w1·x1 + w2·x2 + … + b`. Each weight *w* says how much that input matters; the *bias b* shifts the result. Example: `study_hours_per_week` might get a large positive weight if more study tends to go with a higher category. Our network has 227 such numbers to learn.

**Activation function (ReLU).** After `z` is computed, the neuron applies an activation. ReLU is `max(0, z)`: negative values become 0, positive pass through. Without it, stacked layers would collapse into one straight-line model; with it, the network can learn non-linear patterns.

**Forward propagation.** A student's 10 scaled numbers enter the network → each of the 16 hidden neurons computes its weighted sum + ReLU → the 3 output neurons compute weighted sums of those → softmax turns them into three probabilities. That left-to-right pass is also exactly what happens at prediction time.

**Loss function.** The loss measures how wrong the probabilities are. We use *categorical cross-entropy*: if the student is truly "High" and the network gave High only 20 %, the loss is large; if it gave 90 %, the loss is small.

**Backpropagation.** After the loss is computed, backpropagation works backwards through the network and calculates how much each weight and bias contributed to the error (its *gradient*), using the chain rule from calculus. Every weight then moves a small step in the direction that reduces the loss.

**Optimizer (Adam).** The optimizer decides *how* to take those steps. Adam adapts the step size for each weight and works well with little tuning; learning rate 0.001 sets the base step size.

**Epoch and batch size.** One *epoch* = the network has seen every training row once. Rather than updating after every student, we update after each *batch* of 64 rows (10,200 ÷ 64 ≈ 160 updates per epoch).

**Validation data.** 1,800 students that are *not* used to change the weights. After each epoch we measure loss on them. If training loss keeps falling but validation loss rises, the network is memorising (overfitting). Early stopping uses this to stop at the best epoch. The **test set** is separate and touched only once at the end.

**Feature scaling.** `previous_score` (25–100) and `failed_subjects` (0–5) live on very different scales. Large-valued features would dominate the weighted sums and make learning unstable and slow. Standardising (`(x − mean) / std`, using training-set statistics only) puts every feature on a comparable scale. The same saved scaler must be applied at prediction time, otherwise the network receives numbers it was never trained on.

**Softmax and class probabilities.** The output layer yields three raw scores; softmax converts them to positive numbers that sum to 1. For illustration (made-up numbers): scores `[0.2, 1.4, 1.0]` become roughly `[0.15, 0.51, 0.34]`; the largest is the predicted class. These are *model scores*, not guaranteed real-world certainty.

**Model saving and inference.** After training, the learned weights are stored in `student_performance.keras` and the scaler's mean/std in `scaler.joblib`. At inference we load them, scale the new student's values with the saved scaler, run one forward pass and map the highest probability to a label with `class_mapping.json`. No learning happens at inference.

## 14. Limitations

* The ANN performed no better than Logistic Regression here; a more complex network is not automatically a better one.
* ~69.5 % accuracy: many students are misclassified, especially in the Medium class.
* One dataset, one split, one seed; results could shift with a different split.
* The data is very clean and balanced, unlike most real school records, so results may not transfer to real students.
* We do not know how `performance_category` was defined; the model learns the dataset's labelling, not true ability.
* Feature–category relationships are correlations, not causes. The app's "what-if" explanation describes model behaviour only.
* Not suitable for real decisions about students.

## 15. Future improvements

* Cross-validation and several random seeds to quantify variability.
* Compare more baselines (e.g. random forest, gradient boosting).
* Probability calibration; regularisation (dropout) experiments.
* Check performance by age group for fairness.
* Add unit tests and a Docker image for easier deployment.
