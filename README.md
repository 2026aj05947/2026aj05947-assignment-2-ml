# Spam Sentinel — Spam E-mail Classification with Five ML Models

**M.Tech (AIML) · Machine Learning · Assignment 2**

- **GitHub repository:** https://github.com/2026aj05947/2026aj05947-assignment-2-ml
- **Live Streamlit app:** https://2026aj05947-assignment-2-ml.streamlit.app

---

## a. Problem statement

Unsolicited bulk e-mail (spam) wastes users' time and is a common carrier for phishing and malware.
The goal is to build a **binary classifier** that predicts whether an e-mail is **spam (1)** or
**not spam (0)** from word, character and capital-letter statistics of its text. We implement and
compare five classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive
Bayes and Support Vector Machine — on the same train/test split. Each is scored on Accuracy, AUC,
Precision, Recall, F1 and Matthews Correlation Coefficient (MCC), and all five are served through an
interactive Streamlit app.

## b. Dataset description

| Item           | Details                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------- |
| Name / source  | **Spambase**, UCI Machine Learning Repository (dataset id 94) — https://archive.ics.uci.edu/dataset/94/spambase |
| Donors         | Hopkins, Reeber, Forman & Suermondt, Hewlett-Packard Labs (1999)                                                |
| Task           | Binary classification — target column `is_spam` (1 = spam, 0 = not spam)                                        |
| Instances      | 4,601 e-mails (1,813 spam = 39.4 %, 2,788 not spam)                                                             |
| Features       | **57**, all continuous and non-negative                                                                         |
| Missing values | None                                                                                                            |

**Feature groups**

| Group                        | Count | Meaning                                                                                        |
| ---------------------------- | ----- | ---------------------------------------------------------------------------------------------- |
| `word_freq_<WORD>`           | 48    | % of words in the e-mail equal to WORD (e.g. `free`, `remove`, `money`, `hp`, `george`, `000`) |
| `char_freq_<CHAR>`           | 6     | % of characters equal to `;` `(` `[` `!` `$` `#`                                               |
| `capital_run_length_average` | 1     | average length of uninterrupted runs of capital letters                                        |
| `capital_run_length_longest` | 1     | length of the longest capital-letter run                                                       |
| `capital_run_length_total`   | 1     | total number of capital letters                                                                |

**Preparation used in this project**

1. The original UCI column names were applied (`data/spambase.csv`).
2. **391 exact duplicate rows were removed** (4,210 rows remain), so that no copy of an e-mail
   can appear in both the training and the test set.
3. Stratified **80 / 20 train–test split** (`random_state = 2026`): 3,368 training rows, 842 test rows
   (39.9 % spam in both). The test split is saved as **`test_data.csv`** and is the file uploaded to the app.
4. The features are heavily right-skewed (e.g. `capital_run_length_total` goes up to 15,841), so the
   Logistic Regression, KNN, Naive Bayes and SVM pipelines apply `log(1 + x)` followed by
   `StandardScaler`. The Decision Tree uses raw features (it is scale-invariant).
5. Hyper-parameters for every model were chosen by **5-fold stratified cross-validation on the
   training split only** (scoring = F1). The test split was used once, for the final evaluation.

## c. GitHub repository link

https://github.com/2026aj05947/2026aj05947-assignment-2-ml

Repository structure:

```
spam-classifier-ml/
├── app.py                         # Streamlit app (loads saved models, no retraining)
├── train_models.py                # data split, CV tuning, training, evaluation, saving
├── 2026AJ05947.ipynb              # notebook (EDA + training + results)
├── requirements.txt
├── .streamlit/config.toml         # light theme for the app
├── README.md
├── test_data.csv                  # held-out test split (57 features + is_spam)
├── data/
│   └── spambase.csv               # full UCI Spambase data with column names
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv                # comparison table below
    ├── best_params.json           # tuned hyper-parameters
    └── meta.json                  # feature list, target, library versions
```

Each `.joblib` file is a full scikit-learn `Pipeline`, so the preprocessing (log transform + scaler)
is saved together with the classifier.

## d. Models used

| Model                  | Implementation                        | Tuned hyper-parameters (5-fold CV)                        |
| ---------------------- | ------------------------------------- | --------------------------------------------------------- |
| Logistic Regression    | `LogisticRegression` (L2)             | C = 1                                                     |
| Decision Tree          | `DecisionTreeClassifier`              | criterion = entropy, max_depth = 14, min_samples_leaf = 1 |
| KNN                    | `KNeighborsClassifier`                | k = 5, weights = distance                                 |
| Naive Bayes            | `GaussianNB`                          | var_smoothing = 1e-3                                      |
| Support Vector Machine | `SVC`, RBF kernel, probability = True | C = 10, gamma = 0.01                                      |

### Comparison table — evaluation metrics on the held-out test set (842 e-mails)

Positive class = spam. AUC is computed from predicted probabilities; the other metrics use the
default 0.5 threshold.

| ML Model Name          | Accuracy   | AUC        | Precision  | Recall | F1         | MCC        |
| ---------------------- | ---------- | ---------- | ---------- | ------ | ---------- | ---------- |
| Logistic Regression    | 0.9264     | 0.9750     | 0.9335     | 0.8780 | 0.9049     | 0.8460     |
| Decision Tree          | 0.9157     | 0.9219     | 0.9233     | 0.8601 | 0.8906     | 0.8235     |
| KNN                    | 0.9264     | 0.9709     | 0.9228     | 0.8899 | 0.9061     | 0.8459     |
| Naive Bayes            | 0.8278     | 0.9537     | 0.7108     | 0.9583 | 0.8162     | 0.6870     |
| Support Vector Machine | **0.9359** | **0.9799** | **0.9462** | 0.8899 | **0.9172** | **0.8660** |

> KNN AUC is 0.9709 on Linux (Streamlit Cloud); running the notebook on macOS gives 0.9720 for the same saved model because of floating-point differences in distance-weighted neighbour probabilities. All other values are identical.

### Confusion matrix summary on the test set (842 e-mails: 506 genuine, 336 spam)

| ML Model Name          | Genuine e-mails correctly kept (TN) | Genuine e-mails wrongly flagged as spam (FP) | Spam e-mails missed (FN) | Spam e-mails correctly caught (TP) |
| ---------------------- | ----------------------------------- | -------------------------------------------- | ------------------------ | ---------------------------------- |
| Logistic Regression    | 485                                 | 21                                           | 41                       | 295                                |
| Decision Tree          | 482                                 | 24                                           | 47                       | 289                                |
| KNN                    | 481                                 | 25                                           | 37                       | 299                                |
| Naive Bayes            | 375                                 | 131                                          | 14                       | 322                                |
| Support Vector Machine | 489                                 | 17                                           | 37                       | 299                                |

### Observations on model performance

| ML Model Name                       | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression                 | A strong linear baseline: 92.6 % accuracy, AUC 0.975 and MCC 0.846. The log transform matters here — it turns the skewed frequency counts into features that relate roughly linearly to the log-odds of spam. Precision (0.934) is higher than recall (0.878): it rarely flags genuine mail (21 false positives) but lets 41 spam e-mails through. Second-best AUC, so its probability ranking is very reliable.                                                                                                                                                                                                                                 |
| Decision Tree                       | The weakest of the four strong models (accuracy 0.916, MCC 0.824). Its AUC is clearly the lowest (0.922) because a single tree outputs coarse, leaf-level probabilities, so it ranks e-mails less smoothly than the other models. CV picked a deep tree (depth 14, 162 leaves), which tends toward high variance and gives it the most missed spam (47 FN). Its splits are easy to interpret, though: `char_freq_$`, `word_freq_remove`, `char_freq_!` and `word_freq_hp` carry most of the importance.                                                                                                                                          |
| KNN                                 | With k = 5 and distance weighting on log-scaled, standardised features, it ties Logistic Regression on accuracy (0.926). Its recall (0.890) and F1 (0.906) are slightly better, and its precision slightly lower. It works well because spam e-mails form tight neighbourhoods in feature space, but it must store all 3,368 training rows (largest model file) and is slower at prediction time. Without feature scaling KNN would be dominated by the capital-run-length columns.                                                                                                                                                              |
| Naive Bayes                         | Clearly the lowest accuracy (0.828), precision (0.711) and MCC (0.687), but the **highest recall (0.958)**: it misses only 14 spam e-mails, at the cost of 131 false alarms. The feature-independence assumption is violated (e.g. `word_freq_857` / `word_freq_415` / `word_freq_telnet` are strongly correlated — r = 0.99 between `857` and `415`), so it double-counts evidence and pushes probabilities toward "spam". Its AUC is still a respectable 0.954, so the ranking is decent and a higher decision threshold would trade recall for precision. For a mail filter, flagging this much genuine mail as spam is usually unacceptable. |
| Support Vector Machine              | Best on 5 of 6 metrics: accuracy 0.936, AUC 0.980, precision 0.946, F1 0.917 and MCC 0.866 (it ties KNN on recall, 0.890). The RBF kernel captures non-linear interactions between word and character frequencies that the linear model cannot, and the margin-maximising objective with C = 10 generalises well. It has the fewest false positives (17), which matters most for spam filtering because losing a genuine e-mail is costlier than seeing one spam. Downsides: slower to train, and probabilities need Platt scaling.                                                                                                              |
| **Overall winner for this dataset** | **Support Vector Machine (RBF kernel)** — highest Accuracy, AUC, Precision, F1 and MCC, and the fewest genuine e-mails wrongly flagged. Logistic Regression and KNN are close runners-up (MCC ≈ 0.846). Naive Bayes is only preferable if catching every spam matters more than false alarms.                                                                                                                                                                                                                                                                                                                                                    |

---

## Streamlit app

Features (all computed from the **saved** models — nothing is retrained on the uploaded data):

- **CSV upload** of test data (must contain the 57 feature columns and `is_spam`); a sample
  `test_data.csv` can be downloaded or loaded with the "Use bundled test_data.csv" switch.
- **Model selection dropdown** for the five classifiers, plus an adjustable decision threshold.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix**, **classification report** and ROC curve for the selected model.
- **Compare all models** tab: metric table and bar chart for all five models on the uploaded file.
- **Predictions** tab: per-row predictions with spam probability, misclassified-only filter and CSV download.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py          # optional - re-creates model/ and test_data.csv (deterministic)
streamlit run app.py
```

> The saved `.joblib` models were produced with scikit-learn 1.9.0 (see `model/meta.json`);
> `requirements.txt` pins the same version so the models load correctly on Streamlit Community Cloud.
