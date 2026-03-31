# BB-09 – From Data Cleaning To Deep Learning : A Full Stack Approach To Loan Default Risk Prediction In Banking Applications

## Team Info
- 22471A05D5 — **Siddi Venkatesh** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Machine Learning & Modeling Lead — Model design, LightGBM training, threshold optimization, and performance evaluation._

- 22471A05C1 — **Pokala Appaiah** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Data Processing & Backend Lead — Data cleaning, preprocessing, Flask API development, and batch prediction handling._

- 22471A05B3 — **Pallothu Venkata Sai Krishna** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Frontend & System Integration Lead — UI development, model integration, result visualization, and dashboard design._

---

## Abstract
For banks, identifying clients who might default on their loans is still a challenging undertaking. Errors are frequently present in the data they deal with, default situations are typically far fewer than non-default cases, and the relationships between the variables are rarely clear-cut.

Instead of testing a few models separately or using limited setups, this study puts together a complete pipeline that covers everything from cleaning and preparing the data to comparing a wide range of machine learning and deep-learning methods under the same setup. To deal with the imbalance in the dataset, a two-step method is used: SMOTE is applied first to create synthetic minority samples, and then Tomek-link removal is used to clean up borderline cases. SHAP and LIME are also included so the reasoning behind the model predictions can be examined rather than treated as a black box. 

On top of the individual models, the work also develops ensemble versions of DenseNet and ResNet, which regularly outperform the stand-alone versions. After multiple runs and cross-validation, the best ensemble reached a precision of 99.2% and showed clear improvements in recall and MCC, with significance at p < 0.01. The entire framework is built with real-world usage in mind, aiming to give banks both dependable predictions and explanations that make sense in practice.

---

## Paper Reference (Inspiration)
👉 **[Predictive Modeling and Performance Analysis of Loan Default Data
  – Author Names xxxxxxxxxx
 ](<Paper URL here>)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Our project enhances standard loan prediction approaches by integrating **SMOTE combined with Tomek-link removal** to rigorously handle class imbalance without introducing borderline noise. Furthermore, we implemented **ensembles of DenseNet and ResNet architectures** rather than relying solely on traditional ML. We also emphasized explainable AI by incorporating **SHAP and LIME** to interpret predictions for stakeholders, and packaged the entire pipeline into a user-friendly, fully deployable **Flask web application** supporting both single and batch predictions.

---

## About the Project
- **What your project does:** It automates the binary classification of loan applicants into "Low Risk" or "High Risk" categories using demographic and financial data.
- **Why it is useful:** It helps financial institutions reduce human bias, save time during the credit approval process, and confidently mitigate financial loss using explainable AI.
- **General project workflow:** Input CSV Data → Data Cleaning & Imbalance Handling (SMOTE+Tomek) → Feature Encoding → Model Prediction (LightGBM/Ensembles) → Risk category & Probability threshold Output.

---

## Dataset Used
👉 **[Loan Default Prediction Dataset](./Dataset/loan_dataset.csv)**

**Dataset Details:**
- **Total Records:** ~250,000 applicants
- **Features:** 16 input variables (Age, Income, Loan Amount, Credit Score, DTI Ratio, Education, etc.)
- **Target:** `default` (0 = Low Risk, 1 = High Risk)

---

## Dependencies Used
Python 3.10, LightGBM, Flask, Pandas, NumPy, scikit-learn, joblib, imbalanced-learn, SHAP, LIME, Matplotlib.

---

## EDA & Preprocessing
- Normalized and cleaned missing values and outliers across 17 features.
- Categorical variables (e.g., Education, Employment Type) were converted using label encoding and text mapping.
- Boolean columns (e.g., `has_mortgage`) standardized to integer/Yes-No formats.
- Dataset class imbalance was heavily mitigated using a two-step approach: **SMOTE** (Synthetic Minority Over-sampling) followed by **Tomek-links** to remove borderline overlapping samples.

---

## Model Training Info
- Benchmarked multiple ML and DL architectures under identical setups.
- Utilized **LightGBM** as a fast, robust baseline classifier.
- Developed advanced **DenseNet and ResNet Ensembles** to capture complex non-linear feature relationships.
- Hyperparameter tuning and cross-validation applied.
- Optimized the prediction threshold to **80%** (0.8) for strict risk aversion, classifying only the most confident predictions as High Risk.

---

## Model Testing / Evaluation
- Evaluated on unseen test splits using multiple metrics to ensure robustness against class imbalance.
- Metrics used: Precision, Recall, Accuracy, F1-Score, MCC (Matthews Correlation Coefficient), and ROC-AUC.
- SHAP and LIME used during evaluation to ensure the model makes decisions based on logical financial factors rather than artifacts.

---

## Results
- The best ensemble model reached a **Precision of 99.2%**.
- Demonstrated clear, statistically significant improvements (*p < 0.01*) in Recall and MCC over stand-alone benchmark models.
- The Flask backend successfully demonstrated handling batch predictions of 250,000+ rows efficiently.

---

## Limitations & Future Work
- **Limitations:** The model relies heavily on the specific regional/economic conditions present in the training data, which may not generalize perfectly to foreign financial markets without retraining.
- **Future Work:** Integrate the system via REST APIs directly into active banking core software, and continually retrain the model with live, streaming loan repayment data to adapt to changing macroeconomic conditions.

---

## Deployment Info
The system incorporates a full-stack web application built with **Flask**.
- **Run Command:** `python "Source Code/Frontend/loan_app.py"`
- **Access URL:** `http://127.0.0.1:5000/`
- **Features:** Features multiple pages including a Dashboard, Single Prediction Form, and a Batch Prediction Upload tool that processes large CSV files using background threading.
