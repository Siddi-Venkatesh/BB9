# 💳 From Data Cleaning To Deep Learning : A Full Stack Approach To Loan Default Risk Prediction In Banking Applications

A machine learning–based system for **automated binary classification of loan default risk** using financial and demographic applicant data. The project evaluates **LightGBM and ensemble-based architectures** and focuses on building a **robust, reproducible, and deployment-ready** loan risk prediction pipeline with a full-stack Flask web application.

---

## 👥 Team Information

### Siddi Venkatesh
- LinkedIn : https://www.linkedin.com/in/siddi-venkatesh-5689a4319/
- **Role & Contribution:** Machine Learning & Modeling Lead. Responsible for designing, training, and evaluating the loan prediction model. Worked on feature selection, algorithm selection (LightGBM), model tuning, threshold optimization, and performance evaluation to ensure accurate risk classification.

### Pokala Appaiah
- LinkedIn : https://www.linkedin.com/in/appaiahnaidu-pokala-27799a282?utm_source=share_via&utm_content=profile&utm_medium=member_android
- **Role & Contribution:** Data Processing & Backend Lead. Handled data cleaning, preprocessing, and dataset preparation for the loan prediction system. Implemented backend logic for model integration, data flow, batch prediction handling, and Flask API development.

### Pallothu Venkata Sai Krishna
- LinkedIn : https://www.linkedin.com/in/sai-krishna-4529a0282
- **Role & Contribution:** Frontend & System Integration Lead. Developed the user interface and integrated the trained model into the web application. Focused on form inputs, result visualization, dashboard design, and clear risk-status output to make the system usable and understandable.

---

## 📌 Abstract

For banks, identifying clients who might default on their loans is still a challenging undertaking. Errors are frequently present in the data they deal with, default situations are typically far fewer than non-default cases, and the relationships between the variables are rarely clear-cut.

Instead of testing a few models separately or using limited setups, this study puts together a **complete pipeline** that covers everything from cleaning and preparing the data to comparing a wide range of **machine learning and deep-learning methods** under the same setup. To deal with the imbalance in the dataset, a two-step method is used: **SMOTE** is applied first to create synthetic minority samples, and then **Tomek-link removal** is used to clean up borderline cases. **SHAP** and **LIME** are also included so the reasoning behind the model predictions can be examined rather than treated as a black box.

On top of the individual models, the work also develops **ensemble versions of DenseNet and ResNet**, which regularly outperform the stand-alone versions. After multiple runs and cross-validation, the best ensemble reached a **precision of 99.2%** and showed clear improvements in recall and MCC, with significance at *p < 0.01*. The entire framework is built with real-world usage in mind, aiming to give banks both dependable predictions and explanations that make sense in practice.

---

## Paper Reference (Inspiration)
👉 **Machine Learning and Deep Learning for Loan Prediction in Banking:
   -Exploring Ensemble Methods and Data Balancing
 ([Paper Link]([https://ieeexplore.ieee.org/document/10985749](https://ieeexplore.ieee.org/document/10772107)))**

---

## 🧩 About the Project

This project implements an **end-to-end machine learning pipeline** for loan default prediction. The system takes applicant financial and demographic details as input and predicts the corresponding default risk category. The primary goal is to build a **robust, efficient, and deployable** classification system suitable for academic research and financial risk management applications.

### Applications
- Automated credit risk assessment for financial institutions
- Loan approval decision-support systems
- Batch loan portfolio risk analysis
- Research on financial data and machine learning

---

## 🔁 System Workflow

```text
Applicant Input Data (Individual or Batch CSV)
→ Data Preprocessing & Feature Engineering
→ Categorical Encoding & Normalization
→ LightGBM Classification Model
→ Threshold-Based Risk Classification (80% threshold)
→ Risk Output: Low Risk / High Risk + Probability Score
→ Downloadable Results (Batch Mode)
```

---

## 📊 Dataset Used

### 👉 Loan Default Prediction Dataset

#### 🗂 Dataset Details

- **Total Records:** ~250,000 loan applicant records
- **Number of Classes:** 2 (Low Risk / High Risk)
- **Data Format:** CSV
- **File:** `Dataset/loan_dataset.csv`

#### Feature Columns

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age |
| `income` | Numeric | Annual income |
| `loan_amount` | Numeric | Requested loan amount |
| `credit_score` | Numeric | Credit score |
| `months_employed` | Numeric | Months of employment |
| `num_credit_lines` | Numeric | Number of active credit lines |
| `interest_rate` | Numeric | Loan interest rate |
| `loan_term` | Numeric | Loan term in months |
| `dti_ratio` | Numeric | Debt-to-income ratio |
| `education` | Categorical | High School / Bachelor's / Master's / PhD |
| `employment_type` | Categorical | Full-time / Part-time / Self-employed / Unemployed |
| `marital_status` | Categorical | Single / Married / Divorced |
| `loan_purpose` | Categorical | Home / Auto / Personal / Education / Business / Other |
| `has_mortgage` | Categorical | Yes / No |
| `has_dependents` | Categorical | Yes / No |
| `has_cosigner` | Categorical | Yes / No |
| `default` | Target | 0 = Low Risk, 1 = High Risk |

---

## 🧰 Tools & Technologies Used

- **Programming Language:** Python
- **ML Framework:** LightGBM (Gradient Boosting)
- **Web Framework:** Flask
- **Libraries:** NumPy, Pandas, scikit-learn, joblib, Matplotlib
- **Frontend:** HTML5, CSS3, JavaScript (vanilla)
- **Notebook:** Jupyter Notebook (model development & EDA)

### Development Environment
- Windows 11 (local system)
- Python 3.10 virtual environment

---

## 🔍 Data Preprocessing & EDA

- Dataset cleaned and normalized for consistent column formats
- Categorical features encoded using label and target mapping strategies
- Boolean columns (`has_mortgage`, `has_dependents`, `has_cosigner`) standardized to `Yes/No`
- Handled missing values and outliers
- Dataset split into **training, validation, and testing** sets
- EDA performed in Jupyter Notebook (`Source Code/ModelMaking/`)

---

## 🧪 Model Training Information

- LightGBM model trained using **supervised binary classification**
- Hyperparameters tuned experimentally for optimal performance
- **Prediction threshold optimized to 80%** (instead of default 50%) for best accuracy
- Model serialized with `joblib` for Flask deployment
- Schema stored as `model_schema.json` for runtime validation
- Batch prediction supported with background threading for large CSV files

---

## 🧾 Model Evaluation

### Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC
- Confusion Matrix

Evaluation is performed on **unseen test data** to assess generalization capability.

---

## 🏆 Results (Summary)

- Achieved **~99.2% precision** with optimized prediction threshold
- Strong classification performance across both Low Risk and High Risk categories
- Batch prediction supports datasets of **250,000+ records** efficiently
- Demonstrated fast and reliable prediction via Flask REST API

**Note:** Detailed numerical results are provided in the project documentation.

---

## 🌐 Web Application

The system includes a full-stack Flask web application with the following pages:

| Route | Page |
|---|---|
| `/` or `/main` | Landing Page |
| `/home` | Home / Dashboard Overview |
| `/predictloan` | Individual Loan Prediction Form |
| `/predictresult` | Prediction Result Page |
| `/dashboard` | Analytics Dashboard |
| `/dataset` | Dataset Info & Download |
| `/aboutus` | Team Information |
| `/contactus` | Contact Page |

### Running the Application

```bash
# Install dependencies
pip install flask lightgbm pandas numpy scikit-learn joblib

# Run the Flask app
python "Source Code/Frontend/loan_app.py"

# Access at: http://127.0.0.1:5000/
```

---

## 📁 Repository Structure

```
BB-09/
├── Dataset/
│   └── loan_dataset.csv               # Full loan dataset (~250K records)
├── Documents/
│   ├── BB-09_Abstract.pdf
│   ├── BB-09_CameraReady_Paper.pdf
│   ├── BB-09_Conference_PPT.pptx
│   ├── BB-09_Project_Documentation.pdf
│   └── BB-09_Project_PPT.pptx
└── Source Code/
    ├── Frontend/
    │   ├── loan_app.py                 # Flask backend
    │   ├── templates/                  # HTML pages
    │   └── static/                     # Images & assets
    └── ModelMaking/
        └── *.ipynb                     # Jupyter Notebook (EDA + Model Training)
```

---

## 📄 Documentation

Detailed explanations of system design, dataset handling, model architecture, experiments, and results are available in the `Documents/` folder:

- Abstract
- Project documentation
- Review and final presentations
- Camera-ready paper

---

## ⚠️ Notes

- This project is intended for **academic and research purposes only**.
- The dataset is subject to its respective license.
- The system is **not a replacement for professional financial or credit advisory services**.
- Ensure `better_model.pkl` and `model_schema.json` are present in the `Frontend/` folder before running the app.
