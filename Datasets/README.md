# 📊 Loan Default Prediction Dataset

This folder contains the dataset used in the research project:
**"From Data Cleaning To Deep Learning: A Full Stack Approach To Loan Default Risk Prediction In Banking Applications"**

---

## 📁 File

| File | Size | Format |
|---|---|---|
| `loan_dataset.csv` | ~24 MB | CSV (Comma-Separated Values) |

---

## 🗂 Dataset Overview

| Property | Details |
|---|---|
| **Total Records** | ~250,000 loan applicant entries |
| **Total Features** | 17 (16 input features + 1 target) |
| **Target Variable** | `default` — 0 = Low Risk, 1 = High Risk |
| **Task Type** | Binary Classification |
| **Class Imbalance** | Yes — Default cases are minority class |

---

## 📋 Feature Description

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `age` | Numeric | Applicant's age in years |
| 2 | `income` | Numeric | Annual income of the applicant |
| 3 | `loan_amount` | Numeric | Total loan amount requested |
| 4 | `credit_score` | Numeric | Applicant's credit score |
| 5 | `months_employed` | Numeric | Duration of current employment (months) |
| 6 | `num_credit_lines` | Numeric | Number of active credit lines |
| 7 | `interest_rate` | Numeric | Interest rate on the loan |
| 8 | `loan_term` | Numeric | Loan repayment term (months) |
| 9 | `dti_ratio` | Numeric | Debt-to-income ratio |
| 10 | `education` | Categorical | High School / Bachelor's / Master's / PhD |
| 11 | `employment_type` | Categorical | Full-time / Part-time / Self-employed / Unemployed |
| 12 | `marital_status` | Categorical | Single / Married / Divorced |
| 13 | `loan_purpose` | Categorical | Home / Auto / Personal / Education / Business / Other |
| 14 | `has_mortgage` | Categorical | Yes / No |
| 15 | `has_dependents` | Categorical | Yes / No |
| 16 | `has_cosigner` | Categorical | Yes / No |
| 17 | `default` | **Target** | 0 = Low Risk, 1 = **High Risk (Default)** |

---

## ⚙️ Preprocessing Applied

- Missing values handled and outliers treated
- Categorical columns encoded (label encoding / text mapping)
- Class imbalance addressed using **SMOTE + Tomek-link removal**
- Dataset split into **training, validation, and testing** sets

---

## 📥 Download

You can download the dataset directly from the GitHub repository:

👉 **[Download loan_dataset.csv](https://github.com/Siddi-Venkatesh/BB-09/raw/main/Dataset/loan_dataset.csv)**

> Right-click the link above → **"Save link as..."** to download the CSV file directly.

---

## ⚠️ Notes

- This dataset is used strictly for **academic and research purposes**.
- It is subject to its respective license and terms of use.
- Do **not** use this data as a basis for real-world financial decisions.
