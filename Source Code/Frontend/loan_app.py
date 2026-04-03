# app.py — FINAL Flask backend for LightGBM Loan Risk Prediction (OPTIMIZED THRESHOLD)

import json
import joblib
import pandas as pd
import os
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file

# ---------------- Flask init ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "better_model.pkl"
SCHEMA_PATH = BASE_DIR / "model_schema.json"

# ⭐ OPTIMIZED THRESHOLD - Changed from 0.6 to 0.8 for 88.65% accuracy
PREDICTION_THRESHOLD = 0.8  # 80% threshold for High Risk classification

# ---------------- Load model ----------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

model = joblib.load(MODEL_PATH)
schema = json.loads(SCHEMA_PATH.read_text())

print("✅ LightGBM model & schema loaded")
print(f"⚙️  Prediction threshold set to: {PREDICTION_THRESHOLD*100}%")

# ---------------- Template Diagnostics ----------------
print("📁 Current directory:", os.getcwd())
print("📂 Base directory:", BASE_DIR)
print("📂 Templates folder exists:", (BASE_DIR / "templates").exists())
if (BASE_DIR / "templates").exists():
    print("📄 Files in templates folder:")
    for f in (BASE_DIR / "templates").iterdir():
        print(f"   - {f.name}")
else:
    print("⚠️ WARNING: templates folder not found!")

# ---------------- Global Processing Status ----------------
processing_status = {
    'total': 0,
    'processed': 0,
    'low_risk': 0,
    'high_risk': 0,
    'complete': False,
    'file_path': None,
    'error': None
}

# ---------------- Routes ----------------

# 1️⃣ MAIN LANDING PAGE (First page - main.html)
@app.route("/")
@app.route("/main")
def main():
    template_path = BASE_DIR / "templates" / "main.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create main.html in the templates folder", 404
    return render_template("main.html")


# 2️⃣ INDEX PAGE (Second page - index.html)
@app.route("/index")
def index():
    template_path = BASE_DIR / "templates" / "index.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create index.html in the templates folder", 404
    return render_template("index.html")


# 3️⃣ HOME PAGE (Third page - home.html)
@app.route("/home")
def home():
    template_path = BASE_DIR / "templates" / "home.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create home.html in the templates folder", 404
    return render_template("home.html")


# 5️⃣ DATASET PAGE
@app.route("/dataset")
def dataset():
    template_path = BASE_DIR / "templates" / "dataset.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create dataset.html in the templates folder", 404
    return render_template("dataset.html")


# 6️⃣ ABOUT US PAGE
@app.route("/aboutus")
def aboutus():
    template_path = BASE_DIR / "templates" / "aboutus.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create aboutus.html in the templates folder", 404
    return render_template("aboutus.html")


# 7️⃣ PREDICT LOAN PAGE
@app.route("/predictloan")
def predictloan():
    template_path = BASE_DIR / "templates" / "predictloan.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create predictloan.html in the templates folder", 404
    return render_template("predictloan.html")


# 8️⃣ PREDICT RESULT PAGE
@app.route("/predictresult")
def predictresult():
    template_path = BASE_DIR / "templates" / "predictresult.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create predictresult.html in the templates folder", 404
    return render_template("predictresult.html")


# 9️⃣ CONTACT US PAGE
@app.route("/contactus")
def contactus():
    template_path = BASE_DIR / "templates" / "contactus.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create contactus.html in the templates folder", 404
    return render_template("contactus.html")


# 🔟 DASHBOARD PAGE
@app.route("/dashboard")
def dashboard():
    template_path = BASE_DIR / "templates" / "dashboard.html"
    if not template_path.exists():
        return f"❌ Template not found: {template_path}<br>Please create dashboard.html in the templates folder", 404
    return render_template("dashboard.html")


# 📥 DOWNLOAD DATASET ROUTE
@app.route("/download/dataset")
def download_dataset():
    try:
        dataset_path = BASE_DIR / "loan dataset.csv"
        if not dataset_path.exists():
            return jsonify({
                "error": "Dataset file not found. Please ensure 'loan dataset.csv' is in your project folder."
            }), 404
        
        return send_file(
            dataset_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='loan_dataset_full.csv'
        )
    except Exception as e:
        print("❌ Download error:", e)
        return jsonify({"error": str(e)}), 400


# 📊 UPLOAD CSV FOR BATCH PREDICTION
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file
        upload_path = BASE_DIR / "temp_upload.csv"
        file.save(upload_path)
        
        # Reset status
        global processing_status
        processing_status = {
            'total': 0,
            'processed': 0,
            'low_risk': 0,
            'high_risk': 0,
            'complete': False,
            'file_path': None,
            'error': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_csv_file, args=(upload_path,))
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Processing started", "status": "success"})
    
    except Exception as e:
        print("❌ Upload error:", e)
        return jsonify({"error": str(e)}), 400


# Background processing function
def process_csv_file(file_path):
    global processing_status
    
    try:
        print("🔄 Starting CSV processing...")
        
        # Read CSV
        df = pd.read_csv(file_path)
        total_rows = len(df)
        processing_status['total'] = total_rows
        
        print(f"📊 Total rows to process: {total_rows}")
        print(f"📋 Columns in CSV: {list(df.columns)}")
        
        # Normalize column names to lowercase with underscores
        column_mapping = {
            'LoanID': 'loan_id',
            'Age': 'age',
            'Income': 'income',
            'LoanAmount': 'loan_amount',
            'CreditScore': 'credit_score',
            'MonthsEmployed': 'months_employed',
            'NumCreditLines': 'num_credit_lines',
            'InterestRate': 'interest_rate',
            'LoanTerm': 'loan_term',
            'DTIRatio': 'dti_ratio',
            'Education': 'education',
            'EmploymentType': 'employment_type',
            'MaritalStatus': 'marital_status',
            'HasMortgage': 'has_mortgage',
            'HasDependents': 'has_dependents',
            'LoanPurpose': 'loan_purpose',
            'HasCoSigner': 'has_cosigner',
            'Default': 'default'
        }
        
        # Rename columns if they exist in PascalCase format
        df.rename(columns=column_mapping, inplace=True)
        print(f"📋 Columns after normalization: {list(df.columns)}")
        
        num_cols = schema["numeric_features"]
        cat_cols = schema["categorical_features"]
        threshold = PREDICTION_THRESHOLD  # Use optimized threshold (0.8)
        
        print(f"⚙️  Using threshold: {threshold*100}% for High Risk classification")
        
        # Decoders
        education_map = {
            0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"
        }
        employment_map = {
            0: "Full-time", 1: "Part-time", 2: "Self-employed", 3: "Unemployed"
        }
        marital_map = {
            0: "Single", 1: "Married", 2: "Divorced"
        }
        purpose_map = {
            0: "Home", 1: "Auto", 2: "Personal", 
            3: "Education", 4: "Business", 5: "Other"
        }
        
        # Process in batches for speed
        batch_size = 5000  # Larger batches for speed
        predictions = []
        probabilities = []
        error_count = 0
        
        print("🚀 Checking and preprocessing categorical columns...")
        
        # Pre-process all categorical columns at once (vectorized)
        for col in ['education', 'employment_type', 'marital_status', 'loan_purpose']:
            if col not in df.columns:
                print(f"  ⚠️ {col} not found in CSV - will cause prediction issues!")
                continue
                
            # Check if column is completely empty
            if df[col].isna().all():
                print(f"  ❌ {col} is completely empty - CANNOT PREDICT without this data!")
                print(f"     This CSV appears to be missing categorical data.")
                print(f"     Please upload a CSV with all required columns filled in.")
                raise ValueError(f"Column '{col}' is completely empty. Cannot make predictions without categorical data.")
            
            # Check if column has any non-empty values
            non_empty_count = df[col].notna().sum()
            empty_count = df[col].isna().sum()
            
            if empty_count > 0:
                print(f"  ⚠️ {col} has {empty_count} empty rows out of {total_rows}")
            
            print(f"  🔄 Processing {col} ({non_empty_count} non-empty values)...")
            
            # Get a sample non-null value to check data type
            sample_value = df[col].dropna().iloc[0] if non_empty_count > 0 else None
            
            if sample_value is None:
                raise ValueError(f"Column '{col}' has no valid values")
            
            # Check if data is already text (string)
            if isinstance(sample_value, str):
                # Already text - validate it matches expected values
                print(f"  ✅ {col} already contains text values - keeping as-is")
                unique_vals = df[col].dropna().unique()
                print(f"     Unique values: {list(unique_vals)[:5]}...")
            else:
                # Numeric - convert to text labels
                print(f"  🔄 Converting {col} from numeric to text...")
                try:
                    # Select the appropriate mapping
                    if col == 'education':
                        mapping = education_map
                    elif col == 'employment_type':
                        mapping = employment_map
                    elif col == 'marital_status':
                        mapping = marital_map
                    elif col == 'loan_purpose':
                        mapping = purpose_map
                    else:
                        mapping = {}
                    
                    # Convert to int first (handles both float and int types)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
                    
                    # Map to text labels
                    df[col] = df[col].map(mapping)
                    
                    # Check for unmapped values
                    unmapped = df[col].isna().sum()
                    if unmapped > 0:
                        print(f"  ⚠️ Warning: {col} has {unmapped} unmapped values")
                    else:
                        print(f"  ✅ {col} successfully converted ({df[col].nunique()} unique values)")
                        
                except Exception as e:
                    print(f"  ❌ Error converting {col}: {str(e)}")
                    raise
        
        # Convert boolean columns (vectorized)
        for col in ['has_mortgage', 'has_dependents', 'has_cosigner']:
            if col not in df.columns:
                print(f"  ⚠️ {col} not found in CSV - will cause prediction issues!")
                continue
                
            # Check if column is completely empty
            if df[col].isna().all():
                print(f"  ❌ {col} is completely empty - CANNOT PREDICT without this data!")
                raise ValueError(f"Column '{col}' is completely empty. Cannot make predictions without categorical data.")
            
            non_empty_count = df[col].notna().sum()
            empty_count = df[col].isna().sum()
            
            if empty_count > 0:
                print(f"  ⚠️ {col} has {empty_count} empty rows out of {total_rows}")
            
            print(f"  🔄 Processing {col}...")
            
            # Get a sample non-null value
            sample_value = df[col].dropna().iloc[0] if non_empty_count > 0 else None
            
            if sample_value is None:
                raise ValueError(f"Column '{col}' has no valid values")
            
            # Check if already text
            if isinstance(sample_value, str):
                print(f"  ✅ {col} already contains text - keeping as-is")
                # Standardize Yes/No values
                df[col] = df[col].str.lower().replace({
                    'yes': 'Yes', 'no': 'No', 
                    'true': 'Yes', 'false': 'No',
                    '1': 'Yes', '0': 'No'
                })
            else:
                # Numeric - convert to Yes/No
                print(f"  🔄 Converting {col} to Yes/No...")
                try:
                    # Convert to int (handles bool, float, int)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    # Map to Yes/No
                    df[col] = df[col].map({1: 'Yes', 0: 'No'})
                    
                    # Check for unmapped values
                    unmapped = df[col].isna().sum()
                    if unmapped > 0:
                        print(f"  ⚠️ Warning: {col} has {unmapped} unmapped values, filling with 'No'")
                        df[col] = df[col].fillna('No')
                    else:
                        print(f"  ✅ {col} successfully converted")
                        
                except Exception as e:
                    print(f"  ❌ Error converting {col}: {str(e)}")
                    raise
        
        print("✅ Pre-processing complete! Starting predictions...")
        
        # Process in large batches
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()
            batch_end = min(i + batch_size, total_rows)
            
            try:
                # Select only the columns needed for prediction
                X = batch_df[num_cols + cat_cols].copy()
                
                # Convert categorical columns to category dtype
                for c in cat_cols:
                    X[c] = X[c].astype("category")
                
                # Predict entire batch at once (FAST!)
                batch_probs = model.predict_proba(X)[:, 1]
                batch_preds = (batch_probs >= threshold).astype(int)
                
                # Convert to human-readable format
                batch_pred_labels = ["High Risk" if p else "Low Risk" for p in batch_preds]
                batch_prob_pcts = [round(float(p) * 100, 2) for p in batch_probs]
                
                predictions.extend(batch_pred_labels)
                probabilities.extend(batch_prob_pcts)
                
                # Update counts
                low_risk_count = sum(1 for p in batch_preds if p == 0)
                high_risk_count = sum(1 for p in batch_preds if p == 1)
                
                processing_status['low_risk'] += low_risk_count
                processing_status['high_risk'] += high_risk_count
                processing_status['processed'] = batch_end
                
                print(f"✅ Processed {batch_end}/{total_rows} rows ({(batch_end/total_rows*100):.1f}%) - Batch: {low_risk_count} Low Risk, {high_risk_count} High Risk")
                
            except Exception as e:
                # If batch fails, fall back to row-by-row for this batch only
                print(f"⚠️ Batch processing failed at rows {i}-{batch_end}, using row-by-row fallback: {str(e)}")
                
                for idx in range(i, batch_end):
                    try:
                        row = df.iloc[idx]
                        input_row = {}
                        
                        # Numeric features
                        for c in num_cols:
                            input_row[c] = float(row[c])
                        
                        # Categorical features (already converted above)
                        for c in cat_cols:
                            input_row[c] = row[c]
                        
                        X = pd.DataFrame([input_row])
                        for c in cat_cols:
                            X[c] = X[c].astype("category")
                        
                        prob = model.predict_proba(X)[0, 1]
                        pred = int(prob >= threshold)
                        
                        predictions.append("High Risk" if pred else "Low Risk")
                        probabilities.append(round(float(prob) * 100, 2))
                        
                        if pred == 0:
                            processing_status['low_risk'] += 1
                        else:
                            processing_status['high_risk'] += 1
                            
                    except Exception as row_error:
                        error_count += 1
                        print(f"⚠️ Error on row {idx}: {str(row_error)}")
                        predictions.append(f"Error: {str(row_error)[:50]}")
                        probabilities.append(0)
                
                processing_status['processed'] = batch_end
        
        # Add predictions to DataFrame
        df['Predict'] = predictions
        df['Risk_Probability_%'] = probabilities
        
        # Save result
        output_filename = f"predictions_{int(time.time())}.csv"
        output_path = BASE_DIR / output_filename
        df.to_csv(output_path, index=False)
        
        processing_status['complete'] = True
        processing_status['file_path'] = output_filename
        
        print(f"✅ Processing complete! File saved: {output_filename}")
        print(f"📊 Summary: {processing_status['low_risk']} Low Risk, {processing_status['high_risk']} High Risk, {error_count} Errors")
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        processing_status['complete'] = True
        processing_status['error'] = str(e)


# 📈 GET PROCESSING STATUS
@app.route("/processing_status", methods=["GET"])
def get_processing_status():
    return jsonify(processing_status)


# 📥 DOWNLOAD PROCESSED FILE
@app.route("/download_predictions/<filename>")
def download_predictions(filename):
    try:
        file_path = BASE_DIR / filename
        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404
        
        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'loan_predictions_{int(time.time())}.csv'
        )
    except Exception as e:
        print("❌ Download error:", e)
        return jsonify({"error": str(e)}), 400


# 🔟 HEALTH CHECK
@app.route("/health")
def health():
    return jsonify({"ok": True, "model": "LightGBM", "threshold": PREDICTION_THRESHOLD})


# ---------------- Prediction API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        num_cols = schema["numeric_features"]
        cat_cols = schema["categorical_features"]
        threshold = PREDICTION_THRESHOLD  # Use optimized threshold

        # ---------- DECODERS ----------
        education_map = {
            0: "High School",
            1: "Bachelor's",
            2: "Master's",
            3: "PhD"
        }

        employment_map = {
            0: "Full-time",
            1: "Part-time",
            2: "Self-employed",
            3: "Unemployed"
        }

        marital_map = {
            0: "Single",
            1: "Married",
            2: "Divorced"
        }

        purpose_map = {
            0: "Home",
            1: "Auto",
            2: "Personal",
            3: "Education",
            4: "Business",
            5: "Other"
        }

        yesno_map = {True: "Yes", False: "No"}

        # ---------- BUILD INPUT ----------
        row = {}

        for c in num_cols:
            if c not in data:
                raise ValueError(f"Missing numeric field: {c}")
            row[c] = float(data[c])

        row["education"] = education_map[int(data["education"])]
        row["employment_type"] = employment_map[int(data["employment_type"])]
        row["marital_status"] = marital_map[int(data["marital_status"])]
        row["loan_purpose"] = purpose_map[int(data["loan_purpose"])]

        row["has_mortgage"] = yesno_map[bool(int(data["has_mortgage"]))]
        row["has_dependents"] = yesno_map[bool(int(data["has_dependents"]))]
        row["has_cosigner"] = yesno_map[bool(int(data["has_cosigner"]))]

        X = pd.DataFrame([row])

        for c in cat_cols:
            X[c] = X[c].astype("category")

        prob = model.predict_proba(X)[0, 1]
        pred = int(prob >= threshold)

        return jsonify({
            "prediction": pred,
            "result": "High Risk" if pred else "Low Risk",
            "probability": round(float(prob), 4),
            "threshold": threshold
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 400


# ---------------- Main ----------------
if __name__ == "__main__":
    print("🚀 Starting Loan Risk Prediction Flask App")
    print(f"⚙️  Optimized threshold: {PREDICTION_THRESHOLD*100}% (Expected accuracy: ~88.65%)")
    print("🌐 Access the app at: http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)