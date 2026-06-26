# 🩺 DiaPredict: End-to-End Diabetes Prediction & Risk Assessment System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Flask Version](https://img.shields.io/badge/flask-3.1.3-green.svg)](https://flask.palletsprojects.com/)
[![Docker Support](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-Random%20Forest%20(81.8%25%20ROC--AUC)-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DiaPredict is an end-to-end Machine Learning web application designed to predict whether a patient has diabetes based on medical parameters from the Pima Indians Diabetes Dataset. By using clean data processing, robust feature scaling, and advanced classification models, the application delivers accurate predictions alongside categorized risk classification and actionable medical recommendations.

![DiaPredict Dashboard](photo/image.png)

---

## 🌟 Key Features

- **Diagnostic Form with In-Line Tooltips:** Interactive, user-friendly forms with real-time numeric constraints and input range helpers.
- **Advanced Predictive Modeling:** Automated comparisons between Logistic Regression, Random Forest, and XGBoost classifiers.
- **Dynamic Risk Categorization:** Calculates exact percentage probabilities and assigns risk levels:
  - **Low Risk:** Probability < 40% (Green indicator)
  - **Medium Risk:** Probability 40% - 70% (Orange indicator)
  - **High Risk:** Probability > 70% (Red indicator)
- **Tailored Clinical Advice:** Generates dynamic recommendations based on individual indicators (BMI, Glucose, Blood Pressure, etc.).
- **Responsive Theme:** Features a premium healthcare theme incorporating glassmorphism, background shape animations, and dark-mode adaptation.
- **Full Docker Containerization:** Production-ready packaging running a Gunicorn WSGI server.

---

## 🛠️ Technology Stack

- **Backend:** Flask, Python (Pandas, NumPy, Scikit-Learn, XGBoost, Joblib)
- **Frontend:** HTML5, Vanilla CSS3 (Glassmorphism design system), JavaScript (ES6 Fetch API)
- **Web Server:** Gunicorn (production WSGI container)
- **Containerization:** Docker, Docker Compose

---

## 📂 Project Structure

<details>
<summary><b>Click to expand folder hierarchy</b></summary>

```text
diabetes-prediction-system/
│
├── dataset/
│   └── diabetes.csv        # Local Pima Indians dataset cache
│
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis & pipeline testing
│
├── model/
│   ├── train.py            # Model training, comparison, and serialization pipeline
│   ├── predict.py          # Real-time inference wrapper with preprocessing
│   └── diabetes_model.pkl  # Combined model, scaler, and imputation medians bundle
│
├── templates/
│   ├── index.html          # Web landing page
│   ├── predict.html        # Interactive patient form & visual results
│   └── about.html          # Dataset summary & algorithm evaluation dashboard
│
├── static/
│   ├── css/
│   │   └── style.css       # Premium custom stylesheet with modern UI components
│   └── js/
│       └── script.js       # Client side input validations & backend API fetcher
│
├── app.py                  # Core Flask application entry point
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Single-stage containerized setup
├── docker-compose.yml     # Multi-container service configuration
└── README.md               # Professional project documentation
```
</details>

---

## ⚙️ Machine Learning Pipeline

### 1. Ingestion & Validation
The pipeline loads the **Pima Indians Diabetes Dataset** (768 records, 9 columns). It performs checks for null values, duplicate rows, and invalid feature datatypes.

### 2. Imputation & Data Cleaning
Values of `0` in parameters such as **Glucose, BloodPressure, SkinThickness, Insulin, and BMI** are physiologically impossible. Rather than deleting data, DiaPredict replaces them with the feature's median value computed **solely from the training split** to prevent data leakage.

### 3. Feature Engineering
Numerical columns are normalized using `StandardScaler` to ensure uniform scale and faster algorithm convergence.

### 4. Model Training & Comparison
The application trains and compares three classifiers on a 80-20 split:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 70.78% | 60.00% | 50.00% | 54.55% | 81.30% |
| **Random Forest** (Selected) | 74.68% | 65.96% | 57.41% | 61.39% | **81.79%** |
| **XGBoost** | 76.62% | 68.75% | 61.11% | 64.71% | 81.02% |

> The **Random Forest** classifier was selected for production because it yielded the highest overall **ROC-AUC (81.79%)**, maximizing sensitivity while minimizing false-positive diagnostics.

---

## 🚀 Getting Started

### Local Setup
Ensure you have Python installed (3.10+ recommended).

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the models and generate the prediction bundle:**
   ```bash
   python model/train.py
   ```
3. **Launch the web application:**
   ```bash
   python app.py
   ```
4. Access the site locally at `http://127.0.0.1:5000`.

### Containerized Setup (Docker)
1. **Run using Docker Compose:**
   ```bash
   docker-compose up --build
   ```
2. **Or build and run via Docker CLI:**
   ```bash
   docker build -t diabetes-app .
   docker run -p 5000:5000 diabetes-app
   ```
3. Access the site at `http://localhost:5000`.

---

## 📡 API Endpoint Reference

### `POST /predict`
Submits medical features to the server for prediction.

*   **Content-Type:** `application/json`
*   **Request Body Example:**
    ```json
    {
      "Pregnancies": 2,
      "Glucose": 120,
      "BloodPressure": 70,
      "SkinThickness": 20,
      "Insulin": 85,
      "BMI": 28.5,
      "DiabetesPedigreeFunction": 0.5,
      "Age": 35
    }
    ```
*   **CURL Command Example:**
    ```bash
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"Pregnancies":2,"Glucose":120,"BloodPressure":70,"SkinThickness":20,"Insulin":85,"BMI":28.5,"DiabetesPedigreeFunction":0.5,"Age":35}'
    ```
*   **Response Body Example:**
    ```json
    {
      "prediction": "Non-Diabetic",
      "probability": 0.32,
      "risk_level": "Low Risk"
    }
    ```

### `GET /health`
Returns system health status.

*   **Response Body Example:**
    ```json
    {
      "status": "healthy"
    }
    ```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
