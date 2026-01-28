# Customer Churn Prediction – Streamlit App

## Project Overview
Customer Churn Prediction – Streamlit App is an end-to-end machine learning web application built with Python and Streamlit. The app loads a pre-trained classification model and associated metadata to predict whether an individual customer is likely to churn. It provides a probability/confidence score, displays model metadata (e.g., accuracy, recall, training date), and surfaces business-friendly recommendations to help reduce attrition.

Tech stack:
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Problem Statement
Customer churn is when a customer stops using a company's product or service. Predicting churn is important because:
- It costs more to acquire a new customer than to retain an existing one.
- Early identification of at-risk customers enables targeted retention strategies.
- Churn prediction helps prioritize interventions with the highest expected return on investment.

This app is intended to demonstrate how a trained model can be deployed to make interpretable, operational predictions and guide business actions.

## Application Workflow
1. Start the Streamlit app (`app2.py`).
2. The app loads:
   - `churn_prediction_model_20260128_140235.pkl` (pre-trained model)
   - `churn_prediction_model_20260128_140235_metadata.json` (model metadata)
3. User supplies customer attributes through the interactive UI.
4. The model computes:
   - A binary churn prediction (Yes / No)
   - A probability/confidence score for the prediction
5. The UI shows:
   - Prediction and probability
   - Model metadata (accuracy, recall, training date)
   - Business-friendly recommendations tailored to the prediction
6. Users can export or copy results for operational use.

## Input Features
The app expects the same features that were used during model training. Typical features included in churn models are grouped below. Check `churn_prediction_model_20260128_140235_metadata.json` for the exact feature list used for this model.

Numerical features (examples)
- `tenure` — number of months the customer has been with the company
- `monthly_charges` — current monthly billing amount
- `total_charges` — cumulative charges billed to date
- `num_products` — number of products/services the customer subscribes to
- `avg_service_calls` — average number of customer service calls

Categorical features (examples)
- `gender` — customer gender (e.g., Male, Female)
- `senior_citizen` — whether the customer is a senior citizen
- `partner` — whether the customer has a partner
- `dependents` — whether the customer has dependents
- `contract` — contract type (e.g., Month-to-month, One year, Two year)
- `payment_method` — payment method (e.g., Electronic check, Mailed check)
- `internet_service` — type of internet service (e.g., DSL, Fiber optic, None)
- `tech_support` — whether customer has tech support

Notes:
- The exact feature names, types, and preprocessing steps are captured in the model metadata JSON file. Always use the matching feature schema when making predictions.

## Model Details
- Model file: `churn_prediction_model_20260128_140235.pkl`
- Metadata file: `churn_prediction_model_20260128_140235_metadata.json`
- The metadata includes:
  - Model performance metrics (e.g., accuracy, recall, precision)
  - Training date and model version
  - Feature list and preprocessing notes
- The model is a binary classifier built and serialized with Scikit-learn and Joblib.

How the app uses metadata:
- Displays performance metrics in the UI so stakeholders can quickly assess model reliability.
- Uses the feature list and preprocessing parameters to validate input and run consistent transformations prior to prediction.

## Installation & Setup (Local)
Prerequisites:
- Python 3.8+ recommended
- Git (optional, for cloning the repository)

1. Clone the repository (or download the source):
```bash
git clone https://github.com/Mayankchonde/Churn_Prediction.git
cd Churn_Prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On macOS/Linux
source .venv/bin/activate
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Confirm that the model and metadata files are present:
- `churn_prediction_model_20260128_140235.pkl`
- `churn_prediction_model_20260128_140235_metadata.json`

## Running the App Locally
Run the Streamlit application:
```bash
streamlit run app2.py
```

Open the URL printed in the terminal (typically http://localhost:8501) to interact with the app.

Common troubleshooting:
- If the app fails to load the model, verify the model file path and that `joblib` is installed.
- If inputs are rejected, check the metadata file for expected column names and preprocessing steps.

## Deployment on Streamlit Cloud
This repository is compatible with Streamlit Cloud. To deploy:
1. Push your repository to GitHub (if not already pushed).
2. Go to https://streamlit.io/cloud and sign in with your GitHub account.
3. Click "New app" and select the repository and branch.
4. Set the main file to `app2.py`.
5. Ensure `requirements.txt` is present at the repo root so Streamlit Cloud installs dependencies automatically.
6. Deploy. The app will build and be publicly available at a Streamlit-generated URL.

Notes:
- If your model file is large, be mindful of repository size limits. Consider hosting large artifacts externally or using Git LFS.
- No secrets or external credentials are required for this app by default. If you add integrations (databases, APIs), configure them through Streamlit Cloud secrets.

## Project Structure
- `app2.py` — Main Streamlit application (UI and prediction logic)
- `churn_prediction_model_20260128_140235.pkl` — Serialized pre-trained ML model
- `churn_prediction_model_20260128_140235_metadata.json` — Model metadata (performance, feature schema)
- `requirements.txt` — Python package dependencies
- `README.md` / `README.txt` — Project documentation

## Notes & Disclaimer
- The model and recommendations in this repository are for demonstration and educational purposes.
- Do not use the predictions for high-stakes or regulated decision-making without additional validation, monitoring, and human review.
- Model performance may degrade on data distributions that differ from the training set. Retrain and re-evaluate the model periodically with up-to-date, representative data.
- Examine and mitigate potential biases: ensure the model does not unfairly disadvantage protected groups.
- Contributions, improvements, and issues are welcome. When submitting data or pull requests, avoid sharing private customer data.

If you need help adapting this app to your environment (custom feature sets, retraining, or integrating with production systems), open an issue or submit a pull request with details.
