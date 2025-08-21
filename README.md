# AI-Powered-Dashboard

An interactive Streamlit web app for end-to-end machine learning on CSV datasets.
The dashboard supports both classification and regression tasks, featuring automated preprocessing, model training, evaluation metrics, and visualizations.

# Features

Upload your own CSV dataset

Automated preprocessing (missing values, scaling, encoding)

Train and evaluate Regression and Classification models

Linear Regression (R² ~94% on test data)

Random Forest Classification (≈89% accuracy on test data)

Visualizations: confusion matrix, scatter plots, feature importance

Export and import trained models

Downloadable predictions

# Applied Software

Python 3.11+

Streamlit

scikit-learn

Pandas, NumPy, Matplotlib


# Usage

Ensure you have all of these following libraries installed to run this program

streamlit

scikit-learn

pandas

numpy

matplotlib

joblib

After installing all of the libraries run this in your terminal to run the Streamlit app:

streamlit run demo.py


Open the provided local URL in your browser (usually http://localhost:8501).

Upload a CSV file, select a target column, and start training models.

