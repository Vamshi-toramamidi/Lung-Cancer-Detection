🚀 **Lung Cancer Detection using Machine Learning**
This project aims to detect lung cancer in individuals using machine learning techniques, based on patient data and lifestyle attributes. By preprocessing real-world medical data and training a classification model, this project provides a predictive system that can help in early identification of lung cancer risks.

📌 **Project Overview**
The goal of this project is to build a machine learning model that classifies whether an individual has lung cancer based on a variety of features such as age, smoking habits, alcohol use, chronic diseases, fatigue, coughing, and more.

🔍 **Dataset**
Source: Lung_Cancer_Dataset.csv from University of California Irvine.
Features: Age, Gender, Smoking, Alcohol Use, Chronic Disease, Wheezing, Chest Pain, etc.
Target: LUNG_CANCER (Yes/No)

🧠 **Approach**
Data Cleaning:
- Removed duplicates
- Handled categorical features using LabelEncoder
Preprocessing:
- Balanced the dataset using RandomOverSampler from imblearn
- Scaled numerical features (like Age) using StandardScaler
- Transformed binary/categorical values for ML-friendliness
Model Training:
- Used LightGBM Classifier (LGBM) for training the predictive model
- Split data using train_test_split with stratification to maintain label balance
Evaluation:
- Evaluated model using confusion matrix, classification report, and accuracy score
- Achieved accurate predictions with good balance of precision and recall
Prediction:
- Included functionality to test with custom input data

🛠 **Tools & Libraries Used**
Python
- pandas, numpy – Data manipulation
- matplotlib, seaborn – Visualization
- scikit-learn – Preprocessing, metrics, model selection
- lightgbm – Model training
- imbalanced-learn – Oversampling techniques to fix class imbalance

✅ **Results**
- Successfully trained and tested a LightGBM model for lung cancer prediction with 97.02 Accuracy
- Built a simple test interface for real-time predictions
- Achieved strong accuracy with clear interpretability of results
