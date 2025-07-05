# Detecting Fraudulent Transactions Using Machine Learning

## 1. Problem Statement
Fraudulent transactions cost businesses billions annually. The goal of this project is to build a machine learning model that can accurately classify whether a transaction is fraudulent based on behavioral and transactional features.

## 2. Dataset Overview
The dataset includes 150,000 e-commerce transactions with features such as user Id, sign up date, transaction date and time, transaction amount, device Id, store, browser, sex, age and IP location. The target variable is a binary indicator of whether the transaction was fraudulent.

## 3. Tools and Techniques Used
- **Languages:** Python
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib
- **Environment:** Jupyter Notebook, Anaconda
- **Deployment:** Flask API and AWS EC2 instance

## 4. Data Preprocessing
- Mapped IP addresses to Countries using GeoLite2
- Confirmed no missing values
- Encoded categorical variables (one-hot and frequency encoding)
- Scaled numerical features using StandardScaler
- Outlier detection performed

## 5. Exploratory Data Analysis
- Dataset is imbalanced (few fraudulent cases)
- Fraud concentrated in specific countries
- Some browsers and store categories had higher fraud rates
- Correlation matrix used to detect multicollinearity

## 6. Model Building
- **Baseline Models:** Logistic Regression
- **Advanced Models:** Random Forest, XGBoost
- **Deep Learning:** Autoencoder
- Used GridSearchCV and StratifiedKFold

## 7. Evaluation Metrics

| Model          | Precision | Recall | AUC-ROC |
|----------------|-----------|--------|---------|
| Logistic Reg.  | 0.00      | 0.00   | 0.51    |
| Random Forest  | 1.00      | 0.53   | 0.77    |
| XGBoost        | 0.98      | 0.53   | 0.77    |
| Autoencoder    | 0.14      | 0.08   | 0.63    |

## 8. Key Takeaways
- XGBoost and Random Forest had the best fraud detection performance
- Feature importance analysis helped explain model behavior to stakeholders
- SHAP values used for interpretability
- Demonstrated real-time scoring feasibility via Flask

## 9. Resources
- 🗃 [**GitHub Repo**](https://github.com/Sahnoun-A/Fraud-Prevention-ML-Project)
- 📘 [**Kaggle Notebook**](https://www.kaggle.com/code/abdelkabirsahnoun/fraud-prevention)
- 🌐 [**Flask API Demo**](http://ec2-3-17-9-133.us-east-2.compute.amazonaws.com:8080/)
