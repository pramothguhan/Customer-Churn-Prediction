## Customer Churn Prediction :chart_with_downwards_trend:

## Overview :memo:

In the highly competitive telecommunications industry, customer churn poses a significant challenge, directly affecting revenue and long-term viability. This project aims to leverage data analytics and machine learning to predict customer churn and provide actionable insights to reduce churn rates. By analyzing customer demographics, usage patterns, service satisfaction, and billing details, the project seeks to identify at-risk customers and develop effective retention strategies.

## Project Goals :dart:

- **Predict Customer Churn:** Develop a predictive model to accurately identify customers who are at high risk of churning.
- **Understand Churn Factors:** Identify key variables such as service usage, billing issues, and customer service interactions that contribute to customer churn.
- **Develop Retention Strategies:** Translate data-driven insights into actionable business strategies to reduce churn, including personalized customer engagement initiatives, adjustments to pricing models, and improvements in customer service.

## Data Source :floppy_disk:

The primary data source for this project is the "Telco Customer Churn" dataset available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download). This dataset includes various customer attributes and their churn status, providing a comprehensive foundation for churn prediction analysis.

## Technologies Used :computer:

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

## Data Exploration :mag_right:

- **Gender and Churn Distribution:** The gender distribution is nearly balanced, allowing for unbiased gender-related analysis.
- **Customer Contract Distribution:** Customers with month-to-month contracts are more likely to churn, highlighting the importance of contract type in churn prediction.
- **Payment Method Analysis:** Electronic checks are the most commonly used payment method among churned customers, suggesting a potential area for customer service improvement.
- **Internet Services and Churn:** Customers with fiber optic services show higher churn rates compared to DSL users, indicating service type as a significant factor.

## Data Preprocessing :wrench:

- **One-Hot Encoding:** Categorical variables were transformed into binary columns to ensure effective use in machine learning models.
- **Handling Missing Values:** Missing values in 'TotalCharges' were identified and dropped, ensuring a clean dataset for analysis.
- **Data Scaling and Standardization:** Numerical data was scaled and standardized to ensure uniformity across features, aiding in model convergence and accuracy.

## Data Mining Models :bar_chart:

### 1. Random Forest Classifier
- **Hyperparameter Tuning:** Utilized GridSearchCV for optimal parameter selection.
- **Feature Importance:** 'TotalCharges', 'tenure', and 'Contract_Month-to-month' emerged as the most influential features.
- **Performance:** Achieved high training accuracy but lower testing accuracy, indicating potential overfitting.

### 2. Logistic Regression
- **Model Implementation:** Grid search optimized regularization and penalty parameters.
- **Key Features:** 'PaymentMethod_Electronic check' and 'Contract_Month-to-month' were significant predictors.
- **Performance:** Demonstrated good generalization with balanced precision and recall.

### 3. Decision Tree
- **Model Tuning:** Balanced model complexity with max_depth and min_samples parameters.
- **Important Features:** 'Contract_Month-to-month' and 'PaymentMethod_Electronic check' were crucial for predictions.
- **Performance:** Lower predictive accuracy but offers simplicity and ease of interpretation.

### 4. AdaBoost Classifier
- **Model Implementation:** Tuned using Decision Tree as a base estimator.
- **Key Features:** 'TotalCharges' and 'gender' were prominent in influencing predictions.
- **Performance:** High training and testing accuracy with balanced recall and precision.

## Performance Evaluation :clipboard:

- **Logistic Regression and AdaBoost:** Both models showed strong generalization capabilities with AUC scores of 0.85.
- **Random Forest:** Despite high training accuracy, overfitting was observed, necessitating further tuning.
- **Decision Tree:** Provided ease of interpretation with lower AUC, suitable for scenarios requiring simple models.

## Impact :bulb:

The project provides significant value to the telecommunications industry by enhancing the accuracy of customer churn predictions. The insights derived allow for strategic decision-making, improving customer retention, reducing operational costs, and deepening the understanding of customer behavior.

## Project Folder Structure :file_folder:

```plaintext
📦 Customer_Churn_Prediction
├─ data
│  ├─ raw
│  ├─ processed
├─ notebooks
│  └─ data_exploration.ipynb
│  └─ model_training.ipynb
├─ src
│  └─ preprocessing.py
│  └─ modeling.py
├─ results
│  └─ model_performance
│  └─ visualizations
└─ README.md
```
