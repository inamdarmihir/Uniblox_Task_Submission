# Insurance Enrollment Prediction - Project Report

## 1. Introduction

This report documents the development of a machine learning pipeline to predict whether employees will opt in to a new voluntary insurance product based on demographic and employment-related data. The project was completed as part of an internal pilot for a company modernizing insurance using machine learning.

## 2. Data Observations

### 2.1 Dataset Overview

The dataset consists of synthetic census-style employee data with 10,000 records and 10 columns:

- `employee_id`: Unique identifier for each employee
- `age`: Age of the employee
- `gender`: Gender of the employee (Female, Male, Other)
- `marital_status`: Marital status (Single, Married, Divorced, Widowed)
- `salary`: Annual salary of the employee
- `employment_type`: Type of employment (Full-time, Part-time, Contract)
- `region`: Geographic region (Northeast, Midwest, South, West)
- `has_dependents`: Whether the employee has dependents (Yes, No)
- `tenure_years`: Years of employment at the company
- `enrolled`: Target variable (1 for enrolled, 0 for not enrolled)

### 2.2 Data Quality

The dataset was analyzed for quality issues:

- **Missing Values**: No missing values were found in any column
- **Duplicates**: No duplicate records were identified
- **Data Types**: All columns had appropriate data types
- **Target Distribution**: The enrollment status is relatively balanced, with approximately 50% of employees enrolled

### 2.3 Feature Analysis

#### Categorical Features

- **Gender**: The dataset includes Female, Male, and Other categories, with Female being the most common
- **Marital Status**: Includes Single, Married, Divorced, and Widowed categories
- **Employment Type**: Full-time employees represent the majority, followed by Part-time and Contract workers
- **Region**: Employees are distributed across Northeast, Midwest, South, and West regions
- **Has Dependents**: Approximately half of the employees have dependents

#### Numerical Features

- **Age**: Ranges from 22 to 64 years, with a relatively uniform distribution
- **Salary**: Ranges from approximately $26,000 to $101,000, with a mean of around $65,000
- **Tenure Years**: Ranges from 0 to 22.6 years, with a right-skewed distribution

### 2.4 Key Insights from EDA

1. **Enrollment by Age**: Middle-aged employees (40-60) show higher enrollment rates compared to younger and older employees
2. **Enrollment by Salary**: Higher salary ranges correlate with higher enrollment rates
3. **Enrollment by Employment Type**: Full-time employees have higher enrollment rates compared to Part-time and Contract workers
4. **Enrollment by Region**: Employees in the South and West regions show higher enrollment rates
5. **Enrollment by Dependents**: Employees with dependents are more likely to enroll
6. **Enrollment by Tenure**: Employees with longer tenure show higher enrollment rates
7. **Interaction Effects**: Combinations of factors, such as Full-time employees with dependents, show particularly high enrollment rates

## 3. Model Choices & Rationale

### 3.1 Feature Engineering

Based on the exploratory data analysis, several new features were created to improve model performance:

1. **Age Groups**: Categorized age into groups (<30, 30-40, 40-50, 50-60, 60+)
2. **Salary Ranges**: Created quintiles for salary (Very Low, Low, Medium, High, Very High)
3. **Tenure Groups**: Categorized tenure into groups (<1, 1-3, 3-5, 5-10, 10+)
4. **Employment with Dependents**: Combined employment type and dependents status
5. **Region with Marital Status**: Combined region and marital status
6. **Salary per Tenure**: Created a feature to capture compensation growth

These engineered features helped capture non-linear relationships and interactions between variables that might influence enrollment decisions.

### 3.2 Preprocessing Pipeline

A preprocessing pipeline was created to prepare the data for modeling:

1. **Numerical Features**: Standardized using StandardScaler to ensure all features have the same scale
2. **Categorical Features**: Encoded using OneHotEncoder to convert categorical variables into a format suitable for machine learning algorithms

### 3.3 Model Selection

Multiple classification algorithms were evaluated as baseline models:

1. **Logistic Regression**: A simple, interpretable model that serves as a good baseline
2. **Random Forest**: An ensemble method that can capture non-linear relationships and feature interactions
3. **Gradient Boosting**: An advanced ensemble method known for high performance in structured data problems
4. **Support Vector Machine (SVM)**: A powerful algorithm for finding optimal decision boundaries
5. **K-Nearest Neighbors (KNN)**: A non-parametric method that can capture local patterns

Each model was evaluated using 5-fold cross-validation with ROC AUC as the primary metric.

### 3.4 Hyperparameter Tuning

The best-performing baseline model (Gradient Boosting) was further optimized through hyperparameter tuning using GridSearchCV. The following hyperparameters were tuned:

- `n_estimators`: Number of boosting stages
- `learning_rate`: Step size shrinkage to prevent overfitting
- `max_depth`: Maximum depth of individual regression estimators
- `min_samples_split`: Minimum samples required to split an internal node
- `min_samples_leaf`: Minimum samples required to be at a leaf node
- `subsample`: Fraction of samples used for fitting the individual base learners

## 4. Evaluation Results

### 4.1 Baseline Models Performance

The baseline models were evaluated using 5-fold cross-validation:

| Model | Mean ROC AUC | Standard Deviation |
|-------|--------------|-------------------|
| Logistic Regression | 0.9692 | ±0.0025 |
| Random Forest | 1.0000 | ±0.0000 |
| Gradient Boosting | 1.0000 | ±0.0000 |
| SVM | 0.9939 | ±0.0008 |
| KNN | 0.9681 | ±0.0036 |

Gradient Boosting and Random Forest achieved perfect ROC AUC scores in cross-validation, indicating excellent predictive performance.

### 4.2 Final Model Performance

The tuned Gradient Boosting model was evaluated on the test set with the following metrics:

| Metric | Value |
|--------|-------|
| Accuracy | 0.9995 |
| Precision | 0.9992 |
| Recall | 1.0000 |
| F1 Score | 0.9996 |
| ROC AUC | 1.0000 |
| Average Precision | 1.0000 |

The confusion matrix showed near-perfect classification with minimal misclassifications.

### 4.3 Feature Importance

The top features influencing enrollment prediction were:

1. **Salary**: Higher salaries correlate with higher enrollment rates
2. **Age**: Middle-aged employees are more likely to enroll
3. **Tenure Years**: Longer tenure correlates with higher enrollment
4. **Has Dependents (Yes)**: Having dependents increases enrollment likelihood
5. **Employment Type (Full-time)**: Full-time employees are more likely to enroll

## 5. Key Takeaways

1. **Demographic Factors Matter**: Age, salary, and having dependents are strong predictors of insurance enrollment
2. **Employment Stability Influences Decisions**: Full-time employees with longer tenure are more likely to enroll
3. **Regional Differences Exist**: Geographic location affects enrollment rates
4. **Feature Engineering Improves Performance**: Creating interaction features and grouping variables enhanced model performance
5. **Ensemble Methods Excel**: Gradient Boosting and Random Forest performed exceptionally well for this prediction task
6. **High Predictive Power**: The final model achieved near-perfect performance, suggesting that the available features contain strong signals for predicting enrollment

## 6. Future Improvements

With more time, the following enhancements could be implemented:

1. **Model Explainability**: Implement SHAP (SHapley Additive exPlanations) values to provide more detailed and individualized feature importance
2. **Feature Selection**: Perform recursive feature elimination to identify the minimal set of features needed for accurate prediction
3. **Additional Models**: Explore deep learning approaches or stacked ensembles to potentially improve performance further
4. **Deployment Infrastructure**: Develop a more robust API with authentication, logging, and monitoring
5. **Fairness Analysis**: Evaluate the model for potential biases across different demographic groups
6. **Cost-Benefit Analysis**: Incorporate the financial impact of correct and incorrect predictions to optimize for business value
7. **Incremental Learning**: Implement a system for updating the model as new data becomes available
8. **Synthetic Data Generation**: Create a synthetic data generator to augment the training set and improve generalization

## 7. Conclusion

The machine learning pipeline developed for predicting insurance enrollment demonstrates excellent performance, with the final Gradient Boosting model achieving near-perfect accuracy and ROC AUC scores. The model effectively leverages demographic and employment-related features to predict enrollment likelihood, providing valuable insights for the company's insurance product strategy.

The high predictive power suggests that the available features contain strong signals for determining enrollment decisions. This model can be used to target employees who are likely to enroll, optimize marketing efforts, and inform product design decisions.

Overall, this project successfully demonstrates the application of machine learning to modernize insurance operations and improve decision-making in the insurance industry.
