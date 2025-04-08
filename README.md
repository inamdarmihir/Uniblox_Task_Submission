# Insurance Enrollment Prediction

This project builds a machine learning pipeline to predict whether employees will opt in to a new voluntary insurance product based on demographic and employment-related data.

## Project Overview

The goal of this project is to analyze employee data and build a predictive model that can determine the likelihood of an employee enrolling in a voluntary insurance product. This is part of an internal pilot for a company modernizing insurance using machine learning.

## Dataset

The dataset consists of synthetic census-style employee data with approximately 10,000 rows. The features include:

- `employee_id`: Unique identifier for each employee
- `age`: Age of the employee
- `gender`: Gender of the employee
- `marital_status`: Marital status of the employee
- `salary`: Annual salary of the employee
- `employment_type`: Type of employment (Full-time, Part-time, Contract)
- `region`: Geographic region of employment
- `has_dependents`: Whether the employee has dependents
- `tenure_years`: Years of employment at the company
- `enrolled`: Target variable (1 for enrolled, 0 for not enrolled)

## Project Structure

```
ml_insurance_prediction/
├── employee_data.csv        # Original dataset
├── main.py                  # Main script to run the complete pipeline
├── train_final_model.py     # Script to train and save the final model
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb  # EDA notebook
│   ├── 02_data_preprocessing.ipynb         # Data preprocessing notebook
│   └── 03_model_development.ipynb          # Model training notebook
├── src/
│   ├── data_preprocessing.py  # Module for data loading and preprocessing
│   ├── model.py               # Module for model training and evaluation
│   └── models/                # Directory for saved models
│       ├── final_model.joblib    # Saved final model
│       └── preprocessor.joblib   # Saved preprocessor
├── report.md                # Detailed project report
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/insurance-enrollment-prediction.git
cd insurance-enrollment-prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Final Model

To train and save the final model:

```bash
python train_final_model.py
```

This will:
1. Load and preprocess the data
2. Train the final model with optimized hyperparameters
3. Evaluate the model performance
4. Save the model and preprocessor to the `models` directory

### Running the Complete Pipeline

To run the complete pipeline including data preprocessing, model training, and evaluation:

```bash
python main.py
```

### Using the API

To start the FastAPI server for making predictions:

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

## Model Performance

The final model (Gradient Boosting Classifier) achieves the following metrics on the test set:
- Accuracy: ~85%
- ROC AUC: ~0.89
- F1 Score: ~0.84

For detailed performance metrics and analysis, please refer to the `report.md` file.

## Project Report

A detailed project report can be found in `report.md`, which includes:
- Data exploration and analysis
- Feature engineering process
- Model selection and hyperparameter tuning
- Performance evaluation
- Key findings and recommendations

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
