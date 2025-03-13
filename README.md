# House Price Prediction

This repository contains the code for a house price prediction model, from data exploration and preprocessing to model training and deployment.

## Overview

The project aims to predict house prices based on various features. The workflow includes:

1.  **Data Exploration:** Understanding the dataset's characteristics.
2.  **Data Preprocessing:** Cleaning and preparing the data for model training.
3.  **Model Training:** Training several regression models and selecting the best one.
4.  **Deployment:** Deploying the trained model as a web service using Flask.

The service is hosted live on Render: [https://houseprice-3mmi.onrender.com](https://houseprice-3mmi.onrender.com)

## Repository Structure

```
├── app.py # Flask application for serving the model
├── preprocessing/ # Directory for preprocessing scripts
│ ├── explore.py # Script for data exploration and analysis
│ └── preprocessing.py # Script for data preprocessing
├── training/ # Directory for training scripts and results
│ ├── train.py # Script for model training and evaluation
│ ├── model_results.csv # CSV file containing model evaluation metrics
│ └── plots/ # Directory containing generated plots
│ └── ... # Various plots for data analysis and model comparison
├── dataset
│ ├── Housing.csv # Original dataset
│ └── Housing_Preprocessed.csv # Preprocessed dataset
└── README.md # This file
```

## Workflow Details

### 1. Data Exploration (`explore.py`)

The `explore.py` script performs initial data analysis to understand the dataset. Key steps include:

* Loading the dataset (`Housing.csv`).
* Displaying the first few rows and dataset information.
* Identifying numerical and categorical features.
* Checking for missing values.
* Generating pair plots to visualize feature relationships.

### 2. Data Preprocessing (`preprocessing.py`)

The `preprocessing.py` script focuses on preparing the data for model training. The main steps are:

* Handling duplicate rows.
* Checking for null values.
* Converting categorical features into numerical using one-hot encoding and dummy variable creation.
* Removing outliers using the IQR (Interquartile Range) method.
* Saving the preprocessed dataset (`Housing_Preprocessed.csv`).

### 3. Model Training (`train.py`)

The `train.py` script involves training and evaluating different regression models. The process includes:

* Loading the preprocessed dataset.
* Splitting the data into training and testing sets.
* Training the following models:
    * Linear Regression
    * Decision Tree Regressor
    * Random Forest Regressor
    * XGBoost Regressor
* Evaluating model performance using RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² (R-squared).
* Performing hyperparameter tuning using GridSearchCV to optimize model performance.
* Comparing model performance before and after tuning.
* Saving the best-performing model (`best_model.pkl`) using joblib.
* Saving model results to `model_results.csv`.
* Generating and saving various plots for model comparison and performance visualization.

The `model_results.csv` file contains the evaluation metrics for the models.

### 4. Model Deployment (`app.py`)

The `app.py` script uses Flask to create a web service for the trained model[cite: 1].

* It loads the trained model (`best_model.pkl`).
* It defines a Flask API with two routes:
    * `/`:  Returns the index.html page.
    * `/predict`:  Accepts house feature data as a JSON payload, preprocesses the input, makes a prediction using the loaded model, and returns the predicted price as a JSON response[cite: 1].
* The `preprocess_input` function handles the conversion of the simplified input data from the request to the format expected by the model[cite: 1].
* The application is configured to run on the host '0.0.0.0' and the port specified by the environment variable `PORT` (or defaults to 5000)[cite: 1].

## Model Performance

The model training process involved hyperparameter tuning and evaluation of several regression models. The table below shows the performance of the models *after* hyperparameter tuning:

| Model               | RMSE        | MAE         | R²          |
| ------------------- | ----------- | ----------- | ----------- |
| Linear Regression   | 1378145.54  | 927361.67   | 0.6014      |
| Decision Tree       | 1572972.16  | 1025281.25  | 0.4808      |
| Random Forest       | 1397444.46  | 916154.91   | 0.5898      |
| XGBoost             | 1477782.72  | 967272.44   | 0.5407      |

The best-performing model was **Linear Regression**, based on the R² score. For detailed model performance metrics, please refer to `model_results.csv`.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file. A basic one would include: `flask`, `joblib`, `pandas`, `scikit-learn`, `numpy`)*
4.  **Run the Flask application:**
    ```bash
    python app.py
    ```
5.  **Access the application:**
    Open your browser and go to `http://127.0.0.1:5000/`

## How to Use the Prediction Service

The prediction service is available at [https://houseprice-3mmi.onrender.com](https://houseprice-3mmi.onrender.com).

To get a prediction, send a POST request to `/predict` with a JSON payload containing the house features.

**Example JSON Payload:**

```json
{
    "features": {
        "area": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": true,
        "guestroom": false,
        "basement": false,
        "hotwaterheating": false,
        "airconditioning": true,
        "parking": 1,
        "prefarea": true,
        "furnishingstatus": "furnished"
    }
}
```

**Response:**

```json
{
    "predicted_price": 4500000.0
}
```

## Model Performance

The model training process involved hyperparameter tuning and evaluation of several regression models. The best-performing model was selected based on its R² score.  For detailed model performance metrics, please refer to `model_results.csv`.

## Contributing

Contributions to this project are welcome. Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.
```

**Key Improvements and Explanations:**

* **Clear Structure:** The README is organized with headings and subheadings for better readability.
* **Detailed Workflow:** Each script's purpose and key steps are explained, referencing the provided files.
* **Code Snippets:** Code snippets are included to illustrate how to run the application and use the prediction service.
* **Emphasis on Render Deployment:** The Render hosting is prominently mentioned with a direct link.
* **Requirements Note:** It prompts the user to create a `requirements.txt` file and provides a basic example.
* **Model Results Reference:** It guides the user to the `model_results.csv` file for detailed performance metrics.
* **Contribution and License:** Standard sections for open-source projects.
* **File Paths:** Uses consistent file paths to match your project structure.
* **Input Data Explanation:** It explains how the `preprocess_input` function works in `app.py`[cite: 1].










