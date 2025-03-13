import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data = pd.read_csv("../dataset/Housing_Preprocessed.csv")

target_column = data.columns[0]
X = data.drop(columns=[target_column])
y = data[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

# Train and evaluate initial models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

# Print initial evaluation results
print("Initial Model Performance:")
for model, metrics in results.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()

# Create visualizations for model comparison

# Create a directory for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Plot comparison of metrics across models
metrics_df = pd.DataFrame(results).T
metrics_df.reset_index(inplace=True)
metrics_df.rename(columns={'index': 'Model'}, inplace=True)

# Save results to CSV
metrics_df.to_csv("model_results.csv", index=False)
print("Model evaluation results saved to model_results.csv")

# Plot bar charts for each metric
for metric in ['RMSE', 'MAE', 'R²']:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, data=metrics_df)
    plt.title(f'{metric} Comparison Across Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{metric}_comparison.png")
    plt.close()

# Plot actual vs predicted values for each model
for name, model in models.items():
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f"plots/{name.replace(' ', '_')}_actual_vs_predicted.png")
    plt.close()

# Define parameter grids for each model type
param_grids = {
    "Linear Regression": {},  # Linear Regression doesn't have hyperparameters to tune
    
    "Decision Tree": {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2]
    }
}

# Perform hyperparameter tuning for each model
tuned_models = {}
tuned_results = {}

print("\nPerforming hyperparameter tuning for each model...")
for name, model in models.items():
    print(f"\nTuning {name}...")
    
    # Skip tuning for Linear Regression as it doesn't have hyperparameters
    if name == "Linear Regression":
        tuned_models[name] = model
        y_pred = model.predict(X_test)
        
    else:
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='r2', 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        tuned_models[name] = grid_search.best_estimator_
        y_pred = tuned_models[name].predict(X_test)
    
    # Calculate metrics for the tuned model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    tuned_results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

# Print tuned model results
print("\nTuned Model Performance:")
for model, metrics in tuned_results.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()

# Determine the best model based on R² score (higher is better)
best_model_name = max(tuned_results, key=lambda x: tuned_results[x]["R²"])
print(f"\nBest model after tuning: {best_model_name}")
print(f"R² score: {tuned_results[best_model_name]['R²']:.4f}")
print(f"RMSE: {tuned_results[best_model_name]['RMSE']:.4f}")
print(f"MAE: {tuned_results[best_model_name]['MAE']:.4f}")

# Save the best model
best_model = tuned_models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
print(f"Best model ({best_model_name}) saved as best_model.pkl")

# Create comparison plot of before and after tuning
comparison_data = []
for model in models.keys():
    comparison_data.append({
        'Model': model, 
        'Stage': 'Before Tuning', 
        'R²': results[model]['R²'],
        'RMSE': results[model]['RMSE'],
        'MAE': results[model]['MAE']
    })
    comparison_data.append({
        'Model': model, 
        'Stage': 'After Tuning', 
        'R²': tuned_results[model]['R²'],
        'RMSE': tuned_results[model]['RMSE'],
        'MAE': tuned_results[model]['MAE']
    })

comparison_df = pd.DataFrame(comparison_data)

# Plot comparison of R² before and after tuning
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R²', hue='Stage', data=comparison_df)
plt.title('R² Score Before and After Hyperparameter Tuning')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/tuning_comparison_r2.png")
plt.close()

# Plot comparison of RMSE before and after tuning
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='RMSE', hue='Stage', data=comparison_df)
plt.title('RMSE Before and After Hyperparameter Tuning')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/tuning_comparison_rmse.png")
plt.close()
