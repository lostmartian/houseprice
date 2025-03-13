from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "training/model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(
        "Model file not found. Train and save the model first.")

# Get feature names from the dataset
dataset_path = "dataset/Housing_Preprocessed.csv"
if os.path.exists(dataset_path):
    data = pd.read_csv(dataset_path)
    # First column is the target (price)
    feature_names = data.columns[1:].tolist()
else:
    raise FileNotFoundError(
        "Dataset file not found. Make sure the dataset is available.")


@app.route('/')
def home():
    """Render the home page with input fields."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get simplified input data
        input_data = request.json["features"]

        print(input_data)
        
        # Process the simplified input into the format expected by the model
        processed_data = preprocess_input(input_data)
        
        # Convert to DataFrame with correct order
        input_df = pd.DataFrame([processed_data])[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({"predicted_price": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})


def preprocess_input(input_data):
    """
    Convert simplified input data to the format expected by the model.
    
    Args:
        input_data (dict): Simplified input with fields like 'area', 'bathrooms', etc.
        
    Returns:
        dict: Processed data with one-hot encoded features
    """
    processed = {}
    
    # Copy direct features
    processed['area'] = input_data.get('area', 0)
    processed['mainroad'] = input_data.get('mainroad', False)
    processed['guestroom'] = input_data.get('guestroom', False)
    processed['basement'] = input_data.get('basement', False)
    processed['hotwaterheating'] = input_data.get('hotwaterheating', False)
    processed['airconditioning'] = input_data.get('airconditioning', False)
    processed['prefarea'] = input_data.get('prefarea', False)
    
    # Process furnishing status
    furnishing = input_data.get('furnishingstatus', 'furnished')
    processed['furnishingstatus_semi_furnished'] = furnishing == 'semi_furnished'
    processed['furnishingstatus_unfurnished'] = furnishing == 'unfurnished'
    
    # Process bathrooms (one-hot encoding)
    bathrooms = input_data.get('bathrooms', 1)
    processed['bathrooms_2'] = bathrooms == 2
    processed['bathrooms_3'] = bathrooms == 3
    processed['bathrooms_4'] = bathrooms == 4
    
    # Process stories (one-hot encoding)
    stories = input_data.get('stories', 1)
    processed['stories_2'] = stories == 2
    processed['stories_3'] = stories == 3
    processed['stories_4'] = stories == 4
    
    # Process parking (one-hot encoding)
    parking = input_data.get('parking', 0)
    processed['parking_1'] = parking == 1
    processed['parking_2'] = parking == 2
    processed['parking_3'] = parking == 3
    
    # Process bedrooms (one-hot encoding)
    bedrooms = input_data.get('bedrooms', 3)
    processed['bedrooms_2'] = bedrooms == 2
    processed['bedrooms_3'] = bedrooms == 3
    processed['bedrooms_4'] = bedrooms == 4
    processed['bedrooms_5'] = bedrooms == 5
    processed['bedrooms_6'] = bedrooms == 6
    
    return processed


if __name__ == '__main__':
    app.run(debug=True)
