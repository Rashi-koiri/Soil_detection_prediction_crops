import csv
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Crop_recommendation.csv")
    
    print(f"Loading dataset from: {csv_path}")
    
    features = []
    labels = []
    crop_data = {}
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = float(row['N'])
            p = float(row['P'])
            k = float(row['K'])
            temp = float(row['temperature'])
            hum = float(row['humidity'])
            ph = float(row['ph'])
            rain = float(row['rainfall'])
            crop = row['label'].strip()
            
            features.append([n, p, k, temp, hum, ph, rain])
            labels.append(crop)
            
            if crop not in crop_data:
                crop_data[crop] = []
            crop_data[crop].append([n, p, k, temp, hum, ph, rain])
            
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Total records loaded: {X.shape[0]} samples with {X.shape[1]} features.")
    unique_crops = sorted(list(crop_data.keys()))
    print(f"Total unique crop classes ({len(unique_crops)}): {unique_crops}")
    
    # 80/20 train/test split matching Colab
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training split: {X_train.shape[0]} samples | Testing split: {X_test.shape[0]} samples")
    
    # Random Forest Classifier (100 estimators, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Evaluation Accuracy on Unseen Test Fields: {acc * 100:.2f}%")
    
    # Save model file
    model_path = os.path.join(current_dir, "crop_recommendation_model.pkl")
    joblib.dump(model, model_path)
    print(f"Trained model exported successfully to: {model_path}")
    
    # Agronomic averages per crop for real-world agronomy guidance card
    crop_stats = {}
    for crop, rows in crop_data.items():
        arr = np.array(rows)
        crop_stats[crop] = {
            "avg_N": round(float(np.mean(arr[:, 0])), 1),
            "avg_P": round(float(np.mean(arr[:, 1])), 1),
            "avg_K": round(float(np.mean(arr[:, 2])), 1),
            "avg_temp": round(float(np.mean(arr[:, 3])), 1),
            "avg_humidity": round(float(np.mean(arr[:, 4])), 1),
            "avg_ph": round(float(np.mean(arr[:, 5])), 2),
            "avg_rainfall": round(float(np.mean(arr[:, 6])), 1)
        }
        
    stats_path = os.path.join(current_dir, "crop_statistics.json")
    with open(stats_path, "w", encoding='utf-8') as f:
        json.dump(crop_stats, f, indent=2)
    print(f"Agronomic benchmark statistics exported to: {stats_path}")
    print("Training pipeline completed!")

if __name__ == "__main__":
    train()
