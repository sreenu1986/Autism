import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_data_path, results_dir):
    """Evaluate model performance and save metrics"""
    model = load_model(model_path)
    data = np.load(test_data_path)
    X_test, y_test = data['X'], data['y']
    
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Save results
    with open(f"{results_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--output", required=True, help="Output directory for results")
    args = parser.parse_args()
    
    metrics = evaluate_model(args.model, args.data, args.output)
    print("Evaluation Metrics:", metrics)
