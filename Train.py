import argparse
import numpy as np
from model import build_alroh_model
from dfo import DragonflyOptimizer
from utils import load_processed_data

def train_model(data_path, model_save_path):
    """Train ALROH model with optional DFO optimization"""
    X_train, y_train = load_processed_data(data_path)
    
    # Hyperparameter optimization
    dfo = DragonflyOptimizer()
    best_params = dfo.optimize(X_train, y_train)
    
    # Final model training
    model = build_alroh_model(X_train.shape[1:])
    model.compile(optimizer=Adam(learning_rate=best_params['lr']),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
                       batch_size=best_params['batch_size'],
                       epochs=100,
                       validation_split=0.2)
    
    model.save(model_save_path)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed data")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    args = parser.parse_args()
    
    train_model(args.data, args.output)
