import numpy as np

def create_sequences(features, labels, seq_length):
    """Convert time series data into sequences"""
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

def load_processed_data(path):
    """Load processed numpy data"""
    data = np.load(path)
    return data['X'], data['y']
