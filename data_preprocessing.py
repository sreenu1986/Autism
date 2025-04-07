import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import create_sequences

def preprocess_data(input_path, output_path, sequence_length=10):
    """Process raw CSV data into sequences for CNN-LSTM"""
    df = pd.read_csv(input_path)
    
    # Feature engineering and normalization
    features = df.drop(columns=['Class'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = create_sequences(scaled_features, df['Class'].values, sequence_length)
    
    # Save processed data
    np.savez(output_path, X=X, y=y)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Output path for processed data")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output, args.seq_len)
