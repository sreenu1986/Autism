from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, TimeDistributed

def build_alroh_model(input_shape, num_classes=1):
    """Build the hybrid CNN-LSTM model architecture"""
    model = Sequential([
        TimeDistributed(Conv1D(32, 3, activation='relu'), input_shape=input_shape),
        TimeDistributed(MaxPooling1D(2)),
        TimeDistributed(Conv1D(64, 3, activation='relu')),
        TimeDistributed(MaxPooling1D(2)),
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])
    return model
