
import numpy as np
import pandas as pd

def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    data['pixels'] = data['pixels'].apply(lambda x: np.array([int(pixel) for pixel in x.split()]).reshape(48, 48))
    data['pixels'] = data['pixels'] / 255.0
    train_data = data[data['Usage'] == 'Training']
    val_data = data[data['Usage'] == 'PublicTest']
    test_data = data[data['Usage'] == 'PrivateTest']
    X_train = np.stack(train_data['pixels'].values).reshape(-1, 48, 48, 1)
    y_train = train_data['emotion'].values
    X_val = np.stack(val_data['pixels'].values).reshape(-1, 48, 48, 1)
    y_val = val_data['emotion'].values
    X_test = np.stack(test_data['pixels'].values).reshape(-1, 48, 48, 1)
    y_test = test_data['emotion'].values
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
