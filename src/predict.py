import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_data

def predict():
    _, (x_test, y_test) = load_data()
    model = load_model('../models/cnn_model.h5')
    predictions = model.predict(x_test)
    return np.argmax(predictions, axis=1)

if __name__ == "__main__":
    predictions = predict()
    print(predictions)
