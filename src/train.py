import os
from data_loader import load_mnist_data, preprocess_data
from model import build_model

# Function to train the model and save it
def train_and_save_model():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Build the model
    model = build_model()
    
    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/cnn_model.h5')
    print("Model saved to models/cnn_model.h5")

if __name__ == "__main__":
    train_and_save_model()
