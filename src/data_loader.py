import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Function to download and save the MNIST dataset
def download_and_save_mnist(file_path='data/mnist.npz'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    np.savez(file_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Dataset saved to {file_path}")

# Function to load the dataset from a .npz file
def load_mnist_data(file_path='data/mnist.npz'):
    with np.load(file_path) as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
    return (x_train, y_train), (x_test, y_test)

# Function to preprocess the data
def preprocess_data(x, y):
    x = x.astype('float32') / 255
    x = np.expand_dims(x, -1)
    y = to_categorical(y, 10)
    return x, y

if __name__ == "__main__":
    # Download and save the dataset
    download_and_save_mnist()
    # Load and preprocess the dataset
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
