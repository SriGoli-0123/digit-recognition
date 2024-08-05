# Digit Recognition with CNN

This project implements a digit recognition system using a Convolutional Neural Network (CNN) on the MNIST dataset.

## Project Structure

- `data/`: Contains dataset files.
- `models/`: Stores trained models.
- `src/`: Contains source code for data loading, model definition, training, and prediction.
- `README.md`: Project description.
- `requirements.txt`: List of dependencies.
- `run.sh`: Script to set up and run the project.

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x

### Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/digit-recognition.git
   cd digit-recognition

2. Install dependencies:

   ```sh
   pip install -r requirements.txt

3. Load the dataset:

   ```sh
   python src/data_loader.py
   
4. Train the model:

   ```sh
   python src/train.py

5. Make predictions:

   ```sh
   python src/predict.py

6. **Dependencies (`requirements.txt`)**

   ```sh
   tensorflow>=2.0.0
   numpy

7. **Run Script (`run.sh`)**

   ```sh
    #!/bin/bash
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Train the model
    python src/train.py
    
    # Make predictions
    python src/predict.py

## Result

Number of epochs: 20, Accuracy: 99.33% on MNIST dataset
