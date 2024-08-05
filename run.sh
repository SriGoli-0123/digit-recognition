!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Make predictions
python src/predict.py
