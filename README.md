# DeepLearning-On-UWB-Data-for-Beamforming
 This is a repository designed to present my findings and research work, along with a simulation of how UWB data will be used by Deep Learning algorithms to derive the location of smart devices inside a room.

## UWB Data Analysis and Prediction

This repository contains scripts for generating synthetic Ultra-Wideband (UWB) data, training neural network models, and implementing backpropagation algorithms to predict smartphone locations based on UWB measurements.

### Files Overview

1. **data-generator.py**

   This script generates synthetic UWB data by simulating distances between a smartphone and three anchor points in a room. It then saves the data to a CSV file.

2. **specific-example-bp.py**

   Demonstrates a specific example of predicting smartphone locations using a neural network trained with backpropagation. It loads the generated data, preprocesses it, builds and trains the neural network model, and finally predicts smartphone locations.

3. **backpropagation-algo.py**

   Implements a more detailed backpropagation algorithm. It loads the data, splits it into training and testing sets, builds and trains a neural network model, evaluates the model, and verifies its authenticity by predicting a subset of test data.

### Instructions

1. **Generating Synthetic UWB Data**

   Run `data-generator.py` to generate synthetic UWB data. Adjust parameters such as room dimensions, anchor coordinates, and the number of data points as needed.

2. **Specific Example with Backpropagation**

   Execute `specific-example-bp.py` to see a specific example of predicting smartphone locations using a neural network trained with backpropagation.

3. **Backpropagation Algorithm Implementation**

   Run `backpropagation-algo.py` for a detailed implementation of the backpropagation algorithm. This script includes data preprocessing, model training, evaluation, and verification of prediction accuracy.

### Dependencies

- **numpy**: For numerical computing.
- **pandas**: For data manipulation and CSV handling.
- **scikit-learn**: For data preprocessing and model evaluation.
- **keras**: For building and training neural network models.

### Usage

1. Ensure you have the necessary dependencies installed. You can install them via pip:

   ```
   pip install numpy pandas scikit-learn keras
   ```

2. Run the scripts in the order described above.

### Notes

- Adjust parameters and hyperparameters in the scripts as needed for your specific use case.
- Ensure that the generated synthetic data adequately represents your real-world scenario for accurate model training and testing.
- Experiment with different neural network architectures and optimization algorithms for improved prediction performance.
