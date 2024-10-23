## File Structure and Descriptions

This repository contains two main models which have been implemented for the purpose of my master's thesis: **ANNBFIS** and **CBNFS**. Each model is accompanied by its own set of data preparation and evaluation files. Below is a breakdown of the key files:

### 1. **ANNBFIS (Artificial Neural Network Based on Fuzzy Inference System)**
The **ANNBFIS** model combines neural network learning with fuzzy logic to improve classification performance by using a hybrid approach. The files related to this model are:

- **`annbfis.py`**:  
  This file implements the core logic of the ANNBFIS model, integrating Artificial Neural Networks (ANN) with a Fuzzy Inference System (FIS). It defines the structure, training process, and evaluation methods for the model.
  
- **`data_preparation_annbfis.py`**:  
  This script is responsible for preparing the input data specifically for the ANNBFIS model. It includes data preprocessing steps such as normalization, feature selection, and splitting the data into training and testing sets.
  
- **`evaluate_annbfis.ipynb`**:  
  A Jupyter notebook used to evaluate the performance of the ANNBFIS model. 

### 2. **CBNFS (Classifier Based on Neuro-Fuzzy System)**
The **CBNFS** model leverages a neuro-fuzzy approach for classification, combining the learning capability of neural networks with fuzzy logic rules. It is a modification of original ANNBFIS model. The files associated with this model are:

- **`cbnfs.py`**:  
  This file contains the implementation of the CBNFS model, a classifier that integrates neural network learning with fuzzy logic rules. It includes model architecture, training, and testing procedures, following the Direct Training Direct Testing (DTDT) method. DTDT improves supervised learning by using fuzzy scores for training, selecting patterns with the highest diagnostic value, and evaluating the model with predefined class labels.

- **`data_preparation.py`**:  
  This script prepares the dataset for the CBNFS model, performing necessary preprocessing tasks such as data cleaning, transformation, and creating train-test splits.

- **`evaluate_cbnfs.ipynb`**:  
  A Jupyter notebook used to evaluate the CBNFS model's classification performance. **Note**: The evaluation process relies on fuzzy scores that cannot be publicly shared, meaning the full evaluation can only be performed with access to these proprietary scores.

Each model follows its own data preparation and evaluation flow to ensure the proper preprocessing steps and tailored evaluation metrics are applied.
