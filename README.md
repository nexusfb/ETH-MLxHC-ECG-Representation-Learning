# Machine Learning for Health Care - ETH Zurich 
# Project 2 - Group72
# Time Series and Representation Learning for ECG Classification

This repository contains the implementation of **Project 2** for the course **Machine Learning for Healthcare**, focused on time series analysis and representation learning for classifying electrocardiogram (ECG) signals. We explore different machine learning techniques, from classic models to deep learning architectures, using the PTB Diagnostic ECG Database and the MIT-BIH Arrhythmia Dataset for transfer learning.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Contributors](#contributors)

## Introduction
In this project, we apply supervised learning techniques and representation learning to classify ECG time series as either healthy or abnormal. We explore various machine learning models, including:
- Classic machine learning algorithms (Random Forest, Logistic Regression)
- Recurrent Neural Networks (LSTM, Bidirectional LSTM)
- Convolutional Neural Networks (CNN)
- Transformer-based models for time series data

The project also involves transfer learning and self-supervised representation learning using the MIT-BIH dataset to enhance classification performance on the PTB dataset.

## Project Structure
├── data/ # Contains the processed ECG data (train/test) 

├── models/ # Model implementations (RNNs, CNNs, Transformers) 

├── results/ # Performance metrics and plots 

├── notebooks/ # Jupyter notebooks for analysis 

└── README.md # This file

## Datasets
We utilize the following datasets for training and evaluation:
- **PTB Diagnostic ECG Database**: Contains ECG recordings from healthy individuals and patients with various heart conditions.
- **MIT-BIH Arrhythmia Dataset**: Used for transfer learning, this dataset contains annotated ECG signals of different arrhythmias.

## Models
We implemented several models to evaluate their performance in classifying ECG signals:
1. **Random Forest** and **Logistic Regression**: Classic machine learning methods, trained both on raw data and with engineered features.
2. **LSTM and Bidirectional LSTM**: Recurrent neural networks designed to capture temporal dependencies in the ECG signals.
3. **CNN and Residual CNN**: Convolutional neural networks applied to capture local patterns in time series data.
4. **Transformer-based models**: Explores attention mechanisms to learn long-range dependencies in the ECG sequences.

## Results
| Model                 | Balanced Accuracy |
|-----------------------|-------------------|
| Random Forest (Raw)    | 94.99%            |
| LSTM                  | 81.13%            |
| Bidirectional LSTM     | 86.38%            |
| Residual CNN           | 97.63%            |
| Transformer            | 94.29%            |

## Usage
To train the models and reproduce the results:
1. Clone the repository:
   ```bash
   git clone https://github.com/ilboglions/Project2_ML4HC_Group72
   cd Project2_MLHC_Group72
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    This command will install all the required Python packages specified in the `requirements.txt` file.
   
3. **Execute jupyter notebooks**

## Contributors
- Federica Bruni
- Mateo Boglioni
- Paula Momo Cabrera

