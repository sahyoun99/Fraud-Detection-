
# 💳 Credit Card Fraud Detection using Self-Organizing Maps and ANN

This project demonstrates an **end-to-end machine learning pipeline** for detecting **fraudulent credit card applications** using **Unsupervised Learning (Self-Organizing Maps)** followed by **Supervised Deep Learning (Artificial Neural Network)**.

## 📌 Project Overview

The process involves two phases:

1. **Unsupervised Learning using Self-Organizing Maps (SOM)**  
   - Identifies potential anomalies or outliers (fraudulent applications) without labeled data.
2. **Supervised Deep Learning using an Artificial Neural Network (ANN)**  
   - Learns from the suspected frauds and generalizes to detect similar patterns in the full dataset.

## 📁 Dataset

- File: `Credit_Card_Applications.csv`  
- Structure: Contains customer application data (columns like ID, credit score, age, income, etc.) and a final column indicating approval (1) or rejection (0).

## 📚 Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn` (for feature scaling)
- `minisom` (for SOM implementation)
- `keras` (for building the ANN)

## 🔍 Steps

### ✅ 1. Load & Preprocess Data

- Load dataset and separate features `X` and label `y`
- Normalize feature values to `[0, 1]` using `MinMaxScaler`

### 🧠 2. Train SOM (Unsupervised Phase)

- Create a 10x10 SOM grid
- Train the SOM to identify outlier nodes (potential frauds)
- Visualize the SOM and manually inspect suspicious nodes

### 🔎 3. Identify Fraudulent Applications

- Extract data points mapped to outlier SOM nodes (e.g., `(2,8)` and `(3,8)`)
- Reverse transform them to their original feature values

### 🤖 4. Build ANN (Supervised Phase)

- Create a binary label `is_fraud` (1 for frauds detected by SOM)
- Standardize the customer features using `StandardScaler`
- Build a simple ANN using `Keras` with:
  - Input Layer: 15 neurons
  - Hidden Layer: 2 neurons with `ReLU`
  - Output Layer: 1 neuron with `Sigmoid`
- Compile with `adam` optimizer and `binary_crossentropy` loss

### 📈 5. Train and Predict

- Train ANN on the labeled data
- Predict the probability of fraud for each application
- Sort and review results

## 🛠 How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn keras minisom
   ```

2. Place the `Credit_Card_Applications.csv` file in the working directory.

3. Run the script:
   ```bash
   python fraud_detection.py
   ```

## 📊 Output

- A color-coded SOM plot highlighting potential frauds
- Printed list of Customer IDs flagged as frauds
- Probabilities of fraud for all customers (sorted)

## 📘 Concepts Used

- **Self-Organizing Maps (SOMs)** for anomaly detection
- **Artificial Neural Networks (ANNs)** for classification
- **Feature Scaling** and **Data Transformation**
- Transition from **Unsupervised to Supervised Learning**

## 🤝 Contributions

Feel free to open issues or pull requests to improve the project!


