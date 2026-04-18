🚀 Fashion MNIST Classification & Model Optimization

An end-to-end machine learning pipeline for image classification on the **Fashion MNIST dataset**, focusing on **dimensionality reduction, model comparison, and performance optimization**.

---

📌 Overview

This project builds a complete ML workflow to classify fashion images using traditional machine learning techniques. It emphasizes:

* Efficient preprocessing of high-dimensional image data
* Dimensionality reduction using PCA
* Model optimization via cross-validation
* Comparative analysis of different algorithms

---

## 🧠 Key Results

* ✅ **Logistic Regression Accuracy:** 85.95%
* ✅ **Gaussian Naive Bayes Accuracy:** 78.27%
* ✅ **Top-3 Accuracy:** ~98%
* ✅ Significant performance improvement using PCA

---

## ⚙️ Pipeline

### 1. Data Preprocessing

* Flattened image data (28×28 → 784 features)
* Feature scaling using `StandardScaler`
* Dataset transformation for efficient ML training

---

### 2. Dimensionality Reduction

* Applied **Principal Component Analysis (PCA)**
* Reduced dimensionality while preserving variance
* Improved training efficiency and model performance

---

### 3. Model Training

Implemented and compared:

* Logistic Regression (multinomial)
* Gaussian Naive Bayes

Used:

* GridSearchCV
* 3-fold cross-validation
* Hyperparameter tuning

---

### 4. Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* Top-3 Accuracy
* Training time comparison

---

## 📊 Results Summary

| Model               | Accuracy | F1 Score | Top-3 Accuracy |
| ------------------- | -------- | -------- | -------------- |
| Logistic Regression | 0.8595   | ~0.858   | ~0.982         |
| GaussianNB          | 0.7827   | ~0.782   | ~0.953         |

---

## 📁 Project Structure

```
.
├── preprocess_fmnist.py
├── Fashion_MNIST_Classification_and_Model_Comparison.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/MadhavKathuria826/Fashion-MNIST-Image-Classification-System.git
cd Fashion-MNIST-Image-Classification-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run preprocessing

```bash
python preprocess_fmnist.py
```

### 4. Run notebook

Open:

```
Fashion_MNIST_Classification_and_Model_Comparison.ipynb
```

---

## 📦 Dataset

This project uses the **Fashion MNIST dataset**.

👉 Download from:
https://github.com/zalandoresearch/fashion-mnist

Place the dataset files in the project directory before running preprocessing.

---

## 💡 Insights

* PCA significantly reduces computational cost with minimal accuracy loss
* Logistic Regression performs strongly on reduced feature space
* Simpler models (like Naive Bayes) are faster but less accurate
* Dimensionality reduction is critical for high-dimensional image data

---

## 🎯 Future Improvements

* Deep learning models (CNNs) for higher accuracy
* Advanced feature extraction techniques
* Real-time inference system

---

## 👨‍💻 Author

**Madhav Kathuria**
B.Tech CSE, South Asian University

---

## ⭐ If you found this useful

Give it a star ⭐ — it helps!
