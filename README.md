# 📚 AIML Training Projects

A comprehensive collection of machine learning and deep learning projects for learning and experimentation purposes.

---

## 📂 Folder Structure

```
AIML-training-projects/
├── .gitignore
├── README.md (this file)
├── deep learning/          # Neural network projects
│   ├── cifar/             # CIFAR-10 image classification
│   │   ├── cifar.ipynb    # Main training notebook
│   │   ├── cifar.py       # Python script version
│   │   ├── cifar10_high_accuracy_model.keras  # Best trained model
│   │   ├── mobilenetv2_cifar10.keras          # MobileNetV2 model
│   │   ├── custom_cnn.keras                   # Custom CNN model
│   │   └── README.md      # CIFAR-10 project details
│   └── mnist/             # MNIST handwritten digit recognition
│       ├── ANN.ipynb      # Neural network training
│       └── README.md      # MNIST project details
└── machine learning/      # Classical ML projects
    ├── supervised learning/
    │   ├── customer churn/           # Telco churn prediction
    │   ├── customer support intent/  # Intent classification
    │   ├── employee salary prediction/  # Income prediction
    │   ├── housing california/       # House price regression
    │   └── OCR/                      # Optical Character Recognition
    └── unsupervised learning/
        └── customer segmentation/    # Customer clustering
```

---

## 🎯 Active Deep Learning Projects

### 1. 🚀 CIFAR-10 Image Classification

**Goal**: Classify 32×32 RGB images into 10 object categories

| Model | Architecture | Parameters | Best Accuracy | Training Time |
|-------|-------------|------------|---------------|---------------|
| **Custom CNN** | 5-layer CNN (PyTorch) | 1.34M | ~80% | ~15 min |
| **MobileNetV2** | Transfer Learning (TF) | 2.42M | ~82% | ~10 min |

**Architecture**:
```
Custom CNN: 3→32→64→128 channels with MaxPool and Dropout
MobileNetV2: Pretrained ImageNet backbone + Dense(64) + Dense(10)
```

**Features**:
- ✅ Model freezing for faster training
- ✅ 5 epochs for quick iteration
- ✅ Side-by-side performance comparison
- ✅ Gradio web interface for testing
- ✅ Training curves visualization

![CIFAR-10 Training Curves](deep%20learning/cifar/training_curves.png)

### 2. 🎯 MNIST Digit Recognition

**Goal**: Classify 28×28 grayscale handwritten digits

| Model | Architecture | Best Accuracy |
|-------|-------------|---------------|
| **ANN** | 4-layer Neural Network | 97.8% |

---

## 📊 Supervised Learning Projects

| Project | Type | Algorithm | Accuracy |
|---------|------|-----------|----------|
| **Customer Churn** | Classification | CatBoost | 81.26% |
| **Employee Salary** | Classification | Random Forest | 85%+ |
| **Customer Support Intent** | Classification | BERT-based | ~90% |
| **OCR** | Classification | CNN | ~95% |
| **Housing California** | Regression | Random Forest | RMSE ~$50K |

---

## 🧪 Dataset Sources

| Dataset | Source | Size |
|---------|--------|------|
| **CIFAR-10** | CIFAR | 60K images (10 classes) |
| **MNIST** | NIST (Yann LeCun) | 70K images (10 classes) |
| **Adult Income** | UCI | 48K records |
| **Customer Churn** | IBM Watson | 7K records |
| **Customer Support** | UCI | 22K tickets |

---

## 🛠️ Technologies Used

### Deep Learning
- **PyTorch**: Custom CNN implementation
- **TensorFlow/Keras**: MobileNetV2, ResNet50
- **Keras Tuner**: Hyperparameter optimization

### Classical ML
- **scikit-learn**: Random Forest, SVM, Logistic Regression
- **CatBoost**: Gradient boosting
- **XGBoost**: Extreme gradient boosting

### Deployment
- **Gradio**: Web interface for model testing
- **Flask**: API development (future)

---

## 📈 Performance Metrics

### CIFAR-10 Results
- Custom CNN: 80% validation accuracy
- MobileNetV2: 82% validation accuracy (faster training)
- Both models use **model freezing** for efficiency

### Training Optimizations
- ✅ Frozen base layers for transfer learning
- ✅ Reduced epochs (5 instead of 15)
- ✅ Learning rate scheduling
- ✅ Early stopping

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch torchvision torchaudio tensorflow keras matplotlib scikit-learn gradio

# Run training notebook
jupyter notebook deep\ learning/cifar/cifar.ipynb

# Launch Gradio interface (after training)
python -c "import gradio; gradio.Interface(...).launch(share=True)"
```

---

## 📝 Notes

1. All models are trained with **frozen base layers** for faster training
2. Training reduced to **5 epochs** for quick iteration
3. MobileNetV2 typically outperforms custom CNN with less training time
4. Gradio interface provides easy model testing via web

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 📄 License

This project is for educational purposes only.
- **Housing California**: UCI Machine Learning Repository

---

## 🛠️ Technologies

- **Python** (3.11+)
- **PyTorch** (deep learning)
- **TensorFlow/Keras** (deep learning)
- **scikit-learn** (classical ML)
- **pandas**, **numpy** (data handling)
- **matplotlib**, **seaborn**, **plotly** (visualization)
- **Streamlit** (ML web apps)

---

## 📝 Notes

- Raw datasets (MNIST, CIFAR) are not committed to git (add to `.gitignore`)
- Model weights are saved in each project folder (e.g., `digit_cnn.pth`, `cifar10_high_accuracy_model.keras`)
- Each project includes a `README.md` with details

---

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt` (or per-project requirements)
3. Run any notebook or script to train models

---

## 📄 License

All projects are for educational/training purposes.
