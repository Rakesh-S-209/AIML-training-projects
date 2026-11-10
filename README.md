# AIML Training Projects

A collection of machine learning and deep learning projects for learning purposes.

---

## 📂 Folder Structure

```
AIML-training-projects/
├── .gitignore
├── README.md (this file)
├── deep learning/          # Neural network projects
│   ├── cifar/             # CIFAR-10 image classification
│   └── mnist/             # MNIST handwritten digit recognition
└── machine learning/      # Classical ML projects
    ├── supervised learning/
    │   ├── customer churn/           # Telco churn prediction
    │   ├── customer support intent/  # Intent classification
    │   ├── employee salary prediction/  # Income prediction
    │   ├── housing california/       # House price regression
    │   ├── OCR/                      # Optical Character Recognition
    │   └── ...
    └── unsupervised learning/
        └── customer segmentation/    # Customer clustering
```

---

## 🎯 Projects

| Project | Type | Description |
|---------|------|-------------|
| **MNIST** | Deep Learning | Handwritten digit classification (28×28 grayscale) |
| **CIFAR-10** | Deep Learning | 10-class object recognition (32×32 RGB) |
| **Customer Churn** | Supervised | Predict telco customer attrition |
| **Employee Salary** | Supervised | Income prediction (>$50K or not) |
| **Customer Support Intent** | Supervised | Intent classification for support tickets |
| **OCR** | Supervised | Document image text extraction |
| **Housing California** | Supervised | House price regression |
| **Customer Segmentation** | Unsupervised | Market segmentation via clustering |

---

## 📊 Dataset Sources

- **MNIST**: NIST (Yann LeCun)
- **CIFAR-10**: Canadian Institute for Advanced Research
- **Adult Income**: UCI Machine Learning Repository
- **Customer Churn**: IBM Watson Analytics
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
