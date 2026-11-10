# CIFAR-10 Image Classification Dataset

The **Canadian Institute For Advanced Research** dataset is a standard benchmark for object recognition in natural images.

---

## Overview

| Property | Description |
|----------|-------------|
| **Source** | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **Total samples** | 60,000 (50,000 train + 10,000 test) |
| **Image size** | 32 × 32 pixels, RGB (3 channels) |
| **Classes** | 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Format** | RGB PNG (8-bit per channel) |

---

## Dataset Structure

```
cifar/
├── .gradio/            # Gradio demo artifacts (flagged images, logs)
├── cifar-10-python.tar.gz   # Raw dataset archive
├── cifar10_high_accuracy_model.keras   # Trained Keras model
└── cifar.ipynb         # Training notebook
```

---

## Class Labels

| Index | Label |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## Usage

### Load via `torchvision`

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True,  download=True, transform=transform
)
test_dataset  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
```

### Load via `keras.utils`

```python
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

---

## Preprocessing

1. **ToTensor** — converts H×W×C → C×H×W, scales to [0, 1]
2. **Normalize** — per-channel mean/std (CIFAR-10 specific values)
3. **DataLoader** — batched, shuffled; `num_workers > 0` recommended

---

## Models

See `cifar.ipynb` for:

- Convolutional Neural Network architecture
- Training loop with Adam optimizer
- Learning rate scheduling
- Training & validation curves

---

## License

The dataset is released under the MIT license.

---

## References

- Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009).
- [Official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html)
- [HuggingFace dataset](https://huggingface.co/datasets/cifar10)
