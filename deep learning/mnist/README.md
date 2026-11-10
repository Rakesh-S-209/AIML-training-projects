# MNIST Handwritten Digit Dataset

The **Modified National Institute of Standards and Technology** dataset is the classic benchmark for image classification.

---

## Overview

| Property | Description |
|----------|-------------|
| **Source** | NIST (Modified by Yann LeCun et al.) |
| **Total samples** | 70,000 (60,000 train + 10,000 test) |
| **Image size** | 28 × 28 pixels, grayscale |
| **Classes** | 10 digits (0–9) |
| **Format** | 8-bit grayscale (0–255) |

---

## Dataset Structure

```
mnist/
├── data/               # Raw MNIST files (NOT tracked in git)
│   └── MNIST/
│       └── raw/
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte.gz
│           ├── t10k-images-idx3-ubyte.gz
│           └── t10k-labels-idx1-ubyte.gz
└── ANN.ipynb           # Example: ANN & CNN training
```

---

## Usage

### Via `torchvision` (recommended)

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True,  download=True, transform=transform
)
test_dataset  = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
```

### Direct file extraction (advanced)

Raw files follow the IDX format:
- `idx3` files: images (magic number 0x00000803)
- `idx1` files: labels (magic number 0x00000801)

---

## Preprocessing

1. **ToTensor** — scales pixel values to [0, 1]
2. **Normalize** — uses dataset mean=0.1307, std=0.3081 for stable gradients
3. **Flatten** (for ANN) — 28×28 → 784 for dense layers
4. **DataLoader** — batched, shuffled training data

---

## Models

See `ANN.ipynb` for:

- **Phase 1** — Data loading, 8 visualization plots (samples, class balance, mean digits, etc.)
- **Phase 2** — CNN (2 conv blocks + dense head) with Adam + CrossEntropyLoss

---

## License

The dataset is freely available for research and education.

---

## References

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-Based Learning Applied to Document Recognition* (1998).
- [Original NIST link](https://www.nist.gov/srd/nist-special-database-19)
- [HuggingFace dataset](https://huggingface.co/datasets/mnist)
