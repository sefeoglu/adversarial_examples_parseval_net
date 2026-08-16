# Adversarial Examples for Improving Eye-State Classification

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Project-orange)](https://github.com/)

This project studies how adversarial examples can be used to improve the robustness of eye-state classification models. The work compares standard training, adversarial augmentation, and adversarial training strategies, with a focus on models such as Wide Residual Networks and Parseval Networks.

## Goals

### First objective
Improve model robustness by incorporating adversarial examples into the training data and evaluating the resulting classifier on clean test data.

### Second objective
Increase robustness and accuracy against small input perturbations by training with adversarial examples and evaluating performance under attacked test data.

## Methodology

- Develop Wide Residual Network and Parseval Network architectures.
- Train neural networks on the eye-state dataset.
- Generate adversarial examples using FGSM and random noise.
- Compare clean-data performance and adversarial robustness.
- Use signal-to-noise ratio (SNR) and transferability as evaluation metrics.

## Models

### Wide Residual Network
A residual CNN baseline used for comparison.

### Parseval Network
A constrained network designed to improve robustness, using:

- [Orthogonality constraint in convolutional layers](src/models/Parseval_Networks/constraint.py)
- [Convexity constraint in aggregation layers](src/models/Parseval_Networks/convexity_constraint.py)

### Visualization and examples

- [Adversarial image notebook](src/visualization/Adversarial_Images.ipynb)
- [ROC curve scripts](src/visualization/ROC_curves.py)
- [Learning curve scripts](src/visualization/Learning_Curves.py)

## Repository layout

```text
adversarial_examples_parseval_net/
├── README.md
├── requirements.txt
├── documents/
│   ├── Expose/
│   ├── Report/
│   └── slide/
├── src/
│   ├── data/
│   ├── logs/
│   ├── models/
│   ├── preprocessing/
│   ├── train/
│   └── visualization/
└── LICENSE
```

## Environment and setup

The project depends on Python scientific, visualization, and deep-learning libraries.

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Current environment note

The current workspace was checked with Python 3.13.12, and the pinned package versions in [requirements.txt](requirements.txt) match the installed environment:

- numpy==2.4.6
- pandas==3.0.3
- scikit-learn==1.9.0
- scipy==1.17.1
- matplotlib==3.11.0

TensorFlow and CleverHans may require an environment-specific compatible install depending on the Python version and platform. The project code uses TensorFlow/Keras APIs throughout the training and model definitions.

## Results and documentation

- [Results for FGSM experiments](src/logs/AEModels/)
- [Results for random-noise experiments](src/logs/RandomNoisemodels/)
- [Additional training and analysis outputs](src/logs/)
- [Final presentation](documents/slide/)
- [Final report](documents/Report/)
- [Detailed expose](documents/Expose)

## References

[1] Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y., and Usunier, N. Parseval Networks: Improving Robustness to Adversarial Examples. 2017.

[2] Zagoruyko, S. and Komodakis, N. Wide Residual Networks. 2016.

## Author

Sefika Efeoglu

Research Project, Data Science MSc, University of Potsdam
