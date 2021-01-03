## Adversarial Examples for improving the robustness of Eye-State Classification :eyes:

### Aim

Using adversarial examples, the project aims to improve the robustness and accuracy of a machine learning model which detects the eye-states against small perturbation of an image and to solve the misclassification problem caused by natural transformation.
### Methodologies

* Develop Wide Residual Network and Parseval Network.
* Train Neural Networks using training dataset.
* Train Neural Network using Adversarial Training with AEs.
* Attack the new model with different perturbated test dataset.

### Neural Network Models

#### Wide Residual Network

* Baseline of the Model

#### Parseval Network

* [Orthogonality Constraint](/src/models/Parseval_Networks/constraint.py)
* [Convexity Constraint on Aggregation](/src/models/Parseval_Networks/convexity_constraint.py)

#### Convolutional Neural Network

#### Adversarial Examples

##### Fast Gradient Sign Method
[Examples](src/visualization/Adversarial_Images.ipynb)

### Evaluation

* To evaluate the result of the neural network, Signal to Noise Ratio (SNR) is used as metric.
* Use transferability of AEs to evaluate the models.

## Development 

#### Models:

``` bash

adversarial_examples_parseval_net/src/models
├── FullyConectedModels
│   ├── model.py
│   └── parseval.py
├── Parseval_Networks
│   ├── constraint.py
│   ├── convexity_constraint.py
│   ├── parsevalnet.py
├── _utility.py
└── wideresnet
    └── wresnet.py


```
#### Analysis:
``` bash
├── Adversarial_Images.ipynb
├── BasicDeepNetworkResults_Visualization.ipynb
├── LearningCurves.ipynb
├── Prediction.ipynb
└── SignalToNoiseRatio.ipynb

```

### Result:
![Alt text](src/logs/images/SNR.png?raw=true "Signal to Noise Ratio Results of the model")



![Alt text](src/logs/images/CNN_SNR.png?raw=true "Signal to Noise Ratio Results of the model")



![Alt text](src/logs/images/compare_parseval_res.png?raw=true "Compare ResNet with Parseval")



#### Documentation:
* [Final Presentation](documents/slide/)
* [Final Report](documents/)    -- wip
* [Detailed Expose](documents/Expose) (Accesible)

References
============
[1] Cisse, Bojanowski, Grave, Dauphin and Usunier, Parseval Networks: Improving Robustness to Adversarial Examples, 2017.

[2] Zagoruyko and Komodakis, Wide Residual Networks, 2016.

``` 

@misc{ParsevalNetworks,
  author= "Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, Nicolas Usunier"
  title="Parseval Networks: Improving Robustness to Adversarial Examples"
  year= "2017"
}
```

``` 

@misc{Wide Residual Networks
  author= "Sergey Zagoruyko, Nikos Komodakis"
  title= "Wide Residual Networks"
  year= "2016"
}
```

### Author

Sefika Efeoglu

Research Project, Data Science MSc, University of Potsdam, SS 2020
