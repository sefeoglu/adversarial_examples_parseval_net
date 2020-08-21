## Using Adversarial Examples, to improve the robustness of Eye-State Classification :eyes:

### Aim:

Using adversarial examples, the project aims to improve the robustness and accuracy of a machine learning model which detects the eye-states against small perturbation of an image, and to solve the misclassification problem caused by natural transformation.

### Definition

### Methodologies:


### Neural Network Models

#### Wide Residual Network

* Baseline of the Model

#### Parseval Network

* Orthogonality Constraint
* Lipschitz constant


#### Adversarial Examples

##### Fast Gradient Sign Method



## Development 


#### Models:
``` bash
├── Parseval_network
│   ├── __init__.py
│   └── Parseval_resnet.py
├── Parseval_Networks_OC
│   ├── constraint.py
│   ├── parsnet_oc.py
│   └── README.md
├── README.md
├── _utility.py
└── wideresnet
    └── wresnet.py
```
References
============
[1] Cisse, Bojanowski, Grave, Dauphin and Usunier, Parseval Networks: Improving Robustness to Adversarial Examples, 2017.

[2] Zagoruyko and Komodakis, Wide Residual Networks, 2016.

```

@paper{ParsevalNetworks,
  author= "Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, Nicolas Usunier"
  title="Parseval Networks: Improving Robustness to Adversarial Examples"
  year= "2017"
}
```

```
@paper{Wide Residual Networks
  author= "Sergey Zagoruyko, Nikos Komodakis"
  title= "Wide Residual Networks"
  year= "2016"
}
```
### Author

Sefika Efeoglu

Data Science MSc, University of Potsdam, SS 2020
