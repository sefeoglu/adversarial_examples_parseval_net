## Using Adversarial Examples, to improve the robustness of Eye-State Classification :eyes:

### Aim:

Using adversarial examples, the project aims to improve the robustness and accuracy of a machine learning model which detects the eye-states against small perturbation of an image, and to solve the misclassification problem caused by natural transformation.

### Definition

The safe control between vehicle and driver is a significant prerequisite for automated driving whether the driver is able to take a control can be evaluated using eye state detection. It is important that such systems are robust against changing real world conditions like lighting and natural transformation. These real world conditions, which might not be aware of human, lead to fool a machine learning or deep learning model. Deep neural networks which the kind of machine learning models that have recently result in dramatic performance improvements in a wide range of applications are vulnerable to tiny perturbations of their inputs (images). This leads to misclassification problem, namely error in the accuracy of the model. Adversarial examples are specialised inputs created with the purpose of confusing a neural network, resulting in the misclassification of a given input.These notorious inputs are indistinguishable to the human eye, but cause the network to fail to identify the contents of the image. Additionally, the aim of adversal examples is to disturbed the well trained machine learning model. However, small adversarial perturbation should not result in a significant impact on the out of the model for a trained and robust machine learning model. Consequently, generating adversarial perturbation as negative training examples can improve the robustness of the model.
With respect to our dataset, images of eyes consist of various  eye-states which are labelled as open, partially open, closed, and not visible. Natural transformations like angle of input images, viewpoints might lead to misclassification problem in machine learning model. The machine learning model has sensitive measurement to decide the eye-states, so small perturbation of an image fools the deep learning model.


### Methodologies:

* Develop Wide Residual Network and Parseval Network 
* Train Neural Networks using training dataset
* Generate Adversarial Examples from pre-training neural networks using Fast Gradient Sign Method
* Augmenting training data with adversarial examples
* Train new model.

### Neural Network Models

#### Wide Residual Network

* Baseline of the Model

#### Parseval Network

* Orthogonality Constraint
* Lipschitz constant


#### Adversarial Examples

##### Fast Gradient Sign Method

### Evaluation

* To evaluate the result of the neural network, Signal to Noise Ratio (SNR) is used as metric.

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
