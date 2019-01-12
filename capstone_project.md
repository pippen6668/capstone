# Machine Learning Engineer Nanodegree
## Capstone Project
yfchiena  
January 3, 2019

## I. Definition

### Project Overview

**Dogs vs. Cats** is a competition topic for kaggle in 2013. The goal is to train a model to distinguish whether it is a cat or a dog from a given picture. This is a problem in the field of computer vision, and also a binary classifaction problem.

Deep learning is one of the most important breakthroughs in the field of artificial intelligence in the past decade. It has achieved great success in speech recognition, natural language processing, computer vision, image and video analysis, multimedia, and many other fields.

Hinton's research team proposed the AlexNet architecture and won the ImageNet Image Classification Competition in 2012, which the accuracy is more than 10% above the second place. Since then, the annual champions are all methods of neural networks.


| Year | Model | Top 5 error rate |
| ------ | ------: | ------: |
| 2012 | AlexNet | 15.4%	 |
| 2013 | ZFNet | 11.2%	 |
| 2014 | GoogLeNet | 6.7%	 |
| 2015 | ResNet | 3.57%	 |
| 2016 | GBD-Net | 2.99%	 |
| 2017 | SeNet | 2.3%	 |


### Problem Statement

Input: A color photo.

Output: The probability(0~1) of a cat or a dog.

The project needs to identify cats and dogs, which is essentially a binary classification problem. Corresponding to supervised learning is to use the existing pictures with labels, and after the training is completed, the unlabeled pictures are classified. Therefore, it is also possible to solve this problem using a supervised learning method such as SVM. But due to the excellent performance of deep learning, I will choose the deep learning method to complete this project.

### Metrics

A standard evaluation formula was proposed from Kaggle.

![](https://latex.codecogs.com/gif.latex?LogLoss=-\frac{1}{n}\sum_{i=1}^n[y_ilog(\hat{y}_i)+(1-y_i)log(1-\hat{y}_i)])

where,

* ![](https://latex.codecogs.com/gif.latex?n) is the number of images in the test set
* ![](https://latex.codecogs.com/gif.latex?\hat{y}_i) is the predicted probability of the image being a dog
* ![](https://latex.codecogs.com/gif.latex?y_i) is 1 if the image is a dog, 0 if cat
* ![](https://latex.codecogs.com/gif.latex?log()) is the natural (base e) logarithm


'Log loss' is also called 'Logistic regression loss' or 'Cross-entropy loss', which is one of the commonly used evaluation methods in classfication problem. 


## II. Analysis

### Data Exploration and Exploratory Visualization

Project data sets can be downloaded from kaggle.

The training set has a total of 25,000 images(12,500 cats and 12500 dogs), which are distinguished by file names. The tag name of each photo contains the tag of dog or cat.

![Training data with label](https://github.com/pippen6668/capstone/blob/master/images/dog%20and%20cat.png)

The test set has a total of 12,500 images and there are no tags in the file names.

![Testing data without label](https://github.com/pippen6668/capstone/blob/master/images/test%20sample.png)

You can see the distribution of the image size as shown below. The distribution shows that the image size of the training set is scattered. To make the images size are consistent, I will use Keras-ImageDataGeneratro to solve this problem.

![Image size distribution](https://github.com/pippen6668/capstone/blob/master/images/image%20size.png)

The photos were taken in everyday life. Shooting techniques are free, taking the picture of the dog as an example.

Dogs appear alone, multiple dogs, humans enter the mirror, etc.

![](https://github.com/pippen6668/capstone/blob/master/images/different%20dog.png)

Due to  the data is presented as an image, so it can be quickly swept by the naked eye, and some abnormal images are obviously found.

![Abnormal images](https://github.com/pippen6668/capstone/blob/master/images/abnormal.png)

### Algorithms and Techniques

Cat and dog recognition is a binary classification problem. Along with the increase of data sets and the improvement of computing power, especially the explosive improvement of GPU computing power, the deep learning using convolutional neural network solves the problem of picture recognition. This project will use tensorflow and keras to build a learning model that is quick and easy to use.
CNN (Convolutional Neural Network) is one of the most powerful deep learning neural networks, especially in image recognition applications. CNN's design prototype originated from LeNet5. It was mainly used for text recognition. Later, people continued to optimize the structure and performance of the network to form the CNN that we use today.
CNN's advantages in image recognition are mainly reflected in:

* Reduce the number of weights in the calculation process

* It can resist slight distortion, deformation and other disturbances in object recognition

* The recognition of an object is not affected by the positional change of the object in the picture

The main components of CNN are convolution layer, pooling layer, fully connected layer. By continuously reducing the dimensions of the data, it can eventually be used to train. Below I will introduce the knowledge of 'Convolution layer', 'Pooling layer', 'Dropout layer' and 'Transfer learning'.

![CNN](https://github.com/pippen6668/capstone/blob/master/images/CNN.png)

(1) Convolution layer

Through the convolution operation, the computationally intensive image recognition problem is continuously reduced, and finally it can be trained by the neural network. Intuitively, the convolutional layer can be thought of as a series of training/learning filters. A convolution filter slides over each pixel to form a new output data(feature map).

![Convolution principle](https://github.com/pippen6668/capstone/blob/master/images/Convolution.png)

(2) Pooling layer

Theoretically, the data after the convolution process can directly use for training, but the amount of calculation is still too large. In order to reduce the amount of calculation and improve the generalization ability of the model, we will pool it.

![Pooling principle](https://github.com/pippen6668/capstone/blob/master/images/Pooling_Simple_max.png)

(3) Dropout layer

During the training of the deep learning network, the neural network unit is temporarily discarded from the network with a certain probability. This way can reduce the interaction between feature detectors and enhance the generalization ability of the model.

![Dropout principle](https://github.com/pippen6668/capstone/blob/master/images/drop_out.png)




### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?




