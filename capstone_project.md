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

**(1) Convolution layer**

Through the convolution operation, the computationally intensive image recognition problem is continuously reduced, and finally it can be trained by the neural network. Intuitively, the convolutional layer can be thought of as a series of training/learning filters. A convolution filter slides over each pixel to form a new output data(feature map).

![Convolution principle](https://github.com/pippen6668/capstone/blob/master/images/Convolution.png)

**(2) Pooling layer**

Theoretically, the data after the convolution process can directly use for training, but the amount of calculation is still too large. In order to reduce the amount of calculation and improve the generalization ability of the model, we will pool it.

![Pooling principle](https://github.com/pippen6668/capstone/blob/master/images/Pooling_Simple_max.png)

**(3) Dropout layer**

During the training of the deep learning network, the neural network unit is temporarily discarded from the network with a certain probability to prevent over-fitting problems. This way can reduce the interaction between feature detectors and enhance the generalization ability of the model.

![Dropout principle](https://github.com/pippen6668/capstone/blob/master/images/drop_out.png)

**(4) Transfer learning**

As the name suggests, it is to migrate the learned model parameters to the new model to help the new model training. Considering that most of the data or tasks are related, we can use the migration learning to share the model parameters (also known as the knowledge learned by the model) in a certain way to the new model to speed up and optimize. The learning efficiency of the model does not have to learn from zero like most networks. For example, if you have already played Chinese chess, you can learn chess in analogy. If you have learned English, you can learn French in analogy. Everything in the world has commonalities. How to reasonably find similarities between them and use this bridge to help learn new knowledge is the core issue of migration learning.

![Transfer learning](https://github.com/pippen6668/capstone/blob/master/images/transfer%20learning.png)


### Benchmark

In this project, I decided to adopt the classic ResNet model. Since its introduction in 2015, ResNet has won the first place in the ImageNet competition classification task because it is “simple and practical”, and many methods are based on ResNet50 or ResNet101. On the basis of the completion, detection, segmentation, identification and other fields have a wide range of applications.

On kaggle, there are total of 1,314 teams participated in the competition. In the Chinese version of the capstone project, there is a threshold for students to pass. The minimum requirement is reaching the top 10% (1314*0.1 = 131) of the kaggle Public Leaderboard. That is the logloss on the Public Leaderboard is below 0.06127.

![](https://github.com/pippen6668/capstone/blob/master/images/leaderboard.png)


## III. Methodology

### Data Preprocessing

* Split the training set into dog/ and cat/

![](https://github.com/pippen6668/capstone/blob/master/images/split%20folder.png)

* Use ImageDataGenerator to unify the image size into (224,224) and cat label as 0, dog label as 1


### Implementation

1. Export bottleneck feature

Use Keras' pre-training model ResNet50 to extract features and save for subsequent training and testing. Use GlobalAveragePooling2D to directly average each activation map of the convolutional layer output, otherwise the output file will be very large and easy to overfit. Then we use the model.predict_generator function to derive the bottleneck feature.

![](https://github.com/pippen6668/capstone/blob/master/images/extract%20bottleneck%20feature.png)

2. Load bottleneck feature

In addition, it needs to shuffle the data, otherwise we will have problems after setting validation_split. This is because there is a trap here, the program executes the validation_split and then shuffle, so this happens: If your training set is ordered, for example, the positive sample is in the front negative sample, and then set With validation_split, then your validation set will most likely be a negative sample. Similarly, this thing will not be reported any errors, because Keras can't know if your data has been shuffled. If you are not shuffled, it is best to shuffle it manually.

![](https://github.com/pippen6668/capstone/blob/master/images/load%20bottleneck%20feature.png)

3. Construct and train the model

A part of the training set is taken out as a validation set in a certain proportion. Here we set to training set 8: Validation set 2 split ratio(0.2)

![](https://github.com/pippen6668/capstone/blob/master/images/train.png)

4. Remove abnormal images

Use step3 well-trained model to predict the training set and try to remove  pictures with a large difference in the probability of owning the label ( eg. image is cat, but the result probability is 0.92, which classify as a dog). But even with such a strict setting, there are still many pictures that have been deleted by mistake.

![](https://github.com/pippen6668/capstone/blob/master/images/remove%20abnormal%20image.png)

5. Re-train the model

![](https://github.com/pippen6668/capstone/blob/master/images/re%20train.png)

6. Predict for testing set and upload csv to Kaggle

Here we have a clip operation on the score of the result, and each prediction is limited to [0.005, 0.995] intervals. The official evaluation criterion of kaggle is LogLoss. For the prediction of the correct sample, the difference between 0.995 and 1 is very small, but for the sample that predicted the error, the gap between 0 and 0.005 is very large.

![](https://github.com/pippen6668/capstone/blob/master/images/predict.png)
![](https://github.com/pippen6668/capstone/blob/master/images/score.png)

The score of the submission result is 0.08762, bigger than 0.06127.

### Refinement

After several times changes for parameter( such as optimizer, dropout ratio .etc), the score still can't reach top 10%. Then, I found a method to "merge model" when I was looking for another solution, and the result looks very powerful.

The concept is to integrate the bottleneck features of different models (the author choose ResNet50, Xception, Inception V3) into a new bottleneck.

Using the merge model, I found that even if I relaxed image filter conditions, the situation in which the picture was misjudged improved a lot.

![](https://github.com/pippen6668/capstone/blob/master/images/remove%20abnormal%20image2.png)
![](https://github.com/pippen6668/capstone/blob/master/images/cd.png)
![](https://github.com/pippen6668/capstone/blob/master/images/dd.png)
![](https://github.com/pippen6668/capstone/blob/master/images/merge_train.png)


## IV. Results


### Model Evaluation and Validation

The final model is shown below.

![](https://github.com/pippen6668/capstone/blob/master/images/merge.png)

And the training accuracy increased to 0.998, training loss reduced to 0.03, validation accuracy increased to 0.995 and validation loss reduced to 0.11.

![](https://github.com/pippen6668/capstone/blob/master/images/accuracy%20and%20loss.png)

Finally, the score is improve to 0.04391(27/1314).
![](https://github.com/pippen6668/capstone/blob/master/images/score2.png)
![](https://github.com/pippen6668/capstone/blob/master/images/27.png)

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
