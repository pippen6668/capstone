# Machine Learning Engineer Nanodegree
## Capstone Proposal
yfchiena  
January , 2019

## Proposal

### Domain Background

(Dogs vs. Cats) The project is a competition topic for kaggle in 2013. The goal is to train a model to distinguish whether it is a cat or a dog from a given picture. This is a problem in the field of computer vision, and also a binary classifaction problem.

Deep learning is one of the most important breakthroughs in the field of artificial intelligence in the past decade. It has achieved great success in speech recognition, natural language processing, computer vision, image and video analysis, multimedia, and many other fields.

The most influential breakthrough in deep learning in computer vision occurred in 2012, and Hinton's research team won the ImageNet Image Classification Competition with deep learning. Since then, the annual ImageNet image classification competition has won the championship of the neural network.


| Year | Model | Top 5 error rate |
| ------ | ------ | ------ |
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

### Datasets and Inputs

The datasets comes from Kaggle, there are 25,000 pictures in the training data, half each for cats and dogs, and each picture has a category label. The test data has a total of 12,500 images. And the training:validation ratio is 4:1.

In all the above pictures, all the color pictures contain RGB three-channel information, but the picture quality are different, and the images size are inconsistent. There is no way to directly input it into the neural network, so 'resize' is needed.

### Solution Statement

Convolutional Neural Network (CNN) is one of the most representative network structures in deep learning technology. It has achieved great success in the field of image processing. Many successful models are on the international standard ImageNet dataset. It is based on CNN. One of the advantages of CNN over conventional image processing algorithms is that it avoids complex pre-processing of images (especially feature extraction) and can directly input the original image. The CNN network performs multiple convolutional layers and pooling layers processing on the image, gives two nodes in the output layer to obtain the respective probabilities of the two categories.

### Benchmark Model
_(approximately 1-2 paragraphs)_

Use keras-based network models to complete the project. On kaggle, there are total of 1,314 teams participated in the competition. So we can compare our results with all teams to get a relative comparison.

In the Chinese version of the capstone project, there is a threshold for students to pass. The minimum requirement is reaching the top 10% (the score of 131 is 0.06127) of the kaggle Public Leaderboard. We can use this condition as a target to get the score less than 0.06127.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
