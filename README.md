# Offline Signature Verification System

# Group Member
Chenghong Lin [(chenghonglin2020@u.northwestern.edu)](mailto:chenghonglin2020@u.northwestern.edu)
Xiangguo Liu [(xg.liu@u.northwestern.edu)](mailto:xg.liu@u.northwestern.edu)

# Abstract
## Task
Given a library of handwritten signatures with genuine or fake labels, our task is to use machine learning techniques to tell whether a signature is real or fake by the offline signature image.

## Motivation
There are two reasons why we are interested in this task. Firstly, nowadays the world is in the age of information, our signature can be easily stolen by potential criminals. Then they can make fake signatures. Usually, we just distinguish whether it’s fake or real manually which it’s very time-consuming and its accuracy is not very high. Different people may even give different answer. So, we want to apply the machine learning knowledge to build a handwritten signature classification system to judge whether the signature is fake or not. Secondly, we decide to focus on the offline handwriting verification process. Because Offline verification can be used in information-limited situation and is considered to be the harder case.


## Approach
We adopted three methods (K-Nearest Neighbor, Neural Networks, Support Vector Machine) to solve this problem. At first, pixel values of images are extracted as features to be used in KNN(pixel), however, this accuracy is pretty low, it only gets about 17% accuracy. Next, we tried to use transfer learning to extra features of the images. And VGGNet is utilized to get new representative features of the images. Then these new features are used in KNN(vgg), NN and SVM. 80% dataset is utilized as the training dataset, 10% is validation dataset, and the remain 10% is test dataset. Neural Networks provides the best performance, its accuracy is 94.77%. Use VGG representation is considered to have better performance than pixel representation.


# Results
Put IMAGE HERE!
The results show the model accuracy based on different methods. **Insert TABLE!!!*

And after we trained the network, it can output the accuracy and predict whether this image is real or fake.


# Related Work
## Dataset Cleaning
We obtaint the dataset from ICDAR 2011 Signature Verification Competition. The whole dataset contains the signatures of 67 persons. Each person has 12 real and fake images. At first, our dataset is a bit messy, and we spent some time on cleaning the data and make each images to have the correct label.

## Features Extraction
At first, we just simply extract the pixel value of each image and calculate the pixel difference between each image. However, the result is bad, we only get 17% accuracy on nearest neighbor algorithm. Thanks to Prof. Downey, after talking with him, we found that we can use transfer learning to extract the features of our images, and after that we can build our own classifier on top of that.

The architecture of VGG Network is shown below. It contains 4 convolutional layers, 3 fully connected layers and softmax output layer, the hidden layer uses ReLU activation function. Because VGG Network is a good feature extractor, so the idea in our project is that we uses VGG Network to extract the feature of the raw images, and get the feature vector of these images.

## Model Training 
In this project, we applied three training methods: K Nearest Neighbor, Neural Networks, Support Vector Machine.



### Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/Hclmaster/EECS349-Group-Project/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Hclmaster/EECS349-Group-Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
