# Weather Classification
This notebook trains and tests a neural network using PyTorch to classify images of weather conditions into 4 classes: Cloudy, Rain, Shine, Sunrise.

VGG16 (Simonyan et al., 2014) convolutional features are used for transfer learning, and a feed-forward network is trained to label weather images with over 98% accuracy.

Weather images available at: https://data.mendeley.com/datasets/4drtyfjtfy/1

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/117173628-ddb8c080-adc4-11eb-9e96-945cc2f6ebe4.png" width="900">
</p>

## Data preparation

The Mendeley dataset must be split into 4 folders, one for each class, with each named after the class of images it contains. I.e. all 'cloudy' images must be in WeatherImages/cloudy/.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/117256655-da641a00-ae42-11eb-977d-f983141a96e0.png" height="200">
</p>


## Dataset

The Mendeley dataset contains 1125 images of classes: Cloudy, Rain, Shine and Sunrise. These make up the dataset in the proportions shown below. Augmentation is used to increase the size of the training dataset.
<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/117173631-de515700-adc4-11eb-8453-2ad84b17e1a6.png" height="300"><img src="https://user-images.githubusercontent.com/47857277/117173622-dd202a00-adc4-11eb-94c0-2357e4efe3dd.png" height="300">
</p>

## Training

VGG16 is used for transfer learning, and a simple fully connected network is trained as a classifier on its convolutional features. The complete architecture of the network is shown below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/117173637-dee9ed80-adc4-11eb-901f-e9091d8fa160.png" width="500">
</p>

## Results

The network has a test accuracy of **98.23%** (222/226 test predictions correct). Further analysis of results demonstrates the model's strong recall and precision scores.
<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/117173625-ddb8c080-adc4-11eb-96e8-a4e0ea70a0b5.png" height="200"><img src="https://user-images.githubusercontent.com/47857277/117174841-1016ed80-adc6-11eb-952c-38c258856be8.png" height="200"><img src="https://user-images.githubusercontent.com/47857277/117174843-10af8400-adc6-11eb-8a66-3c19130bf4d2.png" height="200">
</p>


Example labels and predictions:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/117173634-dee9ed80-adc4-11eb-971b-d8cafb9385d6.png" width="900">
</p>

The network has low confidence scores for the two incorrect predictions shown. This further emphasises the model's ability to predict labels accurately.


**References:** *Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. 9 2014. https://arxiv.org/abs/1409.1556* 
