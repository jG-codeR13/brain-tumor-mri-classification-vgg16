# brain-tumor-mri-classification-vgg16
1. Project Overview and Objectives
The main purpose of this project was to build a CNN model that would classify if subject has a tumor or not base on MRI scan. I used the VGG-16, architecture and weights to train the model for this binary problem. I used accuracy as a metric to justify the model performance which can be defined as:


Final results look as follows:

Set	Accuracy
Validation Set*	~92%
Test Set*	~92%

* Note: there might be some misunderstanding in terms of set names so I want to describe what do I mean by test and validation set:

validation set - is the set used during the model training to adjust the hyperparameters.
test set - is the small set that I don't touch for the whole training process at all. It's been used for final model performance evaluation.
1.1. Data Set Description
The image data that was used for this problem is Brain MRI Images for Brain Tumor Detection. It conists of MRI scans of two classes:

NO - no tumor, encoded as 0
YES - tumor, encoded as 1

What is Brain Tumor?¶
A brain tumor occurs when abnormal cells form within the brain. There are two main types of tumors: cancerous (malignant) tumors and benign tumors. Cancerous tumors can be divided into primary tumors, which start within the brain, and secondary tumors, which have spread from elsewhere, known as brain metastasis tumors


Transfer Learning

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.
Transfer Learning differs from traditional Machine Learning in that it is the use of pre-trained models that have been used for another task to jump start the development process on a new task or problem.

CONCLUSION
This project was a combination of CNN model classification problem (to predict wheter the subject has brain tumor or not) & Computer Vision problem (to automate the process of brain cropping from MRI scans). The final accuracy is much higher than 50% baseline (random guess). However, it could be increased by larger number of train images or through model hyperparameters tuning.




