# Machine-learning-clasification

Project Description

This project focuses on comparing the classification of red and blue colors in images. The aim is to develop machine learning models using algorithms such as Naive Bayes and K-Nearest Neighbors (KNN) to compare the results obtained from the applied algorithms.

Preprocessing Stage:

The dataset contains 430 images for the red color and 429 for the blue color, with 10 images for testing.
During the preprocessing stage, the images are resized to a uniform size to facilitate processing, converted to grayscale to reduce complexity, and standardized to bring the data to the same scale.

Presentation of the applicability of the proposed algorithms for the chosen problem:

Both Naive Bayes and KNN are used on the same dataset to determine their performance, allowing for a relevant comparison of the results.
The Naive Bayes algorithm is used with Gaussian distribution, which is used in various domains such as text recognition, medical diagnosis, and image classification.
Application of algorithms:

Description of the training and validation process:

Training is done on the aforementioned datasets after they have been loaded, converted to grayscale, and standardized. The algorithms are applied to the same dataset in turn.


Explanation of the testing process:

The testing process involves using the trained models to make predictions on a set of images that were not used in the training phase. The purpose of this process is to evaluate the models' ability to generalize and make correct classifications for unknown images.
The accuracy of the algorithms is evaluated, and the confusion matrix is displayed.
Testing results:

Personal conclusions and relevant observations on the approach:

The Naive Bayes model correctly recognized 8 out of the 10 test images, while KNN recognized 7. Both models exhibit fairly good performance to be used in a concrete application, with an advantage in terms of accuracy for the Naive Bayes model.
