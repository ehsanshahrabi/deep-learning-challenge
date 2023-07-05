# Report on Neural Network Model for Alphabet Soup

## Overview of the Analysis

The purpose of this analysis is to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. 
The data set provided includes over 34,000 organizations that have received funding from Alphabet Soup over the years.

## Results

### Data Preprocessing

#### 1. What variable(s) are the target(s) for your model?

The target variable for the model is 'IS_SUCCESSFUL', which indicates whether or not the charity application was successful.

#### 2. What variable(s) are the features for your model?
   
The feature variables for the model include 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', and 'ASK_AMT'.

#### 3. What variable(s) should be removed from the input data because they are neither targets nor features?

Variables 'EIN' and 'NAME' were removed from the input data as they are neither targets nor features. They are simply identifiers and do not provide meaningful information for the model.

### Compiling, Training, and Evaluating the Model

#### 4. How many neurons, layers, and activation functions did you select for your neural network model, and why?

Original Model:

The original model contained two layers with 80 and 30 neurons respectively. The activation function used in these layers was 'ReLU'. 'ReLU' was selected as the activation function due to its efficient computation and because it helps mitigate the vanishing gradient problem, which can hinder the learning process in deep neural networks. The output layer used 'sigmoid' as the activation function because this is a binary classification problem, and 'sigmoid' is suitable for such problems as it outputs probabilities between 0 and 1.

Optimized Models:

Model 1: This model retained the original structure of two layers but changed the activation functions. The activation function of the first hidden layer was switched to 'tanh', and the second hidden layer used 'LeakyReLU'. 'tanh' was used because it can sometimes perform better in practice compared to 'ReLU' by mapping negative inputs to negative outputs. 'LeakyReLU' was selected to avoid dead neurons, which can occur with 'ReLU' if negative inputs result in a 0 output.

Model 2: This model had a more complex architecture with four hidden layers containing 100, 50, 25, and 10 neurons respectively. The additional layers were added in an attempt to capture more complex patterns in the data. The 'ReLU' activation function was used for all the hidden layers due to its efficient computation and its ability to handle the vanishing gradient problem.

Model 3: This model had two layers with 80 and 30 neurons respectively, just like the original model. However, it introduced dropout layers for regularization to prevent overfitting. The dropout layers randomly set a fraction of input units to 0 at each update during training time, which helps prevent overfitting. The 'ReLU' activation function was used for the hidden layers.

Model 4: The structure of this model was identical to the original model, with two layers containing 80 and 30 neurons. However, the model employed 'SGD' as the optimizer with a learning rate of 0.01, unlike the original model which used 'adam'. Stochastic Gradient Descent (SGD) was used to see if the slower, more robust nature of this optimizer could lead to better performance.

#### 5. Were you able to achieve the target model performance?

No, the model did not achieve the target performance. The original model achieved an accuracy of ~72.7%.

6. What steps did you take in your attempts to increase model performance?

Several steps were taken to increase the model performance:

Model 1: The activation functions were altered, changing the first hidden layer to "tanh" and the second to "LeakyReLU".

Model 2: More layers and neurons were added to the model. The new model included four hidden layers with 100, 50, 25, and 10 neurons respectively.

## Summary

In my analysis of the Alphabet Soup Charity data, I implemented a deep learning model that achieved a satisfactory accuracy of about 72.5% - 72.7%. Despite several optimization attempts, I could not significantly improve the model's performance.

This led me to consider other machine learning methods, such as Random Forest or Gradient Boosting, for this binary classification task. Alternatively, refining the feature selection process or applying different dimensionality reduction techniques could help optimize the model.

If we persist with deep learning, we could explore extensive hyperparameter tuning, various learning rate schedules, or more complex architectures like convolutional or recurrent neural networks. The key learning point is that model selection should always align with the data, task, and available resources. There isn't a one-size-fits-all approach. Trial and error is an inherent part of finding the best solution.

Model 3: Dropout layers were added to prevent overfitting.

Model 4: The optimizer was changed from 'adam' to 'SGD' with a learning rate of 0.01.

