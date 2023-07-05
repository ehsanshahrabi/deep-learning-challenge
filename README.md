# Report on Neural Network Model for Alphabet Soup

## Overview of the Analysis

This analysis aims to help Alphabet Soup, a non-profit organization, predict which projects will succeed if funded. By leveraging historical data and deep learning techniques, the goal is to create a model that can efficiently classify applications based on their likelihood of success. This would aid Alphabet Soup in making impactful and effective funding decisions.

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

Optimization  1: This model retained the original structure of two layers but changed the activation functions. The activation function of the first hidden layer was switched to 'tanh', and the second hidden layer used 'LeakyReLU'. 'tanh' was used because it can sometimes perform better in practice compared to 'ReLU' by mapping negative inputs to negative outputs. 'LeakyReLU' was selected to avoid dead neurons, which can occur with 'ReLU' if negative inputs result in a 0 output.

Optimization  2: This model had a more complex architecture with four hidden layers containing 100, 50, 25, and 10 neurons respectively. The additional layers were added in an attempt to capture more complex patterns in the data. The 'ReLU' activation function was used for all the hidden layers due to its efficient computation and its ability to handle the vanishing gradient problem.

Optimization  3: This model had two layers with 80 and 30 neurons respectively, just like the original model. However, it introduced dropout layers for regularization to prevent overfitting. The dropout layers randomly set a fraction of input units to 0 at each update during training time, which helps prevent overfitting. The 'ReLU' activation function was used for the hidden layers.

Optimization  4: The structure of this model was identical to the original model, with two layers containing 80 and 30 neurons. However, the model employed 'SGD' as the optimizer with a learning rate of 0.01, unlike the original model which used 'adam'. Stochastic Gradient Descent (SGD) was used to see if the slower, more robust nature of this optimizer could lead to better performance.

#### 5. Were you able to achieve the target model performance?

No, the model did not achieve the target performance. The original model achieved an accuracy of ~72.7%.

#### 6. What steps did you take in your attempts to increase model performance?

Several steps were taken to increase the model performance:

Optimization  1: The activation functions were altered, changing the first hidden layer to "tanh" and the second to "LeakyReLU".

Optimization  2: More layers and neurons were added to the model. The new model included four hidden layers with 100, 50, 25, and 10 neurons respectively.

Optimization  3: Dropout layers were added to prevent overfitting.

Optimization  4: The optimizer was changed from 'adam' to 'SGD' with a learning rate of 0.01.

## Summary

The deep learning model designed for this task did not achieve the targeted accuracy of 75%, reaching approximately 72.7% at its best. The model was subjected to multiple optimization attempts, which included changing the activation function, increasing the number of layers and neurons, introducing dropout for regularization, and changing the optimizer.

This led me to consider other machine learning methods, such as Random Forest or Gradient Boosting, for this binary classification task. Alternatively, refining the feature selection process or applying different dimensionality reduction techniques could help optimize the model.

If we persist with deep learning, we could explore extensive hyperparameter tuning, various learning rate schedules, or more complex architectures like convolutional or recurrent neural networks. The key learning point is that model selection should always align with the data, task, and available resources. There isn't a one-size-fits-all approach. Trial and error is an inherent part of finding the best solution.

