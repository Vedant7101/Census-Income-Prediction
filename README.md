## Census Income Prediction

### Introduction

- US Adult Census data relating income to social factors such as Age, Education, race etc.
The Us Adult income dataset was extracted by Barry Becker from the 1994 US Census
Database. The data set consists of anonymous information such as occupation, age, native
country, race, capital gain, capital loss, education, work class and more. Each row is labelled
as either having a salary greater than ">50K" or "<=50K".

- This Data set is in dataset.csv The goal here is to train a binary classifier on the training
dataset to predict the column income_bracket which has two possible values ">50K" and
"<=50K" and evaluate the accuracy of the classifier with the test dataset.

### Dataset Description

This dataset contains income of people in United States. Each row represents particular
number of people and the number is described in fnlwgt (Final Weight).

**Features**

**1. age** - Age of group <br>
**2. education.num** - Number of years spent on eduction <br>
**3. workclass** - Type of job <br>
**4. occupation** - Occupation of group <br>
**5. Workclass** - Whether person is government employee or private. <br>
**6. Sex** - Sex of group <br>
**7. capital.gain** <br>
**8. capital.loss** <br>
**9. hours.per.week** - Number of hours spent on work <br>
**10. income** - It is categorical features. (1 for income > 50K and 0 for income <= 50K) <br>

### Prediction Accuracy

![alt text](/images/image2.png)

### Dashboard

![alt text](/images/image1.png)

### Prediction Portal 

![alt text](/images/image3.png)

### Conclusion

1. Decision Tree Classifier and Random Forest Classifier works best on training data. <br>
2. Gradient Boosting Classifier and Random Forest Classifier works best on testing data. <br>
3. We have performed classification using Random Forest Classifier. <br>
4. We have got 87% accuracy on training data and 84% accuracy on testing data. <br>
5. Age, Qualification and Occupation are most important factor in this classification. <br>
6. If age and qualification is high of person, then he/she will more likely to have income > 50K. <br>
