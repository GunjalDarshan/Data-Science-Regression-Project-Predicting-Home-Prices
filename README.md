# Data-Science-Regression-Project-Predicting-Home-Prices
Using Pandas, Numpy, matplotlib, GridSearchCV Designed a model which predicts the price of the house, with 13000+ locations you can ﬁnd an accurate price of your dream home. using GridSearchCV from sklearn ﬁnding the optimal value for the hyperparameter of our model gives the best score that helps to ﬁnd accurate price of house.

 Introduction
This project aims to build a regression model to predict home prices based on various features. The dataset used for this project is a real estate dataset that contains information about different properties such as the location, size, number of bedrooms, and price.
This is a data science regression project that aims to predict home prices based on various features. The project utilizes machine learning techniques to build a regression model that can accurately estimate the price of a house given its attributes.

### Project Steps

#### 1. Importing Libraries and Loading the Dataset
The required libraries such as `numpy`, `pandas`, and `matplotlib` are imported. The dataset is loaded using the `read_csv` function from pandas.

#### 2. Exploratory Data Analysis and Data Cleaning
The dataset is explored by examining its shape, columns, and unique values. Irrelevant features that are not required for building the model are dropped. Missing values in the dataset are handled by removing the rows with missing values.

#### 3. Feature Engineering
New features are created based on existing features to extract useful information. For example, the number of bedrooms (bhk) is extracted from the 'size' feature. The 'total_sqft' feature is converted into a numeric value by handling range values and other non-numeric entries.

#### 4. Outlier Removal
Outliers are identified and removed from the dataset to improve the accuracy of the model. The outliers are identified based on domain knowledge and statistical analysis.

#### 5. Data Transformation
Categorical variables are converted into numerical variables using one-hot encoding. This step is necessary as machine learning models generally require numerical input.

#### 6. Model Building and Evaluation
The dataset is divided into training and testing sets. A linear regression model is built using the training data and evaluated using the testing data. The model's performance is measured using the R-squared score.

#### 7. Cross-Validation and Hyperparameter Tuning
Cross-validation is performed to assess the model's stability and generalization capability. GridSearchCV is used to tune the hyperparameters of different regression models and identify the best-performing model.

#### 8. Prediction
A function is created to predict the home prices based on the provided input features. The function takes input parameters such as location, square footage, number of bathrooms, and number of bedrooms.

### Conclusion
This regression project demonstrates the process of building a model to predict home prices. By analyzing the dataset, performing feature engineering, handling outliers, and selecting the best model, we can accurately predict home prices based on the given features.





## Project Setup
To get started, we need to import the necessary libraries and load the dataset. The code snippet below demonstrates this process:

 ![image](https://github.com/GunjalDarshan/Data-Science-Regression-Project-Predicting-Home-Prices/assets/126502930/cf16c8f4-9097-4b26-9447-9b57403e1c0c)


In this code, we import the required libraries, including NumPy, Pandas, and Matplotlib. We also load the dataset using the `read_csv` function from Pandas and display the first few rows of the dataframe.
## Data Cleaning and Feature Engineering
Before building the regression model, we need to clean the data and perform feature engineering to prepare the dataset. The code snippet below demonstrates the steps involved:
 
![image](https://github.com/GunjalDarshan/Data-Science-Regression-Project-Predicting-Home-Prices/assets/126502930/8f583710-1997-4083-92a1-21e752fc38b6)
![image](https://github.com/GunjalDarshan/Data-Science-Regression-Project-Predicting-Home-Prices/assets/126502930/0ded8797-7d7f-4dfc-9a89-cbc8bba46dde)


 
In this code, we perform various data cleaning and feature engineering tasks. We drop unnecessary columns, handle missing values, convert the total square footage to a numeric value, and calculate the price per square foot. We also preprocess categorical variables using one-hot encoding and split the dataset into input features (X) and the target variable (y).

## Model Building and Evaluation
Once the data is prepared, we can proceed to build the regression model. In this project, we use linear regression as our chosen algorithm. The code snippet below demonstrates how to train and evaluate the model:
![image](https://github.com/GunjalDarshan/Data-Science-Regression-Project-Predicting-Home-Prices/assets/126502930/459e6808-de03-417d-aa05-8fd965ad4278)

 ```
In this code, we split the dataset into training and testing sets using the `train_test_split` function from scikit-learn
