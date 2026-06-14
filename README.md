# House Price Prediction Using Linear Regression

## Project Overview

This project aims to predict house prices using a Linear Regression model. The dataset contains various numerical features related to a house, such as square footage, number of bedrooms, number of bathrooms, age of the property, garage capacity, crime index, school quality score, and distance from the city center.

The objective is to build a machine learning model capable of estimating the market price of a house based on these characteristics.

---

## Dataset Description

The dataset consists of the following features:

| Feature          | Description                                  |
| ---------------- | -------------------------------------------- |
| sqft             | Total area of the house in square feet       |
| bedrooms         | Number of bedrooms                           |
| bathrooms        | Number of bathrooms                          |
| age              | Age of the property                          |
| garage_spaces    | Number of garage spaces                      |
| crime_index      | Crime index of the locality                  |
| school_score     | Quality score of nearby schools              |
| distance_city_km | Distance from the city center in kilometers  |
| house_price      | Target variable representing the house price |

---

## Exploratory Data Analysis

Several exploratory data analysis techniques were performed to better understand the dataset:

* Dataset structure and summary statistics were examined.
* Missing values were checked and none were found.
* Histograms were generated to study feature distributions.
* Scatter matrix visualizations were used to analyze relationships between important variables.
* Correlation analysis was conducted to identify the features most strongly related to house prices.

### Key Findings

* **sqft** showed the strongest positive correlation with house price.
* **age** showed the strongest negative correlation with house price.
* Larger houses generally tend to have higher prices.
* Older houses generally tend to have lower prices.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Separation of features and target variable.
2. Train-test split using an 80-20 ratio.
3. Feature standardization using StandardScaler.

Feature scaling was applied only to the input variables while the target variable was left unchanged.

---

## Model Selection

A Linear Regression model was selected for this project because:

* The target variable is continuous.
* The dataset contains numerical features.
* Linear Regression provides interpretable results.
* It serves as an excellent baseline model for regression tasks.

---

## Model Training

The model was trained using the training dataset after feature scaling.

The trained model learned the relationship between the input features and the target variable (house price).

---

## Model Evaluation

The following evaluation metrics were used:

### Mean Absolute Error (MAE)

31,908.61

This indicates that, on average, the model's predictions differ from the actual house prices by approximately 31,909 units.

### Mean Squared Error (MSE)

1,605,346,850.58

This metric penalizes larger prediction errors more heavily.

### R² Score

0.973

The model explains approximately 97.3% of the variation in house prices, indicating excellent predictive performance.

---

## Prediction System

The project includes an interactive prediction system that allows users to enter house details such as:

* Square footage
* Bedrooms
* Bathrooms
* Property age
* Garage spaces
* Crime index
* School score
* Distance from city center

The trained model then predicts the expected house price based on the provided inputs.

---

## Conclusion

A Linear Regression model was successfully developed to predict house prices using multiple housing-related features.

The model achieved an R² score of 0.973, demonstrating strong predictive capability on the dataset. Correlation analysis revealed that square footage has the strongest positive influence on house prices, while property age has the strongest negative influence.

This project demonstrates the complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, and deployment of a simple prediction system.

---

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-Learn
* Linear Regression
* StandardScaler
