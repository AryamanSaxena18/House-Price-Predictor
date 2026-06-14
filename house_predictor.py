# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Creating the dataframe
df = pd.read_csv('Git Project/data.csv')

# Dataset Overview

print(df.head())
print(df.describe())

# Understanding the shape of the dataset
print(df.shape)

# Checking for missing values and data types
print(df.isnull().sum())
print(df.info())

# No missing values were found in the dataset.

# Exploratory Data Analysis

scatter_matrix(df[['sqft', 'crime_index', 'school_score', 'distance_city_km']],figsize=(10, 12))
df.hist(bins=100)
plt.show()

# Correlation Analysis
corr = df.corr()
print(corr['house_price'].sort_values(ascending=False))

# sqft has the strongest positive correlation with house_price
# age has the strongest negative correlation with house_price

# Feature Selection
x = df.drop('house_price', axis=1)
y = df['house_price']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Feature Scaling
scaled = StandardScaler()
x_train_scaled = scaled.fit_transform(x_train)
x_test_scaled = scaled.transform(x_test)

# Model Training
model = LinearRegression()
model.fit(x_train_scaled, y_train)
prediction = model.predict(x_test_scaled)

# Model Evaluation
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")# R² above 0.9 indicates excellent performance.

# Actual vs Predicted Plot

plt.scatter(y_test, prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

# House Price Prediction

sqft = float(input("Enter the sqft: "))
bedrooms = float(input("Enter the bedrooms: "))
bathrooms = float(input("Enter the bathrooms: "))
age = float(input("Enter the age: "))
garage_spaces = float(input("Enter the garage spaces: "))
crime_index = float(input("Enter the crime index: "))
school_score = float(input("Enter the school score: "))
distance_city_km = float(input("Enter the distance from city: "))

userdata = {
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'garage_spaces': garage_spaces,
    'crime_index': crime_index,
    'school_score': school_score,
    'distance_city_km': distance_city_km
}

newdf = pd.DataFrame([userdata])

newdf_scaled = scaled.transform(newdf)

outcome = model.predict(newdf_scaled)

print(f'The expected price of your house is: {outcome[0]}')