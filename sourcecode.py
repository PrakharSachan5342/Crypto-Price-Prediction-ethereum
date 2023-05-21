!pip install pandas scikit-learn matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the cryptocurrency price dataset
df = pd.read_csv('Aaveconfig.csv')  # Replace with your dataset file

# Preprocess the data
# (Data cleaning, feature selection, normalization, etc.)

# Split the dataset into training and testing sets
X = df[['High', 'Low', 'Open' , 'Close']]  # Replace with relevant features
y = df['Volume']  # Replace with the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)

# Visualize the predictions
plt.plot(y_test.index, y_test, label='Actual Prices')
plt.plot(y_test.index, y_pred, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
