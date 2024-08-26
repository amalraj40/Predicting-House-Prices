import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   

from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("housing_data.csv")   


# Preprocess the data (e.g., handle missing values, outliers)
# ...

# Feature engineering (e.g., create new features)
# ...

# Split the data
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create   
 and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)   

print("Mean Squared Error:",   
 mse)