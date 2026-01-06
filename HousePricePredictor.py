import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    'Size_sqft': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 5],
    'Age_years': [10, 5, 3, 2, 1],
    'Price': [200000, 250000, 300000, 400000, 500000]
}

df = pd.DataFrame(data)

# Step 2: Split features and target
X = df[['Size_sqft', 'Bedrooms', 'Age_years']]
y = df['Price']

# Step 3: Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict price for a new house
test_input = [[2200, 3, 4]]
predicted_price = model.predict(test_input)
print("Predicted House Price:", predicted_price[0])
