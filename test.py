from sklearn.tree import DecisionTreeClassifier

# Sample data: [Age, Income]
# Age in years, Income in thousands
X = [
    [25, 50],   # young, low income
    [30, 60],   # young, medium income
    [45, 80],   # middle-aged, high income
    [35, 40],   # middle-aged, low income
    [50, 90],   # older, high income
    [23, 30],   # young, very low income
]

# Labels: 1 = will buy, 0 = won't buy
y = [0, 1, 1, 0, 1, 0]

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for a new person: 40 years old with income 70k
new_data = [[40, 70]]
prediction = model.predict(new_data)

# Output result
print("Prediction:", "Will Buy" if prediction[0] == 1 else "Won't Buy")
