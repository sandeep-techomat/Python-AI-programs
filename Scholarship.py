# Predict if a student gets a scholarship based on attendance, marks, and extracurricular activities

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Step 1: Create dataset
data = {
    'Attendance_Percentage': [60, 85, 90, 75, 55, 95, 70],
    'Average_Marks': [50, 70, 80, 65, 40, 85, 60],
    'Extra_Curricular': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Scholarship': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Step 2: Encode categorical column (Extra_Curricular)
encoder = LabelEncoder()
df['Extra_Curricular'] = encoder.fit_transform(df['Extra_Curricular'])
# No = 0, Yes = 1

# Step 3: Split features (X) and target (y)
X = df[['Attendance_Percentage', 'Average_Marks', 'Extra_Curricular']]
y = df['Scholarship']

# Step 4: Train Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=3)  # keep tree small & simple
model.fit(X, y)

# Step 5: Test prediction 1
test_input = [[80, 75, encoder.transform(['Yes'])[0]]]  # Example student
prediction = model.predict(test_input)
print("Prediction for test input:", prediction[0])

# Step 6: Test prediction 2
test_input = [[20, 25, encoder.transform(['Yes'])[0]]]  # Example student
prediction = model.predict(test_input)
print("Prediction for test input:", prediction[0])
