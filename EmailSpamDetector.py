import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Step 1: Create dataset
data = {
    'Word_Count': [100, 50, 300, 80, 200],
    'Has_Links': ['Yes', 'Yes', 'No', 'No', 'Yes'],
    'Has_Special_Chars': ['No', 'Yes', 'No', 'Yes', 'Yes'],
    'Spam': ['No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Step 2: Encode categorical features
encoder = LabelEncoder()
df['Has_Links'] = encoder.fit_transform(df['Has_Links'])  # Yes=1, No=0
df['Has_Special_Chars'] = encoder.fit_transform(df['Has_Special_Chars'])
df['Spam'] = encoder.fit_transform(df['Spam'])  # Yes=1, No=0

# Step 3: Split features and target
X = df[['Word_Count', 'Has_Links', 'Has_Special_Chars']]
y = df['Spam']

# Step 4: Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 5: Predict for a new email
test_input = [[120, 1, 0]]  # 120 words, has links, no special chars
prediction = model.predict(test_input)
print("Spam Prediction:", "Yes" if prediction[0] == 1 else "No")
