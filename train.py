import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv("student-por.csv")

# Use only 6 selected features
selected_features = ['age', 'studytime', 'failures', 'absences', 'G1', 'G2']
X = df[selected_features]
y = df['G3']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained with 6 features and saved as model.pkl")
