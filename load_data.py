import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("titanic.csv")
data = data[["Sex", "Age", "Survived"]].dropna()
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})

X = data[["Sex", "Age"]]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Trained model saved to 'titanic_model.pkl'")