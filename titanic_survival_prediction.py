import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
titanic = fetch_openml('titanic', version=1, as_frame=True)

df = titanic.frame
df = df[['pclass', 'sex', 'age', 'fare', 'survived']]
df = df.dropna()
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
X = df[['pclass', 'sex', 'age', 'fare']]
y = df['survived']
plt.scatter(df['age'], df['fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Titanic Survival Prediction")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))