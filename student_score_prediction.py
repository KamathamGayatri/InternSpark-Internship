import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [15, 20, 35, 40, 50, 60, 65, 75, 85, 95]
}
df = pd.DataFrame(data)
print(df)
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Student Score Prediction")
plt.show()
X = df[['Hours']]
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)