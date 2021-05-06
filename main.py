import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")
X = df[['Glucose', 'Insulin']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
log = LogisticRegression(random_state=0)
model = log.fit(X, y)
score = model.score(X_test, y_test)
score = round(score, 2)
score *= 100
values = [100, 0]
val = model.predict([values])
print(val)
