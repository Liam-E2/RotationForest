from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Use case of superiority over simple Random Forest

def generate_sample():
    label = np.random.choice([0, 1], size=1)
    if label == 0:
        d1 = np.random.uniform(1, 10, 1)
        d2 = d1**2 + np.random.uniform(0, 3, 1)
    else:
        d1 = np.random.uniform(3, 13, 1)
        d2 = d1**2 + np.random.uniform(4, 7, 1)

    noise = np.random.normal(0, 1, 8)

    X = np.concatenate([d1, d2, noise.flatten()], axis=0)
    y = label

    return X, y

def generate_data(n_samples):
    X = []
    Y = []
    for i in range(n_samples):
        x, y = generate_sample()
        X.append(x)
        Y.append(y)

    return np.asarray(X), np.asarray(Y)


Xdata, Ydata = generate_data(3000)

xtr, xte, ytr, yte = train_test_split(Xdata, Ydata, test_size=.3)

Rotate = RotationForest(n_trees=200)
Random = RandomForestClassifier(n_estimators=200)
Linear = LogisticRegression()

Rotate.fit(xtr, ytr)
Random.fit(xtr, ytr)
Linear.fit(xtr, ytr)

preds_rotate = Rotate.predict(xte)
preds_random = Random.predict(xte)
preds_linear = Linear.predict(xte)
obs = yte

df = pd.DataFrame(data=[preds_rotate, preds_random, obs, preds_linear]).T
df['rotate'] = df[2].eq(df[0]).astype(int)
df['random'] = df[2].eq(df[1]).astype(int)
df['linear'] = df[2].eq(df[3]).astype(int)

print(df['rotate'].sum() / len(df))
print(df['random'].sum() / len(df))
print(df['linear'].sum() / len(df))

import matplotlib.pyplot as plt

plt.scatter(xtr[:,0], xtr[:,1], c=ytr.flatten())
plt.title("Train Data")
plt.show()

plt.scatter(xte[:,0], xte[:,1], c=preds_random.flatten())
plt.title("Random Forest Decisions")
plt.show()

plt.scatter(xte[:,0], xte[:,1], c=preds_rotate.flatten())
plt.title("Rotation Forest Decisions")
plt.show()

plt.scatter(xte[:,0], xte[:,1], c=preds_linear.flatten())
plt.title("Logistic Regression Decisions")
plt.show()


