from sklearn import datasets,ensemble,metrics
import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap

X,Y = datasets.make_moons(n_samples=100, shuffle=True, noise=0.12, random_state=None)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
rf = ensemble.RandomForestClassifier()


rf.fit(X_train,Y_train)
y_predict = rf.predict(X_test)

acc = metrics.accuracy_score(Y_test, y_predict, normalize=True, sample_weight=None)

print(acc)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(1,1,1)
ax.set_title("Input data")
# Plot the training points
pdb.set_trace()
ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright, edgecolors="k")
# Plot the testing points
ax.scatter(	X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
plt.show()
