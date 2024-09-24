import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from my import model, X_train, y_test, X_test

def matrix():
    test = model.predict(X_test)
    cm = confusion_matrix(y_test,test)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Остались', 'Ушли'], yticklabels=['Остались', 'Ушли'])
    plt.title("Матрица ошибок")
    plt.show()

def w():
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,6))
    plt.title("Важность признаков")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

print(matrix())
print(w())


