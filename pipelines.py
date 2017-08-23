#!/cygdrive/c/Users/awest/AppData/Local/Continuum/Anaconda3/python.exe
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import mglearn

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def main():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target,
        random_state=0
    )

    param_grid = {
        'svm__C': [.001, .01, .1, 1, 10, 100],
        'svm__gamma': [.001, .01, .1, 1, 10, 100]
    }

    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svm', SVC())
    ])

    grid = GridSearchCV(
        pipe, param_grid=param_grid,
        cv=5
    )

    grid.fit(X_train, y_train)

    print('Best cross-val accuracy: {:.2f}'.format(
        grid.best_score_)
    )
    print('Test score: {:.2f}'.format(
        grid.score(
            X_test, y_test
        )
    ))
    print('Best params: {}'.format(
        grid.best_params_
    ))


if __name__ == '__main__':
    main()
