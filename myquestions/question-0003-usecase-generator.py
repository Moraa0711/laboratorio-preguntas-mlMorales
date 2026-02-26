import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

def generar_caso_de_uso_analizar_curva_aprendizaje():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        random_state=None
    )

    modelo = LogisticRegression(max_iter=1000)

    train_sizes, train_scores, val_scores = learning_curve(
        modelo, X, y, cv=5
    )

    train_mean = train_scores.mean()
    val_mean = val_scores.mean()

    if train_mean > val_mean + 0.1:
        diagnostico = "overfitting"
    elif train_mean < 0.6 and val_mean < 0.6:
        diagnostico = "underfitting"
    else:
        diagnostico = "buen_ajuste"

    return {
        "input": {
            "X": X,
            "y": y,
            "modelo": modelo,
            "cv": 5
        },
        "output": {
            "train_mean": train_mean,
            "val_mean": val_mean,
            "diagnostico": diagnostico
        }
    }
