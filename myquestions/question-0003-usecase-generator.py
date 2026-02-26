import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

def generar_caso_de_uso_analizar_curva_aprendizaje():
    np.random.seed()

    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        random_state=None
    )

    modelo = RandomForestClassifier(random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        modelo,
        X,
        y,
        cv=5
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    if train_mean[-1] > val_mean[-1] + 0.1:
        diagnostico = "Posible sobreajuste"
    elif train_mean[-1] < 0.6:
        diagnostico = "Posible subajuste"
    else:
        diagnostico = "Buen ajuste"

    input_data = {
        "X": X,
        "y": y
    }

    output_data = {
        "train_mean": train_mean,
        "val_mean": val_mean,
        "diagnostico": diagnostico
    }

    return input_data, output_data
