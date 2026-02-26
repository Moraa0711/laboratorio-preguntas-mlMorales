import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generar_caso_de_uso_seleccionar_features_por_importancia():
    np.random.seed()

    n_samples = 200
    n_features = 8

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)

    modelo = RandomForestClassifier()
    modelo.fit(X, y)

    importancias = modelo.feature_importances_

    top_k = np.random.randint(2, 6)

    indices_ordenados = np.argsort(importancias)[::-1]
    top_indices = indices_ordenados[:top_k]

    X_reducido = X[:, top_indices]

    input_data = {
        "X": X,
        "y": y,
        "top_k": top_k
    }

    output_data = {
        "indices_seleccionados": top_indices,
        "X_reducido": X_reducido
    }

    return input_data, output_data
