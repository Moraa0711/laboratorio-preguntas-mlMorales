import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generar_caso_de_uso_seleccionar_features_por_importancia():
    np.random.seed()

    n_samples = 100
    n_features = 10

    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)

    modelo = RandomForestClassifier()
    modelo.fit(X, y)

    importancias = modelo.feature_importances_
    top_k = 5
    indices = np.argsort(importancias)[-top_k:][::-1]
    X_filtrado = X[:, indices]

    return {
        "input": {
            "X": X,
            "y": y,
            "modelo_tipo": "random_forest",
            "top_k": top_k
        },
        "output": (list(indices), X_filtrado)
    }
