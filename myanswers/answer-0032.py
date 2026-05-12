import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def evaluar_clasificador(X, y, n_neighbors):
    """
    Entrena un modelo KNeighborsClassifier y devuelve el promedio
    de la validación cruzada.
    """
    # 1. Instanciar el modelo con los vecinos indicados
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # 2. Calcular cross_val_score con cv=3 y scoring="accuracy"
    scores = cross_val_score(
        model, 
        X, 
        y, 
        cv=3, 
        scoring="accuracy"
    )
    
    # 3. Retornar el promedio como float
    return float(np.mean(scores))
