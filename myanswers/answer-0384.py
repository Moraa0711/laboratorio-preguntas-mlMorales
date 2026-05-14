import numpy as np
from collections import defaultdict

def reporte_especifico_clases(y_true, y_pred):
    """
    Calcula la precisión por clase (recall) basándose en la lógica del generador.
    """
    # Identificar las clases únicas
    clases = np.unique(y_true)
    
    # Matriz de confusión manual
    matriz = defaultdict(lambda: defaultdict(int))

    for real, pred in zip(y_true, y_pred):
        matriz[real][pred] += 1

    reporte = {}

    for clase in clases:
        total_reales = sum(matriz[clase].values())
        verdaderos_positivos = matriz[clase][clase]

        if total_reales == 0:
            val = 0.0
        else:
            val = verdaderos_positivos / total_reales

        # Nombre de llave y redondeo exacto
        reporte[f"Clase_{clase}"] = round(float(val), 3)

    return reporte
