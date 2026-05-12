import numpy as np
from collections import defaultdict

def reporte_especifico_clases(y_true, y_pred):
    """
    Calcula la precisión por clase (recall) basándose en la lógica del generador.
    """
    # Identificar las clases únicas presentes en los datos reales
    clases = np.unique(y_true)
    
    # Usar un diccionario para contar la matriz de confusión
    matriz = defaultdict(lambda: defaultdict(int))

    # Construcción de la matriz: matriz[real][predicción]
    for real, pred in zip(y_true, y_pred):
        matriz[real][pred] += 1

    reporte = {}

    # Calcular la tasa de acierto por cada clase
    for clase in clases:
        total_reales = sum(matriz[clase].values())
        verdaderos_positivos = matriz[clase][clase]

        if total_reales == 0:
            precision = 0.0
        else:
            precision = verdaderos_positivos / total_reales

        # Importante: El generador usa el prefijo "Clase_" y redondea a 3 decimales
        reporte[f"Clase_{clase}"] = round(float(precision), 3)

    return reporte
