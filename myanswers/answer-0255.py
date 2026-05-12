import numpy as np
import pandas as pd

def calcular_numero_condicion(df):
    """
    Calcula el número de condición de la matriz de correlación de las
    columnas numéricas de un DataFrame.
    """
    # 1. Seleccionar todas las columnas numéricas
    numericas = df.select_dtypes(include=[np.number])

    # 2. Calcular la matriz de correlación de Pearson
    corr_matrix = numericas.corr()

    # 3. Calcular los autovalores de la matriz de correlación
    # Usamos .values para pasar la matriz como array de numpy
    eigenvalues = np.linalg.eigvals(corr_matrix.values)

    # 4. Calcular el número de condición
    abs_eigs = np.abs(eigenvalues)
    eig_max = abs_eigs.max()
    eig_min = abs_eigs.min()

    # 5. Manejo de estabilidad numérica (evitar división por cero)
    if eig_min == 0:
        return float(np.inf)
    else:
        resultado = np.sqrt(eig_max / eig_min)
        return float(resultado)
