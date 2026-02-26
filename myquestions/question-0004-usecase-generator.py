import pandas as pd
import numpy as np

def generar_caso_de_uso_detectar_data_drift():
    np.random.seed()

    n = 100

    df_base = pd.DataFrame({
        "A": np.random.normal(0, 1, n),
        "B": np.random.normal(5, 2, n),
        "C": np.random.normal(-3, 1, n)
    })

    df_nuevo = pd.DataFrame({
        "A": np.random.normal(0.2, 1, n),
        "B": np.random.normal(6, 2, n),
        "C": np.random.normal(-3, 1, n)
    })

    columnas = ["A", "B", "C"]
    umbral = 0.1

    metricas = {}
    columnas_con_drift = []

    for col in columnas:
        media_base = df_base[col].mean()
        media_nuevo = df_nuevo[col].mean()

        diferencia = abs(media_base - media_nuevo) / abs(media_base)
        metricas[col] = diferencia

        if diferencia > umbral:
            columnas_con_drift.append(col)

    return {
        "input": {
            "df_base": df_base,
            "df_nuevo": df_nuevo,
            "columnas": columnas,
            "umbral": umbral
        },
        "output": {
            "columnas_con_drift": columnas_con_drift,
            "metricas": metricas
        }
    }
