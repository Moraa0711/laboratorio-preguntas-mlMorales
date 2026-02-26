import pandas as pd
import numpy as np

def generar_caso_de_uso_detectar_cambios_significativos():
    np.random.seed()

    empresas = ['A', 'B']
    fechas = pd.date_range(start="2023-01-01", periods=120, freq='D')

    data = []
    for empresa in empresas:
        precio_base = np.random.uniform(50, 150)
        precios = precio_base + np.random.normal(0, 5, size=len(fechas)).cumsum()
        for fecha, precio in zip(fechas, precios):
            data.append([empresa, fecha, precio])

    df_input = pd.DataFrame(data, columns=["empresa", "fecha", "precio"])

    umbral = np.random.uniform(0.05, 0.15)

    df_resultado = df_input.copy()
    df_resultado = df_resultado.sort_values(by=["empresa", "fecha"])
    df_resultado["media_movil_30"] = (
        df_resultado.groupby("empresa")["precio"]
        .transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    )

    df_resultado["desviacion_pct"] = (
        abs(df_resultado["precio"] - df_resultado["media_movil_30"])
        / df_resultado["media_movil_30"]
    )

    df_resultado["cambio_significativo"] = df_resultado["desviacion_pct"] > umbral

    input_data = {
        "df": df_input,
        "empresa_col": "empresa",
        "fecha_col": "fecha",
        "precio_col": "precio",
        "umbral": umbral
    }

    output_data = df_resultado

    return input_data, output_data
