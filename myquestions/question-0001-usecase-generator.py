import pandas as pd
import numpy as np

def generar_caso_de_uso_detectar_cambios_significativos():
    np.random.seed()

    n_empresas = 3
    n_dias = 60

    empresas = [f"Empresa_{i}" for i in range(n_empresas)]
    fechas = pd.date_range("2023-01-01", periods=n_dias)

    data = []

    for empresa in empresas:
        precio = 100
        for fecha in fechas:
            cambio = np.random.normal(0, 1)
            precio = precio * (1 + cambio / 100)
            data.append([empresa, fecha, precio])

    df = pd.DataFrame(data, columns=["empresa", "fecha", "precio"])

    df = df.sort_values(["empresa", "fecha"])
    df["retorno_diario"] = df.groupby("empresa")["precio"].pct_change()
    df["volatilidad_movil"] = (
        df.groupby("empresa")["retorno_diario"]
        .rolling(30)
        .std()
        .reset_index(level=0, drop=True)
    )

    umbral = 2.5

    df["evento_anomalo"] = (
        abs(df["retorno_diario"]) > umbral * df["volatilidad_movil"]
    )

    return {
        "input": {
            "df": df[["empresa", "fecha", "precio"]],
            "empresa_col": "empresa",
            "fecha_col": "fecha",
            "precio_col": "precio",
            "umbral": umbral
        },
        "output": df
    }
