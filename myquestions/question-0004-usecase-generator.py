import pandas as pd
import numpy as np

def generar_caso_de_uso_detectar_data_drift():
    np.random.seed()

    n = 300

    df_base = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n),
        "feature2": np.random.normal(5, 2, n),
        "feature3": np.random.normal(-2, 1.5, n)
    })

    df_nuevo = pd.DataFrame({
        "feature1": np.random.normal(0.5, 1, n),
        "feature2": np.random.normal(5, 2.5, n),
        "feature3": np.random.normal(-1.5, 1.5, n)
    })

    threshold = np.random.uniform(0.1, 0.3)

    media_base = df_base.mean()
    media_nuevo = df_nuevo.mean()

    diferencia_relativa = abs(media_base - media_nuevo) / abs(media_base)

    drift_detectado = diferencia_relativa > threshold

    output_df = pd.DataFrame({
        "media_base": media_base,
        "media_nuevo": media_nuevo,
        "diferencia_relativa": diferencia_relativa,
        "drift_detectado": drift_detectado
    })

    input_data = {
        "df_base": df_base,
        "df_nuevo": df_nuevo,
        "threshold": threshold
    }

    output_data = output_df

    return input_data, output_data
