import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def analizar_estado_bomba(df_entrenamiento, datos_nuevos):
    """
    Entrena un modelo de RandomForest y predice el estado de una bomba.
    """
    # 1. Separar características (X) y objetivo (y)
    columnas_feat = ['flujo_m3h', 'presion_bar', 'temp_c', 'potencia_kw']
    X_train = df_entrenamiento[columnas_feat]
    y_train = df_entrenamiento['estado']

    # 2. Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Entrenar el modelo (Importante: usar random_state=42 como en el generador)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # 4. Preparar el nuevo registro para predicción
    # Se convierte a DataFrame para mantener consistencia con los nombres de columnas
    X_new_df = pd.DataFrame([datos_nuevos], columns=columnas_feat)
    X_new_scaled = scaler.transform(X_new_df)

    # 5. Predecir clase y probabilidad
    pred = clf.predict(X_new_scaled)[0]
    prob = clf.predict_proba(X_new_scaled)[0][1]

    # 6. Formatear la salida según el requisito exacto del compañero
    txt_resultado = "Crítico (Riesgo de Cavitación)" if pred == 1 else "Operativo (Normal)"
    
    return {
        "estado_predicho": txt_resultado,
        "probabilidad_critica": f"{prob * 100:.2f}%"
    }
