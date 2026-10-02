from data_loader import cargar_datos
from features import construir_features
from train import dividir_datos, entrenar_modelo_con_grid_search
from evaluate import evaluar, diagnostico_detallado, comparar_con_cuotas
import joblib

df = cargar_datos("data/MEX.csv")
df = construir_features(df, n=10)

features_cols = [
    'goles_local_ataque', 'goles_local_defensa',
    'goles_visita_ataque', 'goles_visita_defensa',
    'racha_local', 'racha_visita',
    'h2h_puntos_local'
]

X_train, X_test, y_train, y_test = dividir_datos(df, features_cols)

modelo = entrenar_modelo_con_grid_search(X_train, y_train)
evaluar(modelo, X_test, y_test)
diagnostico_detallado(modelo, X_test, y_test)
comparar_con_cuotas(modelo, X_test, y_test, df, features_cols)


joblib.dump(modelo, "modelo_entrenado.pkl")
joblib.dump(df, "df_historico.pkl")  # guardamos también el histórico para calcular features de partidos nuevos
print("\n✅ Modelo y datos guardados en disco")