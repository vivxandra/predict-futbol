from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def dividir_datos(df, features_cols, target_col='Res', test_size=0.2):
    X = df[features_cols]
    y = df[target_col]

    split_idx = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def entrenar_modelo(X_train, y_train, max_depth=5, min_samples_leaf=20):
    """Versión simple, sin búsqueda de hiperparámetros (la dejamos por si acaso)."""
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42
    )
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_modelo_con_grid_search(X_train, y_train):
    """Prueba varias combinaciones de hiperparámetros y se queda con la mejor."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [10, 20, 40],
    }

    modelo_base = RandomForestClassifier(class_weight='balanced', random_state=42)

    # cv=5 significa que prueba cada combinación 5 veces con distintos sub-splits
    grid = GridSearchCV(modelo_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Mejores parámetros encontrados: {grid.best_params_}")
    print(f"Accuracy en validación cruzada: {grid.best_score_:.2%}")

    return grid.best_estimator_

if __name__ == "__main__":
    # Prueba aislada de este archivo
    from data_loader import cargar_datos
    from features import construir_features

    df = cargar_datos("data/MEX.csv")
    df = construir_features(df, n=5)

    features_cols = ['goles_local_prom', 'goles_visita_prom']
    X_train, X_test, y_train, y_test = dividir_datos(df, features_cols)

    print(f"Entrenamiento: {len(X_train)} partidos")
    print(f"Prueba: {len(X_test)} partidos")

    modelo = entrenar_modelo(X_train, y_train)
    predicciones = modelo.predict(X_test)

    print(f"\nPrimeras 10 predicciones: {predicciones[:10]}")
    print(f"Primeros 10 resultados reales: {y_test.values[:10]}")