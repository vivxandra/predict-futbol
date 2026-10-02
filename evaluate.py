from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import log_loss


def cuotas_a_probabilidades(row):
    """Convierte cuotas de apuestas en probabilidades normalizadas (que sumen 100%)."""
    prob_h = 1 / row['AvgCH']
    prob_d = 1 / row['AvgCD']
    prob_a = 1 / row['AvgCA']

    total = prob_h + prob_d + prob_a  # > 1 por el margen de la casa

    return prob_h / total, prob_d / total, prob_a / total


def comparar_con_cuotas(modelo, X_test, y_test, df_original, features_cols):
    """
    Compara las probabilidades del modelo contra las probabilidades implícitas
    en las cuotas de apuestas, usando log_loss (menor = mejor calibrado).
    """
    # Recuperar las filas originales correspondientes al test set (para sacar las cuotas)
    df_test = df_original.loc[X_test.index]

    # Probabilidades del modelo: vienen en el orden alfabético de las clases (A, D, H)
    probs_modelo = modelo.predict_proba(X_test)
    print("Orden de clases del modelo:", modelo.classes_)

    # Probabilidades implícitas de las cuotas
    probs_cuotas = df_test.apply(cuotas_a_probabilidades, axis=1, result_type='expand')
    probs_cuotas.columns = ['prob_H', 'prob_D', 'prob_A']
    # Reordenar a A, D, H para que coincida con modelo.classes_
    probs_cuotas = probs_cuotas[['prob_A', 'prob_D', 'prob_H']].values

    logloss_modelo = log_loss(y_test, probs_modelo, labels=modelo.classes_)
    logloss_cuotas = log_loss(y_test, probs_cuotas, labels=modelo.classes_)

    print(f"\nLog loss de tu modelo:        {logloss_modelo:.4f}")
    print(f"Log loss de las casas de apuestas: {logloss_cuotas:.4f}")
    print("(más bajo = mejor calibrado; el mercado de apuestas es MUY difícil de vencer)")

    return logloss_modelo, logloss_cuotas



def diagnostico_detallado(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)

    print("Matriz de confusión (filas=real, columnas=predicho):")
    print("Orden: A, D, H")
    print(confusion_matrix(y_test, predicciones, labels=['A', 'D', 'H']))
    print()
    print(classification_report(y_test, predicciones))

    # Qué tanto pesa cada feature para el modelo
    print("Importancia de cada feature:")
    for nombre, importancia in zip(X_test.columns, modelo.feature_importances_):
        print(f"  {nombre}: {importancia:.3f}")


def calcular_baseline(y_test):
    """Accuracy de la estrategia tonta: 'siempre predice que gana el local'."""
    return (y_test == 'H').mean()


def evaluar(modelo, X_test, y_test):
    """Compara el accuracy del modelo contra el baseline y muestra el veredicto."""
    predicciones = modelo.predict(X_test)

    acc_modelo = accuracy_score(y_test, predicciones)
    acc_baseline = calcular_baseline(y_test)

    print(f"Accuracy del modelo:    {acc_modelo:.2%}")
    print(f"Accuracy del baseline:  {acc_baseline:.2%}")
    print()

    diferencia = acc_modelo - acc_baseline
    if diferencia > 0:
        print(f"✅ Tu modelo le gana al baseline por {diferencia:.2%}")
    else:
        print(f"⚠️  Tu modelo NO supera al baseline (diferencia: {diferencia:.2%})")
        print("    → Esto es normal con pocas features. Hay que enriquecer features.py")

    return acc_modelo, acc_baseline


if __name__ == "__main__":
    from data_loader import cargar_datos
    from features import construir_features
    from train import dividir_datos, entrenar_modelo

    df = cargar_datos("data/MEX.csv")
    df = construir_features(df, n=5)

    features_cols = ['goles_local_prom', 'goles_visita_prom']
    X_train, X_test, y_train, y_test = dividir_datos(df, features_cols)

    modelo = entrenar_modelo(X_train, y_train)
    evaluar(modelo, X_test, y_test)