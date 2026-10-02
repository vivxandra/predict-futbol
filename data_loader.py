import pandas as pd

def cargar_datos(path):
    """
    Carga el CSV de partidos, convierte fechas y ordena cronológicamente.
    Devuelve un DataFrame listo para construir features.
    """
    df = pd.read_csv(path)

    # Convertir Date de texto a fecha real
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Ordenar cronológicamente - CRÍTICO para no usar info del futuro
    df = df.sort_values('Date').reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Esto solo corre si ejecutas este archivo directamente (para probarlo solo)
    df = cargar_datos("data/MEX.csv")
    print(df[['Date', 'Home', 'Away', 'Res']].head())
    print(f"\nTotal de partidos: {len(df)}")