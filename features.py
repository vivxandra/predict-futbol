import pandas as pd

def goles_local_como_local(df, equipo, fecha, n=5):
    """Promedio de goles que anota 'equipo' SOLO en sus partidos de local."""
    partidos = df[
        (df['Home'] == equipo) &
        (df['Date'] < fecha)
    ].tail(n)

    if len(partidos) == 0:
        return None

    return partidos['HG'].mean()


def goles_visita_como_visita(df, equipo, fecha, n=5):
    """Promedio de goles que anota 'equipo' SOLO en sus partidos de visita."""
    partidos = df[
        (df['Away'] == equipo) &
        (df['Date'] < fecha)
    ].tail(n)

    if len(partidos) == 0:
        return None

    return partidos['AG'].mean()


def goles_recibidos_local_como_local(df, equipo, fecha, n=5):
    """Promedio de goles que RECIBE 'equipo' jugando de local."""
    partidos = df[
        (df['Home'] == equipo) &
        (df['Date'] < fecha)
    ].tail(n)

    if len(partidos) == 0:
        return None

    return partidos['AG'].mean()  # AG = goles del rival


def goles_recibidos_visita_como_visita(df, equipo, fecha, n=5):
    """Promedio de goles que RECIBE 'equipo' jugando de visita."""
    partidos = df[
        (df['Away'] == equipo) &
        (df['Date'] < fecha)
    ].tail(n)

    if len(partidos) == 0:
        return None

    return partidos['HG'].mean()  # HG = goles del rival


def racha_ultimos_n(df, equipo, fecha, n=5):
    """Cuenta victorias en los últimos n partidos (0 a n), sin importar local/visita."""
    partidos = df[
        ((df['Home'] == equipo) | (df['Away'] == equipo)) &
        (df['Date'] < fecha)
    ].tail(n)

    victorias = 0
    for _, row in partidos.iterrows():
        gano_local = row['Home'] == equipo and row['Res'] == 'H'
        gano_visita = row['Away'] == equipo and row['Res'] == 'A'
        if gano_local or gano_visita:
            victorias += 1

    return victorias

def historial_h2h(df, equipo_a, equipo_b, fecha, n=5):
    """
    Promedio de puntos que ha sacado 'equipo_a' en sus últimos n enfrentamientos
    directos contra 'equipo_b' (sin importar quién fue local en esos partidos).
    Puntos: victoria=3, empate=1, derrota=0.
    """
    partidos = df[
        (
            ((df['Home'] == equipo_a) & (df['Away'] == equipo_b)) |
            ((df['Home'] == equipo_b) & (df['Away'] == equipo_a))
        ) &
        (df['Date'] < fecha)
    ].tail(n)

    if len(partidos) == 0:
        return None  # nunca se han enfrentado antes (o no hay historial en el dataset)

    puntos_a = 0
    for _, row in partidos.iterrows():
        if row['Home'] == equipo_a:
            if row['Res'] == 'H':
                puntos_a += 3
            elif row['Res'] == 'D':
                puntos_a += 1
        else:  # equipo_a jugó de visita en ese enfrentamiento histórico
            if row['Res'] == 'A':
                puntos_a += 3
            elif row['Res'] == 'D':
                puntos_a += 1

    return puntos_a / len(partidos)



def construir_features(df, n=5):
    filas_features = []

    for idx, row in df.iterrows():
        equipo_local = row['Home']
        equipo_visita = row['Away']
        fecha = row['Date']

        filas_features.append({
            'goles_local_ataque': goles_local_como_local(df, equipo_local, fecha, n),
            'goles_local_defensa': goles_recibidos_local_como_local(df, equipo_local, fecha, n),
            'goles_visita_ataque': goles_visita_como_visita(df, equipo_visita, fecha, n),
            'goles_visita_defensa': goles_recibidos_visita_como_visita(df, equipo_visita, fecha, n),
            'racha_local': racha_ultimos_n(df, equipo_local, fecha, n),
            'racha_visita': racha_ultimos_n(df, equipo_visita, fecha, n),
            'h2h_puntos_local': historial_h2h(df, equipo_local, equipo_visita, fecha, n),
        })

    df_features = pd.DataFrame(filas_features, index=df.index)
    df = pd.concat([df, df_features], axis=1)

    columnas_nuevas = list(df_features.columns)
    df_limpio = df.dropna(subset=columnas_nuevas)
    return df_limpio



if __name__ == "__main__":
    from data_loader import cargar_datos

    df = cargar_datos("data/MEX.csv")
    df = construir_features(df, n=5)
    print(df[['Date', 'Home', 'Away', 'goles_local_ataque', 'goles_visita_ataque',
              'racha_local', 'racha_visita', 'Res']].head(10))
    print(f"\nPartidos utilizables: {len(df)}")