import streamlit as st
import pandas as pd
import joblib

from features import (
    goles_local_como_local, goles_recibidos_local_como_local,
    goles_visita_como_visita, goles_recibidos_visita_como_visita,
    racha_ultimos_n, historial_h2h
)

st.set_page_config(page_title="Predictor Liga MX", page_icon="⚽")


@st.cache_resource
def cargar_todo():
    modelo = joblib.load("modelo_entrenado.pkl")
    df = joblib.load("df_historico.pkl")
    return modelo, df


modelo, df = cargar_todo()
equipos = sorted(set(df['Home']) | set(df['Away']))

FEATURES_COLS = [
    'goles_local_ataque', 'goles_local_defensa',
    'goles_visita_ataque', 'goles_visita_defensa',
    'racha_local', 'racha_visita',
    'h2h_puntos_local'
]

st.title("⚽ Predictor Liga MX")

modo = st.radio(
    "¿Qué quieres hacer?",
    ["🔮 Predecir un partido nuevo", "📊 Comparar contra un partido real (backtest)"]
)

# ============================================================
# MODO 1: Predecir un partido hipotético
# ============================================================
if modo == "🔮 Predecir un partido nuevo":
    st.caption("Selecciona dos equipos para predecir el resultado")

    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("Equipo local", equipos, index=0)
    with col2:
        equipo_visita = st.selectbox("Equipo visitante", equipos, index=1)

    n = 10
    fecha_prediccion = df['Date'].max() + pd.Timedelta(days=1)

    if st.button("🔮 Predecir resultado", type="primary"):
        if equipo_local == equipo_visita:
            st.error("Selecciona dos equipos distintos")
        else:
            features = {
                'goles_local_ataque': goles_local_como_local(df, equipo_local, fecha_prediccion, n),
                'goles_local_defensa': goles_recibidos_local_como_local(df, equipo_local, fecha_prediccion, n),
                'goles_visita_ataque': goles_visita_como_visita(df, equipo_visita, fecha_prediccion, n),
                'goles_visita_defensa': goles_recibidos_visita_como_visita(df, equipo_visita, fecha_prediccion, n),
                'racha_local': racha_ultimos_n(df, equipo_local, fecha_prediccion, n),
                'racha_visita': racha_ultimos_n(df, equipo_visita, fecha_prediccion, n),
                'h2h_puntos_local': historial_h2h(df, equipo_local, equipo_visita, fecha_prediccion, n),
            }

            if any(v is None for v in features.values()):
                st.warning("⚠️ No hay suficiente historial para uno de estos equipos")
            else:
                X_nuevo = pd.DataFrame([features])
                probs = modelo.predict_proba(X_nuevo)[0]
                prob_dict = dict(zip(modelo.classes_, probs))

                st.subheader(f"{equipo_local} 🆚 {equipo_visita}")

                colA, colD, colH = st.columns(3)
                colA.metric("Gana visitante", f"{prob_dict['A']:.1%}")
                colD.metric("Empate", f"{prob_dict['D']:.1%}")
                colH.metric("Gana local", f"{prob_dict['H']:.1%}")

                resultado = max(prob_dict, key=prob_dict.get)
                mapa = {'H': f'🏠 Gana {equipo_local}', 'D': '🤝 Empate', 'A': f'✈️ Gana {equipo_visita}'}
                st.success(f"**Predicción:** {mapa[resultado]}")

                with st.expander("Ver features usadas para esta predicción"):
                    st.json(features)

# ============================================================
# MODO 2: Backtest contra partidos reales del set de prueba
# ============================================================
else:
    st.caption("Elige un partido ya jugado (de los que el modelo NUNCA vio en entrenamiento) y compara")

    # Reconstruir el mismo split cronológico usado en el entrenamiento
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].sort_values('Date', ascending=False)

    opciones = [
        f"{row.Date.date()} — {row.Home} vs {row.Away} ({row.Res})"
        for _, row in df_test.iterrows()
    ]
    seleccion = st.selectbox("Partido a analizar", opciones)
    partido = df_test.iloc[opciones.index(seleccion)]

    if st.button("📊 Comparar", type="primary"):
        # Probabilidades del modelo (usa las features YA calculadas para ese partido)
        X_partido = pd.DataFrame([partido[FEATURES_COLS].to_dict()])
        probs_modelo = modelo.predict_proba(X_partido)[0]
        prob_modelo = dict(zip(modelo.classes_, probs_modelo))

        # Probabilidades implícitas del mercado (normalizadas)
        prob_h_cuota = 1 / partido['AvgCH']
        prob_d_cuota = 1 / partido['AvgCD']
        prob_a_cuota = 1 / partido['AvgCA']
        total = prob_h_cuota + prob_d_cuota + prob_a_cuota
        prob_cuota = {
            'H': prob_h_cuota / total,
            'D': prob_d_cuota / total,
            'A': prob_a_cuota / total,
        }

        st.subheader(f"{partido['Home']} 🆚 {partido['Away']}")
        st.write(f"📅 {partido['Date'].date()}  —  **Resultado real: {partido['HG']}-{partido['AG']} ({partido['Res']})**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🤖 Tu modelo")
            st.metric("Gana visitante", f"{prob_modelo['A']:.1%}")
            st.metric("Empate", f"{prob_modelo['D']:.1%}")
            st.metric("Gana local", f"{prob_modelo['H']:.1%}")
        with col2:
            st.markdown("### 💰 Mercado de apuestas")
            st.metric("Gana visitante", f"{prob_cuota['A']:.1%}")
            st.metric("Empate", f"{prob_cuota['D']:.1%}")
            st.metric("Gana local", f"{prob_cuota['H']:.1%}")

        resultado_real = partido['Res']
        acierto_modelo = max(prob_modelo, key=prob_modelo.get) == resultado_real
        acierto_mercado = max(prob_cuota, key=prob_cuota.get) == resultado_real

        if acierto_modelo and not acierto_mercado:
            st.success("🎉 ¡Tu modelo acertó y el mercado no!")
        elif acierto_modelo and acierto_mercado:
            st.info("Ambos acertaron el resultado")
        elif not acierto_modelo and acierto_mercado:
            st.warning("El mercado acertó, tu modelo no esta vez")
        else:
            st.error("Ninguno de los dos acertó — este partido fue una sorpresa")