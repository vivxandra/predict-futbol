# ⚽ Predictor de Resultados de Fútbol & Dashboard Interactivo

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

Sistema de Machine Learning end-to-end diseñado para predecir resultados de partidos de fútbol a partir de datos históricos, ingeniería de características (feature engineering) y evaluación de probabilidades en tiempo real a través de un dashboard interactivo en Streamlit.

---

## 🌟 Características Principales

- **Pipeline de Datos Integrado**: Carga, limpia y procesa datos históricos de encuentros deportivos.
- **Ingeniería de Características (Feature Engineering)**:
  - Cálculo de rendimiento reciente (forma de los últimos $N$ partidos).
  - Promedio de goles anotados y recibidos en condición de local/visitante.
  - Métricas de enfrentamientos directos (Head-to-Head).
- **Entrenamiento con Optimización de Hiperparámetros**:
  - Clasificador `RandomForestClassifier` optimizado mediante `GridSearchCV`.
  - Validación cruzada y evaluación con métricas de Log-Loss, Accuracy y Matriz de Confusión.
- **Dashboard Interactivo en Streamlit**:
  - Selección de equipos en tiempo real para predecir probabilidad de Victoria Local, Empate o Victoria Visitante.
  - Visualización comparativa del modelo vs. cuotas implícitas de casas de apuestas.

---

## 📁 Estructura del Proyecto

```text
predict_futbol/
├── data_loader.py         # Carga y preprocesamiento del dataset histórico
├── features.py            # Generación de métricas y variables predictivas
├── train.py               # Entrenamiento del modelo y optimización (GridSearch)
├── evaluate.py            # Módulo de métricas de rendimiento y diagnóstico
├── main.py                # Pipeline principal de ejecución
├── dashboard.py           # Aplicación interactiva web (Streamlit)
├── df_historico.pkl       # Dataset procesado y persistido
└── modelo_entrenado.pkl   # Modelo entrenado listo para inferencia
```

---

## 🛠️ Requisitos e Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/vivxandra/predict-futbol.git
   cd predict-futbol
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Uso del Proyecto

### 1. Ejecutar el Dashboard Interactivo
Para explorar las predicciones de forma visual e interactiva:
```bash
streamlit run dashboard.py
```

### 2. Entrenar / Re-evaluar el Modelo
Para ejecutar el pipeline completo de carga, entrenamiento y evaluación desde consola:
```bash
python3 main.py
```

---

## 👤 Autor

Desarrollado por **Vivian Suastegui** ([@vivxandra](https://github.com/vivxandra)).
