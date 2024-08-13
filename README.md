# Predicción de Lluvia al Día Siguiente

Este repositorio contiene la resolución de un problema de clasificación binaria que predice si lloverá o no al día siguiente, utilizando datos meteorológicos.

## Descripción del Proyecto

El objetivo de este proyecto es predecir si lloverá  al día siguiente basándose en un conjunto de características meteorológicas. Para lograrlo, se ha seguido un enfoque completo que incluye:

1. **Análisis Exploratorio de Datos (EDA)**: 
   - Se realizó un análisis exploratorio profundo para entender la distribución de las variables, relaciones entre ellas y posibles patrones que puedan influir en la predicción de la lluvia.

2. **Modelos de Machine Learning**:
   - **Regresión Logística**: Se utilizó como modelo base para la clasificación binaria.
   - **Árboles de Decisión**: Para capturar posibles interacciones no lineales entre las variables.
   - **Bosques Aleatorios**: Para mejorar la capacidad de generalización y reducir el sobreajuste mediante la agregación de múltiples árboles de decisión.
   - **XBoosting**

3. **Ajuste de Hiperparámetros**:
   - Se aplicó el tuning de hiperparámetros para optimizar el rendimiento de los modelos, utilizando técnicas como la validación cruzada y la búsqueda en cuadrícula.

## Contenido del Repositorio

- `weather_aus.csv`: Dataset utilizado para el análisis y la predicción.
- `weather_prediction.ipynb`: Notebook de Jupyter con el análisis exploratorio, implementación de modelos y ajuste de hiperparámetros.
- `README.md`: Este archivo, que proporciona una descripción general del proyecto.

## Resultados

Los modelos desarrollados fueron evaluados en términos de accuracy, AUC ROC, espe y f1_score para determinar el mejor enfoque para la predicción de la lluvia al día siguiente.

## Requisitos

- Python 3.11.9
- Librerías: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, scipy

## Uso

Clona este repositorio y ejecuta el notebook `weather_prediction.ipynb` para reproducir los resultados y explorar el análisis.

```bash
git clone https://github.com/tmatiwerbin/weather_prediction.git
cd weather_prediction
