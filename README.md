# Predicción de Diabetes

## Descripción del Proyecto
Este proyecto utiliza el dataset Pima Indians Diabetes del UCI Machine Learning Repository para predecir la presencia de diabetes basándose en características médicas y demográficas. Es relevante para biotech, salud pública y medicina personalizada, con aplicaciones en detección temprana y screening de diabetes.

- **Dataset:** Pima Indians Diabetes (768 instancias, 9 features).
- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- **Herramientas:** Python con Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y SciPy.
- **Objetivos:**
  - Realizar análisis exploratorio de datos (EDA) para identificar patrones.
  - Limpiar datos (manejo de valores cero como faltantes).
  - Pruebas de hipótesis (e.g., diferencia en glucosa entre diabéticos y no diabéticos).
  - Modelado de clasificación binaria con Random Forest.
  - Evaluación con accuracy (~73%), F1-score y matriz de confusión.
  - Visualizaciones: histogramas, boxplots, mapa de calor, importancia de variables.

## Requisitos
- Python 3.8+.
- Bibliotecas: Instala con `pip install pandas numpy matplotlib seaborn scikit-learn scipy`.
- Dataset: Descarga de [aquí](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) y coloca `diabetes.csv` en la carpeta del notebook.

## Metodología
1. **Carga y Limpieza:** Dataset cargado con 768 instancias, manejo de valores cero (e.g., Glucose, Insulin) reemplazados por medianas.
2. **EDA:**
   - Distribución del target: ~65% no diabéticos, ~35% diabéticos.
   - Histogramas: Variables como Glucose y BMI muestran sesgos.
   - Boxplots: Glucosa más alta en diabéticos.
   - Correlaciones: Moderadas (e.g., Pregnancies y Age ~0.54).
3. **Pruebas de Hipótesis:** t-test confirma diferencia significativa en glucosa (p-value: 2.64e-36).
4. **Preparación:** Codificación de variables, split 80/20 con estratificación (X_train: 614; X_test: 154), escalado con StandardScaler.
5. **Modelado:** Random Forest (100 árboles, profundidad 10). Accuracy: 72.73%.
6. **Evaluación:** F1-scores (0.80 clase 0, 0.58 clase 1), con Glucose como predictor principal (importancia: 0.287).
7. **Visualizaciones:** Matriz de confusión, gráfico de importancia de features.

## Resultados Clave
- **Accuracy:** 72.73%.
- **Mejor rendimiento:** Clase no diabética (f1-score 0.80).
- **Insights:** Glucosa y BMI son predictores clave, útil para screening de diabetes.
- **Limitaciones:** Desbalance (~65% no diabéticos) reduce recall para diabéticos (0.54).

## Cómo Ejecutar
1. Descarga el dataset y coloca `diabetes.csv` en la carpeta.
2. Abre `Prediccion_diabetes.ipynb` en Jupyter Notebook.
3. Ejecuta las celdas en orden.
4. Nota: El entrenamiento toma ~1 minuto.

## Mejoras Futuras
- Usar SMOTE o ajuste de pesos para balancear clases.
- Probar XGBoost o SVM para mejorar recall en diabéticos.
- Desarrollar una interfaz web para predicciones en tiempo real.
