# 🩺 Predicción de Diabetes  
[English version below ⬇️]  

**Sector:** Salud pública, Biotecnología, Medicina preventiva  
**Herramientas:** Python (Pandas, NumPy, Seaborn, Scikit-learn, SciPy, Matplotlib)  

---

## 📋 Descripción General  
Este proyecto utiliza el dataset **Pima Indians Diabetes** del *UCI Machine Learning Repository* para **predecir la presencia de diabetes** a partir de características médicas y demográficas.  

El objetivo principal es aplicar análisis exploratorio de datos (EDA), pruebas estadísticas y un modelo de clasificación binaria con *Random Forest*, contribuyendo al desarrollo de herramientas de detección temprana de diabetes y medicina personalizada.  

---

## 📊 Dataset  
- **Fuente:** [UCI Machine Learning Repository – Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Tamaño:** 768 instancias, 9 características  
- **Archivo:** `diabetes.csv`  

---

## 🔍 Metodología  
1. **Carga y Limpieza de Datos**  
   - Manejo de valores cero (Glucose, Insulin, BMI) reemplazados por medianas.  
   - Verificación de valores atípicos y distribución de variables.  

2. **Análisis Exploratorio (EDA)**  
   - Distribución del target: ~65 % no diabéticos, ~35 % diabéticos.  
   - Glucosa y BMI muestran sesgos y diferencias notables por clase.  
   - Correlación moderada entre *Pregnancies* y *Age* (~0.54).  

3. **Pruebas de Hipótesis**  
   - *t-test* confirma diferencia significativa en glucosa entre clases (p ≈ 2.64e-36).  

4. **Preparación y Modelado**  
   - División 80/20 (entrenamiento/prueba) con estratificación.  
   - Escalado con *StandardScaler*.  
   - Modelo: *Random Forest Classifier* (100 árboles, profundidad = 10).  

5. **Evaluación del Modelo**  
   - **Accuracy:** 72.73 %  
   - **F1-score:** clase 0 (0.80), clase 1 (0.58)  
   - **Principal predictor:** *Glucose* (importancia = 0.287)  

6. **Visualizaciones Clave**  
   - Histogramas, boxplots, matriz de confusión, gráfico de importancia de variables.  

---

## 🌎 Resultados Clave  
- Glucosa y BMI son los predictores más influyentes.  
- Buen desempeño general, aunque menor recall para la clase diabética (0.54).  
- Útil para *screening* temprano y evaluación de riesgo.  

---

## 🧠 Aplicaciones  
- Apoyo a diagnósticos médicos y prevención.  
- Clasificación automatizada de riesgo de diabetes.  
- Análisis de factores de riesgo poblacionales.  

---

## ⚙️ Requisitos de Ejecución  
- Python 3.8+  
- Librerías necesarias:  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`  

Instalación rápida:  
- pip install pandas numpy matplotlib seaborn scikit-learn scipy


---

## 🚀 Mejoras Futuras  
- Aplicar *SMOTE* o pesos balanceados para mitigar el desbalance de clases.  
- Probar modelos avanzados como *XGBoost* o *SVM*.  
- Desarrollar una interfaz web (*Streamlit*) para predicciones en tiempo real.  

---

## 👨‍💻 Autor  
**Adrián Galván Zamora**  
Proyecto académico orientado a la detección temprana de diabetes mediante análisis de datos y aprendizaje automático.  

---

# 🩺 Diabetes Prediction  

**Sector:** Public Health, Biotechnology, Preventive Medicine  
**Tools:** Python (Pandas, NumPy, Seaborn, Scikit-learn, SciPy, Matplotlib)  

---

## 📋 Overview  
This project uses the **Pima Indians Diabetes** dataset from the *UCI Machine Learning Repository* to **predict diabetes presence** based on medical and demographic attributes.  

It integrates exploratory data analysis (EDA), statistical hypothesis testing, and binary classification with *Random Forest* to support early detection and personalized medicine applications.  

---

## 📊 Dataset  
- **Source:** [UCI Machine Learning Repository – Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Size:** 768 instances, 9 features  
- **File:** `diabetes.csv`  

---

## 🔍 Methodology  
1. **Data Loading and Cleaning**  
   - Zero values (Glucose, Insulin, BMI) replaced by medians.  
   - Outlier detection and distribution review.  

2. **Exploratory Data Analysis (EDA)**  
   - Target distribution: ~65% non-diabetic, ~35% diabetic.  
   - Glucose and BMI show strong differences by class.  
   - Moderate correlation between *Pregnancies* and *Age* (~0.54).  

3. **Hypothesis Testing**  
   - *t-test* confirmed a significant difference in glucose levels (p ≈ 2.64e-36).  

4. **Preparation and Modeling**  
   - 80/20 train-test split with stratification.  
   - Scaling with *StandardScaler*.  
   - Model: *Random Forest Classifier* (100 trees, depth = 10).  

5. **Model Evaluation**  
   - **Accuracy:** 72.73 %  
   - **F1-score:** class 0 (0.80), class 1 (0.58)  
   - **Main predictor:** *Glucose* (importance = 0.287)  

6. **Key Visualizations**  
   - Histograms, boxplots, confusion matrix, feature importance plot.  

---

## 🌎 Key Results  
- Glucose and BMI are the strongest predictors.  
- Good general accuracy but lower recall for diabetic cases (0.54).  
- Useful for early screening and risk assessment.  

---

## 🧠 Real-World Applications  
- Medical diagnosis support and preventive healthcare.  
- Automated diabetes risk classification.  
- Population-level risk analysis.  

---

## ⚙️ Execution Requirements  
- Python 3.8+  
- Libraries required:  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`  

Quick installation:  
- pip install pandas numpy matplotlib seaborn scikit-learn scipy
  
---

## 🚀 Future Improvements  
- Apply *SMOTE* or class weighting to handle imbalance.  
- Test advanced models like *XGBoost* or *SVM*.  
- Build a *Streamlit* web app for real-time predictions.  

---

## 👨‍💻 Author  
**Adrián Galván Zamora**  
Academic project focused on early diabetes detection through data analysis and machine learning.  
