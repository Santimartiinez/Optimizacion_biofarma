# Optimización del Rendimiento en Procesos de Biofabricación mediante Modelos Predictivos en Alta Dimensión

## Descripción
Este proyecto se centra en la predicción y optimización del rendimiento (`Yield`) en un proceso de manufactura biofarmacéutica mediante técnicas avanzadas de Machine Learning. El reto principal abordado es la alta dimensionalidad de los datos ($p$ grande frente a $n$) y la multicolinealidad severa. Para aislar los factores críticos que maximizan la producción sin caer en problemas de sobreajuste, el estudio implementa, evalúa y contrasta técnicas de regularización (Lasso y Ridge) y métodos de reducción de dimensionalidad (PLSR y PCR). Los resultados revelan que la calidad del material biológico y ciertos parámetros de control son los verdaderos determinantes del proceso.

## Conjunto de Datos
El análisis se fundamenta en el archivo `yield_phrama.csv`. 
- **Estructura**: El dataset contiene un total de 176 observaciones y 58 columnas (1 variable objetivo y 57 predictores continuos).
- **Contenido**: 
  - **Variable Objetivo**: `Yield` (Rendimiento).
  - **Predictores**: Características estrictamente cuantitativas divididas en variables de material biológico (ej. `BiologicalMaterial01` a `BiologicalMaterial12`) y variables de procesamiento o manufactura (ej. `ManufacturingProcess01` a `ManufacturingProcess45`).
- **Preprocesamiento**: El conjunto original presenta 106 valores ausentes (`NA`). Se realiza una imputación reemplazándolos con la media aritmética de cada columna para conservar el tamaño muestral.

## Instalación y Ejecución
Para reproducir el análisis o ejecutar el código deducido del archivo `optimizacion_biofarma.Rmd`:
1. Disponer de un entorno de ejecución de R (como RStudio).
2. Descargar el archivo de datos `gcd23_act2_yield_phrama.csv`.
3. Abrir el archivo `optimizacion_biofarma.Rmd`.
4. En el bloque de preparación de datos, actualizar la ruta del archivo en la función de carga para que apunte a tu directorio local: `read.csv("ruta/hacia/gcd23_act2_yield_phrama.csv")`.
5. Ejecutar los bloques de código secuencialmente. El script incluye semillas de aleatoriedad (`set.seed(123)`) preconfiguradas para asegurar que la partición del dataset (70% para entrenamiento y 30% para pruebas) y los resultados de la validación cruzada sean completamente reproducibles.

## Tecnologías y Dependencias
El proyecto está desarrollado en el lenguaje **R** y requiere los siguientes paquetes y librerías estadísticas:
- **`glmnet`**: Empleada para el entrenamiento de los modelos predictivos con técnicas de regularización $L_1$ (Lasso) y $L_2$ (Ridge) mediante validación cruzada.
- **`pls`**: Utilizada para los algoritmos de reducción de dimensionalidad espacial (PLSR - Partial Least Squares Regression y PCR - Principal Component Regression).
- **`ggplot2`**: Requerida para la visualización de datos y gráficos de dispersión en el plano bidimensional de puntuaciones (*Score Plots*).
