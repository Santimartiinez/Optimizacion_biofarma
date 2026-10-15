# Optimización del Rendimiento en Procesos de Biofabricación mediante Modelos Predictivos en Alta Dimensión

## Descripción General
El presente proyecto se centra en la predicción y optimización del rendimiento (*Yield*) en un proceso de manufactura biofarmacéutica empleando técnicas avanzadas de Machine Learning. El reto principal abordado es el escenario matemático de alta dimensionalidad ($p$ grande frente a $n$) típico en la bioingeniería, donde los datos presentan una multicolinealidad severa. Esto provoca que los modelos de Regresión Lineal Múltiple clásica (MCO) sufran de varianza inflada y sobreajuste. Para resolverlo, este estudio implementa arquitecturas de regularización (restricción del espacio de hipótesis) y reducción de dimensionalidad, contrastando su efectividad para aislar los factores críticos que maximizan la producción y aportando valor analítico para la toma de decisiones y el escalado industrial.

## Conjunto de Datos
El estudio se realiza sobre un archivo de datos denominado `gcd23_act2_yield_phrama.csv`. 
* **Dimensiones:** $n = 176$ observaciones y $p = 57$ variables predictoras continuas.
* **Variable Objetivo:** `Yield` (Rendimiento del proceso biofarmacéutico).
* **Características:** Las variables independientes engloban métricas altamente dimensionales, combinando características de los materiales biológicos (ej. `BiologicalMaterial02`, `04`, etc.) y variables de control de procesamiento (ej. `ManufacturingProcess29`).

## Desarrollo y Metodología

### Limpieza y Preparación de Datos
El proceso de preprocesamiento se diseñó para garantizar la viabilidad y estabilidad de los modelos matemáticos en alta dimensión:
1. **Diagnóstico y Tratamiento de Valores Ausentes:** La exploración inicial detectó 106 valores nulos (`NA`) dispersos en el conjunto de datos. Al ser un volumen crítico que comprometería el análisis dado el tamaño muestral reducido ($n=176$), se aplicó una imputación de valores faltantes iterando sobre cada columna e inyectando la **media aritmética** respectiva (`mean(..., na.rm = TRUE)`). Esto permitió conservar todas las observaciones sin sesgar la tendencia central.
2. **Verificación Estructural:** Se confirmó a través de `str()` que ninguna de las 57 variables era categórica, asegurando compatibilidad nativa con las operaciones matriciales posteriores.
3. **Partición de Datos:** Utilizando una semilla de reproducibilidad (`set.seed(123)`), el conjunto se dividió aleatoriamente en:
   * **Training Set (70%):** 123 observaciones para entrenar los modelos.
   * **Testing Set (30%):** 53 observaciones para validar la capacidad de generalización.
   Los datos se separaron funcionalmente en matrices de predictores (`x_train`, `x_test`) y vectores de respuesta (`y_train`, `y_test`).

### Análisis Exploratorio y Visualización
La comprensión del espacio de los datos se apoyó en técnicas de reducción y visualizaciones enfocadas en el *Yield*:
1. **Análisis de Cargas (Loadings):** Se examinaron los pesos absolutos de las dos primeras componentes en los modelos de reducción de dimensionalidad para determinar qué características influyen más en el modelo. Se aisló el top 10 de variables, destacando los factores biológicos y de manufactura.
2. **Proyección en el Espacio Latente (Score Plots):** Se generaron gráficos de dispersión (2D) usando `ggplot2` mapeando las Componentes Principales 1 y 2 (para PCR) y las Variables Latentes 1 y 2 (para PLSR). 
3. **Gradiente de Rendimiento:** A cada punto de los *score plots* se le aplicó un gradiente de color basado en la variable `Yield` (de azul a rojo). Se observó visualmente que PLSR logra una proyección del `Yield` notablemente más ordenada y homogénea que PCR, reflejando cómo alinea directamente sus ejes latentes para la optimización del producto.

### Modelado y Pruebas Estadísticas
Se aplicaron cuatro enfoques avanzados, divididos en dos grandes estrategias:

**A. Técnicas de Regularización (glmnet)**
* **Lasso (Penalización $L_1$, $\alpha = 1$):** Ajustado mediante validación cruzada (`cv.glmnet`). Se analizó la curva del Error Cuadrático Medio (MSE) frente al logaritmo de $\lambda$, identificando un mínimo entre los valores 2 y 4 de $\lambda$. El modelo final logró producir esparsidad, aislando el ruido al reducir la dimensión de 57 a solo **31 predictores activos** (forzando los 26 restantes exactamente a cero).
* **Ridge (Penalización $L_2$, $\alpha = 0$):** Se aplicó una contracción asintótica a través de validación cruzada. Aunque logró estabilizar los coeficientes frente a la multicolinealidad, mantuvo los **57 predictores** originales sin anular ninguno, mostrando menor capacidad que Lasso para simplificar el espacio de variables.

**B. Técnicas de Reducción de Dimensionalidad (pls)**
* **PLSR (Regresión de Mínimos Cuadrados Parciales):** Método supervisado calibrado por validación cruzada. El error RMSEP cayó pronunciadamente en las primeras dimensiones, requiriendo únicamente **4 componentes latentes** óptimos para modelar el proceso, ya que maximiza simultáneamente la varianza explicada en $X$ y su covarianza con la respuesta `Yield`.
* **PCR (Regresión de Componentes Principales):** Método no supervisado. Su perfil de error (RMSEP) mostró un descenso escalonado y requirió **19 componentes principales** óptimos para capturar la señal objetivo de `Yield`, lo que resultó en un subespacio de mayor complejidad que el obtenido mediante PLSR.

Para medir el desempeño final, se generó una función de cálculo de RMSE y se testearon los cuatro modelos (con sus hiperparámetros óptimos) frente a los datos invisibles de prueba (`x_test`), organizando los resultados de menor a mayor error.

## Conclusiones y Hallazgos
* **Mejor Capacidad Predictiva:** El modelo **PCR** logró la mayor capacidad predictiva global en el conjunto de prueba, alcanzando el menor error cuadrático con un **RMSE = 1.430** apoyándose en 19 componentes.
* **Mejor Valor Operativo y Simplicidad:** **Lasso** destacó como la alternativa más valiosa para la toma de decisiones. Logró descartar 26 variables irrelevantes (quedándose con 31 activas) con un incremento de error muy moderado en comparación con PCR, permitiendo un análisis altamente interpretable.
* **Factores Determinantes Aislados:** El análisis de las cargas bajo PLSR demostró consistentemente que la calidad de los materiales biológicos (específicamente `BiologicalMaterial02`, `04`, `06` y `08`) y el parámetro de control `ManufacturingProcess29` son los verdaderos factores críticos que gobiernan el proceso.
* **Impacto:** Los resultados establecen un marco robusto que supera los límites de los modelos clásicos ante la alta dimensionalidad y multicolinealidad, optimizando no solo las predicciones futuras, sino apoyando el control de calidad y las decisiones informadas en la planta biofarmacéutica.

## Tecnologías y Librerías
Este proyecto fue desarrollado en el entorno computacional estadístico **R**, utilizando los siguientes paquetes fundamentales:
* **`glmnet`:** Para la implementación y validación cruzada de modelos de regularización elástica (Lasso y Ridge).
* **`pls`:** Para la generación y validación de los modelos de reducción de dimensionalidad (PLSR y PCR).
* **`ggplot2`:** Empleada para el mapeo bidimensional y la creación de los gradientes visuales en el análisis de interpretabilidad del espacio latente.
* **Base R:** Herramientas nativas para la lectura de datos (`read.csv`), manipulación de NA's, partición de tensores y cálculo de métricas de desempeño.
