## 1. Metodología de Trabajo: CRISP-DM

El proyecto seguirá la metodología **CRISP-DM (Cross-Industry Standard Process for Data Mining)** porque es el estándar más utilizado y probado para proyectos de minería de datos y aprendizaje automático y también para asegurar un desarrollo estructurado, iterativo y orientado a los objetivos. A continuación, se detalla el enfoque que se tomará en cada una de las fases y se enlaza al documento correspondiente que contendrá los entregables de cada etapa.

### Fase 1: Comprensión del Problema (Business Understanding)
**Enfoque:** En esta fase inicial, traduciremos el objetivo general del proyecto en un problema técnico de analítica de datos. Definiremos las metas específicas, los requerimientos desde una perspectiva técnica y los criterios que determinarán el éxito del proyecto. El resultado principal de esta fase son las **preguntas de interés** que guiarán toda la investigación.
*   **Estado:** En progreso (Parte de la **Entrega 1**).
*   **Documento Detallado:** [**Ver Documento de Comprensión del Problema](./1_Comprension_de_problema.md)**](_blank)

### Fase 2: Comprensión de los Datos (Data Understanding)
**Enfoque:** Esta fase se centra en la recolección inicial de datos de video. Realizaremos un análisis exploratorio para familiarizarnos con los datos, identificar posibles problemas de calidad y descubrir primeros insights. Se definirá la estrategia para la captura de un conjunto de datos diverso y representativo.
*   **Estado:** En progreso (Parte de la **Entrega 1**).
*   **Documento Detallado:** [**Ver Documento de Comprensión de los Datos](./2_Comprension_de_los_datos.md)**](_blank)

### Fase 3: Preparación de los Datos (Data Preparation)
**Enfoque:** Aquí se realizarán todas las actividades para construir el conjunto de datos final que se usará para el modelado. Esto incluye el etiquetado (anotación) de los videos, la extracción de landmarks con MediaPipe, la limpieza de datos (filtrado de ruido), la normalización de coordenadas y la creación de características (feature engineering) como velocidades y ángulos.
*   **Estado:** Pendiente (Parte de la **Entrega 2**).
*   **Documento Detallado:** [**Ver Documento de Preparación de los Datos](./Entrega2/3_Preparacion_de_los_Datos.md)**](_blank)

### Fase 4: Modelado (Modeling)
**Enfoque:** Se seleccionarán y aplicarán diversas técnicas de modelado (SVM, Random Forest, XGBoost). Calibraremos los hiperparámetros de los modelos para optimizar su rendimiento y se evaluarán técnicamente. El proceso será iterativo, volviendo a la fase de preparación de datos si es necesario.
*   **Estado:** Pendiente (Parte de la **Entrega 2**).
*   **Documento Detallado:** [**Ver Documento de Modelado](./Entrega2/4_Modelado.md)**](_blank)

### Fase 5: Evaluación (Evaluation)
**Enfoque:** Se evaluarán los modelos desde la perspectiva de los objetivos definidos en la Fase 1. Se medirán los resultados con métricas de rendimiento (precisión, recall, F1-Score) y se compararán con los criterios de éxito. El objetivo es determinar si los modelos responden satisfactoriamente a las preguntas de interés y si la solución es robusta.
*   **Estado:** Pendiente (Parte de la **Entrega 3**).
*   **Documento Detallado:** [**Ver Documento de Evaluación](./Entrega3/5_Evaluacion.md)**](_blank)

### Fase 6: Despliegue (Deployment)
**Enfoque:** En la fase final, se organizarán y presentarán los resultados. Esto incluye el desarrollo de una interfaz gráfica simple para la visualización en tiempo real, la redacción del informe final, y la creación de un video corto que demuestre el funcionamiento y los logros del proyecto.
*   **Estado:** Pendiente (Parte de la **Entrega 3**).
*   **Documento Detallado:** [**Ver Documento de Despliegue](./Entrega3/6_Despliegue.md)**](_blank)