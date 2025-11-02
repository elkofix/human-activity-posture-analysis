## 1. Metodología de Trabajo: CRISP-DM

El proyecto seguirá la metodología **CRISP-DM (Cross-Industry Standard Process for Data Mining)** porque es el estándar más utilizado y probado para proyectos de minería de datos y aprendizaje automático y también para asegurar un desarrollo estructurado, iterativo y orientado a los objetivos. A continuación, se detalla el enfoque que se tomará en cada una de las fases y se enlaza al documento correspondiente que contendrá los entregables de cada etapa.

### Fase 3: Preparación de los Datos (Data Preparation)
**Enfoque:** Aquí se realizarán todas las actividades para construir el conjunto de datos final que se usará para el modelado. Esto incluye el etiquetado (anotación) de los videos, la extracción de landmarks con MediaPipe, la limpieza de datos (filtrado de ruido), la normalización de coordenadas y la creación de características (feature engineering) como velocidades y ángulos.
*   **Estado:** Terminado (Parte de la **Entrega 2**).
*   **Documento Detallado:** [**Ver Documento de Preparación de los Datos](./docs/3_Preparacion_de_los_datos.md)**](_blank)

### Fase 4: Modelado (Modeling)
**Enfoque:** Se seleccionarán y aplicarán diversas técnicas de modelado (SVM, Random Forest, XGBoost). Calibraremos los hiperparámetros de los modelos para optimizar su rendimiento y se evaluarán técnicamente. El proceso será iterativo, volviendo a la fase de preparación de datos si es necesario.
*   **Estado:** Terminado (Parte de la **Entrega 2**).
*   **Documento Detallado:** [**Ver Documento de Modelado](./docs/4_Modelado.md)**](_blank)