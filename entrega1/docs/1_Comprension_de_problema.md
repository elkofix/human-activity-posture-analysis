# Fase 1: Comprensión del Problema

**Documento correspondiente a la Fase 1 de la metodología CRISP-DM.**

Este documento establece el marco estratégico del proyecto. Define el problema a resolver, las preguntas de interés, los objetivos a alcanzar, las métricas para medir el éxito y el contexto ético en el que se desarrollará la solución.

## 1. Definición del Problema
Desarrollar, en un periodo de 5 semanas, una herramienta de software capaz de detectar y clasificar automáticamente cinco actividades humanas específicas (caminar hacia la cámara, caminar de regreso, girar, sentarse y ponerse de pie) a partir de video en tiempo real, mediante el análisis de poses y seguimiento de articulaciones, alcanzando una precisión mínima del 85% en la clasificación de las actividades para validar su uso potencial en contextos de fisioterapia, deporte y ergonomía.

*   **Tipo de Problema:** Desde una perspectiva de Machine Learning, este es un problema de **clasificación multiclase supervisada de series temporales multivariadas**.
    *   **Clasificación Multiclase Supervisada:** Entrenaremos un modelo con datos previamente etiquetados (videos con la actividad correcta) para que aprenda a asignar una categoría  entre varias a datos nuevos.
    *   **Series Temporales Multivariadas:** La entrada del modelo no será una imagen estática, sino una secuencia de datos a lo largo del tiempo (los frames del video). Es multivariada porque en cada instante de tiempo tenemos múltiples valores (las coordenadas x, y, z de todas las articulaciones).

## 2. Preguntas de Interés
Las siguientes preguntas de interés guiarán nuestra investigación y desarrollo:

1.  **Pregunta Central:** ¿Es factible clasificar con alta precisión un conjunto de actividades humanas y cuantificar métricas posturales en tiempo real, utilizando únicamente los datos de landmarks corporales extraídos de un video?
2.  **Pregunta de Modelado:** ¿En 4 semanas es posible identificar cuál de los modelos de clasificación supervisada (SVM, Random Forest, XGBoost) es el mejor modelo para la tarea de detección de actividades humanas, en términos de: ?
3.  **Pregunta de Características:** Para un conjunto de datos de video recopilado durante 5 días, ¿cuál es el ranking de importancia de las características cinemáticas (ángulos de rodilla/cadera, velocidad angular del torso, y distancia vertical cadera-tobillo) extraídas con MediaPipe, medido por su capacidad para que un modelo de clasificación simple (como un Árbol de Decisión) alcance la máxima precisión al diferenciar entre 'caminar' (hacia adelante o atrás), 'girar', 'sentarse' y 'ponerse de pie' al final de una semana de trabajo?

## 4. Métricas de Progreso
Para medir el rendimiento de nuestros modelos y el éxito del proyecto, utilizaremos las siguientes métricas:

*   **Métricas de Clasificación Primarias:**
    *   **Matriz de Confusión:** Para visualizar en qué actividades específicas el modelo está fallando.
    *   **Precisión de cada(Precision):** De todas las veces que el modelo predijo una actividad, ¿cuántas veces acertó?
    *   **Exhaustividad de cada clase (Recall):** De todas las veces que una actividad ocurrió realmente, ¿cuántas veces el modelo la detectó?

*   **Métrica de Rendimiento Técnico:**
    *   **Latencia de Inferencia (o FPS):** Mediremos el tiempo que tarda el sistema en procesar un frame y mostrar el resultado. El objetivo es mantener una tasa de frames por segundo (FPS) que permita una experiencia fluida en tiempo real.

*   **Criterio de Éxito:** Se considerará exitoso el proyecto si el modelo final alcanza un **F1-Score promedio superior a 0.85** en el conjunto de pruebas.

## 5. Análisis de Aspectos Éticos
La implementación de una solución de IA que analiza personas conlleva responsabilidades éticas. Se han identificado los siguientes aspectos a considerar:

*   **Privacidad y Consentimiento:**
    *   **Problema:** El sistema procesa imágenes de personas. Es crucial garantizar la privacidad de los individuos grabados.
    *   **Mitigación:** Para este proyecto, todos los datos serán recolectados de los miembros del grupo, quienes darán su consentimiento explícito para participar en la investigación. No se utilizarán imágenes de terceros sin permiso, y los datos no se compartirán públicamente, asegurando la confidencialidad y el respeto a la autonomía de los participantes,.

*   **Sesgos (Bias) en el Modelo:**
    *   **Problema:** Si los datos de entrenamiento provienen de un grupo demográfico homogéneo (ej., solo hombres jóvenes de una misma etnia), el modelo podría funcionar mal para otros grupos (mujeres, personas mayores, diferentes tipos de cuerpo).
    *   **Mitigación:** Aunque el alcance del proyecto es limitado, se intentará que los datos recolectados incluyan variaciones en la vestimenta y la forma de realizar los movimientos para promover la generalización. En el informe final se documentará esta limitación, una práctica recomendada para abordar la equidad y la justicia en los sistemas de visión por computadora.

*   **Uso Indebido de la Tecnología:**
    *   **Problema:** Una tecnología como esta podría ser utilizada para la vigilancia o el monitoreo de personas sin su consentimiento.
    *   **Mitigación:** El proyecto se enmarca en un contexto académico y con fines de aprendizaje. El código y los resultados se presentarán de forma transparente, discutiendo tanto sus capacidades como sus limitaciones y potenciales riesgos, fomentando un uso responsable de la tecnología y previniendo su aplicación en contextos perjudiciales.

## 6. Siguientes Pasos
1.  **Recolección de Datos:** Proceder con la captura del conjunto de datos inicial según el protocolo definido.
2.  **Análisis Exploratorio:** Realizar un primer análisis de los videos recolectados para entender su calidad y variabilidad.
3.  **Pre-procesamiento:** Desarrollar los scripts iniciales para extraer los landmarks de los videos usando MediaPipe.


### Referencias
 M. K. Lee, J. T. Biega, A. L. Cunliffe, D. Williams, D. Schmit, y T. K. Lee, “A Contextual Ethics Framework for Human Participant AI Research,” *arXiv preprint arXiv:2311.01254*, 2023.

 S. Sharma y S. Singh, “Ethical Considerations in Artificial Intelligence: A Comprehensive Discussion from the Perspective of Computer Vision,” en *2023 3rd International Conference on Advance Computing and Innovative Technologies in Engineering (ICACITE)*, 2023, pp. 1812–1817. doi: 10.1109/ICACITE57410.2023.10182607.