# Clasificación multiclase de cáncer de ovario en imágenes histopatológicas mediante redes convolucionales

Trabajo de Fin de Grado — Grado en Ingeniería de la Salud, Universidad de Málaga.

**Autora:** Patricia Rodríguez Lidueña
**Tutores:** Esteban José Palomo Ferrer (tutor), Rafael Marcos Luque Baena (cotutor)

Comparación de cuatro arquitecturas de redes neuronales profundas (MobileNetV2, ResNet50, ConvNeXt-Tiny y Swin-Tiny) mediante Transfer Learning, para la clasificación multiclase de 5 subtipos histológicos de cáncer de ovario a partir de imágenes histopatológicas. Cada arquitectura se entrena y evalúa con el mismo protocolo: validación cruzada estratificada de 5 pliegues + optimización de hiperparámetros con Optuna, reentrenamiento final sobre todo el conjunto de entrenamiento, y una única evaluación sobre un conjunto de test independiente nunca visto durante el proceso. Incluye además explicabilidad mediante Grad-CAM y un prototipo de inferencia web.

## Resultado final

| Arquitectura | Experimento | Accuracy | F1-macro | AUC-ROC |
|---|---|---|---|---|
| MobileNetV2 | Con DA | 76.00% | 0.7565 | 0.9471 |
| ResNet50 | Con DA | 86.67% | 0.8642 | 0.9680 |
| **ConvNeXt-Tiny** | Con DA | **88.00%** | **0.8801** | **0.9693** |
| Swin-Tiny | Sin DA | 81.33% | 0.8138 | 0.9542 |

## Estructura del repositorio

```
mobileNet.py      # Entrenamiento y evaluación de MobileNetV2
resnet50.py        # Entrenamiento y evaluación de ResNet50
convnext.py         # Entrenamiento y evaluación de ConvNeXt-Tiny
swin.py               # Entrenamiento y evaluación de Swin-Tiny
app.py                # Prototipo de inferencia web (Streamlit)
mobilenetv2_final.pth # Pesos del modelo ganador (necesarios para app.py)
ejemplos/            # Imágenes de ejemplo para la galería de app.py
resultados_*/       # Gráficas, logs y métricas de cada arquitectura
```

Los cuatro scripts de entrenamiento siguen exactamente el mismo pipeline; `mobileNet.py` es el más comentado, y los otros tres remiten a él salvo donde su arquitectura se comporta de forma distinta.

## Manual de instalación

**Requisitos previos:** Python 3.10 o superior. El entrenamiento de las cuatro arquitecturas requiere una GPU con soporte CUDA; el prototipo web puede ejecutarse en CPU, ya que solo realiza inferencia sobre una imagen cada vez. Es necesaria además una cuenta de Kaggle con una clave de API propia (`kaggle.json`), disponible en [kaggle.com/settings](https://www.kaggle.com/settings) → *Create New API Token*, ya que el dataset se descarga automáticamente la primera vez que se ejecuta cualquiera de los scripts de entrenamiento.

**Dependencias:**

```bash
pip install torch torchvision optuna opendatasets scikit-learn
pip install matplotlib pandas numpy Pillow streamlit
```

Las cuatro arquitecturas se cargan directamente desde `torchvision.models` con sus pesos preentrenados en ImageNet, por lo que no requieren ninguna descarga ni instalación adicional.

El dataset ([Kasture, K. — Kaggle](https://www.kaggle.com/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology)) no necesita descargarse manualmente: cada script lo descarga automáticamente en su primera ejecución mediante la librería `opendatasets`, solicitando el usuario y la clave de API de Kaggle mencionados arriba.

## Manual de ejecución

**Entrenamiento de cada arquitectura** (cada una genera su propia carpeta `resultados_<nombre>/` con logs, gráficas, pesos de cada fold y del modelo final, y un resumen en JSON):

```bash
python mobileNet.py
python resnet50.py
python convnext.py
python swin.py
```

**Prototipo web** (requiere que `mobilenetv2_final.pth` esté en la misma carpeta que `app.py`):

```bash
streamlit run app.py
```

La interfaz se abre automáticamente en `http://localhost:8501`.
