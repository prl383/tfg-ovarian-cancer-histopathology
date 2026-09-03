"""
TFG - Ingeniería de la Salud
Clasificación multiclase de subtipos de cáncer de ovario a partir de
imágenes histopatológicas mediante Transfer Learning con ConvNeXt-Tiny.

Tercer modelo del TFG (tras MobileNetV2 y ResNet50): mismo pipeline exacto,
adaptado solo en lo específico de la arquitectura (crear_modelo() y la
capa objetivo de Grad-CAM). Ver mobileNet.py para comentarios más
detallados de cada paso.

Autor: Patricia Rodríguez Lidueña
Entorno: servidor GPU dedicado
Dependencias no preinstaladas: pip install opendatasets optuna
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import csv
import copy
import time
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: funciona igual por SSH y en local
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.convnext import LayerNorm2d

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
import opendatasets as od


# =============================================================================
# PASO 0: CONFIGURACIÓN GENERAL Y SEMILLA DE REPRODUCIBILIDAD
# =============================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# cuDNN determinista: mismo resultado en cada ejecución (algo más lento)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Carpeta propia (no "resultados/" a secas) para no sobrescribir la de otros modelos
CARPETA_RESULTADOS = "resultados_convnext"
CARPETA_GRAFICAS = os.path.join(CARPETA_RESULTADOS, "graficas")
CARPETA_MODELOS = os.path.join(CARPETA_RESULTADOS, "modelos")
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)


class RegistradorSalida:
    """Duplica todo lo impreso por consola también hacia un archivo de log."""

    def __init__(self, ruta_log):
        self.terminal = sys.stdout
        self.archivo = open(ruta_log, "w", encoding="utf-8")

    def write(self, mensaje):
        self.terminal.write(mensaje)
        self.archivo.write(mensaje)

    def flush(self):
        self.terminal.flush()
        self.archivo.flush()


RUTA_LOG = os.path.join(CARPETA_RESULTADOS, f"log_{time.strftime('%Y-%m-%d_%H%M%S')}.txt")
sys.stdout = RegistradorSalida(RUTA_LOG)
print(f"Guardando también una copia de esta ejecución en: {RUTA_LOG}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {DEVICE}")

NOMBRE_MODELO = "ConvNeXt-Tiny"  # usado en la tabla comparativa del Paso 16

# Hiperparámetros de referencia (el entrenamiento real usa los que encuentre Optuna en el Paso 12)
NUM_CLASSES = 5
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64     # solo velocidad, no afecta al resultado
NUM_EPOCHS = 50          # tope por fold; Early Stopping cortará antes
LEARNING_RATE = 1e-3
K_FOLDS = 5
EARLY_STOPPING_PATIENCE = 10     # Rafa: con 5 cortaba precipitadamente
EPOCA_MINIMA_EARLY_STOPPING = 5  # no parar antes, evita paradas por ruido inicial
IMG_SIZE = 224
TEST_SIZE = 0.15         # conjunto de TEST independiente, nunca visto en la CV
OPTUNA_N_TRIALS_POR_EXPERIMENTO = 15   # subido de 10 tras detectar inestabilidad entre ejecuciones

# Normalización estándar de ImageNet (requerida por los pesos preentrenados)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# PASO 1: DESCARGA Y PREPARACIÓN DEL DATASET (KAGGLE)
# =============================================================================
# La primera vez pedirá las credenciales de la API de Kaggle:
# https://www.kaggle.com/settings -> "Create New API Token"
DATASET_URL = (
    "https://www.kaggle.com/datasets/bitsnpieces/"
    "ovarian-cancer-and-subtypes-dataset-histopathology"
)

od.download(DATASET_URL)

# opendatasets genera una estructura de carpetas anidada; las imágenes están en:
DATA_DIR = "./ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer"

print("Clases detectadas en el dataset:")
print(sorted(os.listdir(DATA_DIR)))


# =============================================================================
# PASO 2: TRANSFORMACIONES (DATA AUGMENTATION)
# =============================================================================
# Sin cambios de color (Rafa: los subtipos se diferencian mucho por tono, y
# tocarlo podría arruinar la clasificación). RandomResizedCrop combina zoom
# (~20%) y recorte en una sola transformación.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Sin augmentation (validación/test): solo redimensiona y normaliza
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================================================================
# PASO 3: CARGA DEL DATASET CON ImageFolder
# =============================================================================
# Dos instancias del mismo dataset con transformaciones distintas; cada fold
# indexa con Subset sobre la que corresponda, sin duplicar imágenes en disco.
train_dataset_full = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
val_dataset_full = datasets.ImageFolder(root=DATA_DIR, transform=val_transforms)

class_names = train_dataset_full.classes
print(f"Clases (orden usado por ImageFolder): {class_names}")
assert len(class_names) == NUM_CLASSES, (
    f"Se esperaban {NUM_CLASSES} clases, pero se encontraron {len(class_names)}."
)

all_labels = np.array(train_dataset_full.targets)
all_indices = np.arange(len(all_labels))

# Conjunto de TEST independiente: nunca participa en entrenamiento, Early
# Stopping ni selección de hiperparámetros; solo se usa al final.
indices_pool, indices_test = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    stratify=all_labels,
    random_state=SEED,
)
labels_pool = all_labels[indices_pool]

print(
    f"Reparto inicial -> Pool (Train+Val para CV): {len(indices_pool)} imágenes | "
    f"Test independiente: {len(indices_test)} imágenes"
)

subset_test = Subset(val_dataset_full, indices_test)
dataloader_test = DataLoader(subset_test, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)


# =============================================================================
# PASO 4: DEFINICIÓN DEL MODELO (CONVNEXT-TINY + TRANSFER LEARNING)
# =============================================================================
def crear_modelo(num_classes: int, dropout: float = 0.3) -> nn.Module:
    """
    ConvNeXt-Tiny preentrenada en ImageNet con Fine-Tuning parcial: se
    congela todo el backbone salvo el último bloque (CNBlock) de su última
    etapa (features[-1][-1]), que se reentrena junto con la nueva cabeza
    clasificadora. Mismo criterio que en MobileNetV2 y ResNet50.

    "dropout" lo busca Optuna como hiperparámetro más (Paso 12).
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    modelo = convnext_tiny(weights=weights)

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    # features[-1] es la última etapa (3 CNBlock); se descongela solo el último,
    # igual de restrictivo que descongelar solo el último Bottleneck en ResNet50
    for parametro in modelo.features[-1][-1].parameters():
        parametro.requires_grad = True

    # Cabeza original: LayerNorm2d, Flatten, Linear(768 -> 1000);
    # se mantiene la normalización y el aplanado, con Dropout añadido
    num_features = modelo.classifier[2].in_features
    modelo.classifier = nn.Sequential(
        LayerNorm2d(num_features, eps=1e-6),
        nn.Flatten(1),
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes),
    )

    return modelo.to(DEVICE)


def crear_optimizador(modelo: nn.Module, hparams: dict):
    """Construye el optimizador (adam/adamw/sgd) sobre los parámetros entrenables."""
    parametros_entrenables = filter(lambda p: p.requires_grad, modelo.parameters())
    nombre = hparams["optimizer_name"]

    if nombre == "adam":
        return optim.Adam(parametros_entrenables, lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    if nombre == "adamw":
        return optim.AdamW(parametros_entrenables, lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    if nombre == "sgd":
        return optim.SGD(
            parametros_entrenables, lr=hparams["lr"], weight_decay=hparams["weight_decay"], momentum=0.9
        )
    raise ValueError(f"Optimizador no reconocido: {nombre}")


# =============================================================================
# PASO 5: EARLY STOPPING
# =============================================================================
class EarlyStopping:
    """Detiene el entrenamiento si la Validation Loss no mejora tras 'patience' épocas."""

    def __init__(self, patience: int = 5, delta: float = 0.0, epoca_minima: int = 1):
        self.patience = patience
        self.delta = delta
        self.epoca_minima = epoca_minima  # no se permite parar antes de esta época
        self.contador = 0
        self.mejor_loss = None
        self.early_stop = False
        self.mejores_pesos = None
        self.mejor_epoca = None

    def __call__(self, val_loss: float, modelo: nn.Module, epoca: int):
        if self.mejor_loss is None or val_loss < self.mejor_loss - self.delta:
            self.mejor_loss = val_loss
            self.mejores_pesos = copy.deepcopy(modelo.state_dict())
            self.mejor_epoca = epoca
            self.contador = 0
        else:
            self.contador += 1
            if self.contador >= self.patience and epoca >= self.epoca_minima:
                self.early_stop = True

    def cargar_mejores_pesos(self, modelo: nn.Module):
        if self.mejores_pesos is not None:
            modelo.load_state_dict(self.mejores_pesos)
        return modelo


# =============================================================================
# PASO 6: FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN DE UNA ÉPOCA
# =============================================================================
def entrenar_una_epoca(modelo, dataloader, criterio, optimizador):
    """Ejecuta una época completa de entrenamiento y devuelve (loss, accuracy)."""
    modelo.train()
    perdida_total = 0.0
    aciertos = 0
    total_muestras = 0

    for imagenes, etiquetas in dataloader:
        imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)

        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)
        _, predicciones = torch.max(salidas, dim=1)
        aciertos += torch.sum(predicciones == etiquetas.data).item()
        total_muestras += imagenes.size(0)

    loss_medio = perdida_total / total_muestras
    accuracy_media = aciertos / total_muestras
    return loss_medio, accuracy_media


def evaluar_una_epoca(modelo, dataloader, criterio):
    """Ejecuta una época de validación (sin backpropagation) y devuelve (loss, accuracy)."""
    modelo.eval()
    perdida_total = 0.0
    aciertos = 0
    total_muestras = 0

    with torch.no_grad():
        for imagenes, etiquetas in dataloader:
            imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)

            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)

            perdida_total += perdida.item() * imagenes.size(0)
            _, predicciones = torch.max(salidas, dim=1)
            aciertos += torch.sum(predicciones == etiquetas.data).item()
            total_muestras += imagenes.size(0)

    loss_medio = perdida_total / total_muestras
    accuracy_media = aciertos / total_muestras
    return loss_medio, accuracy_media


# =============================================================================
# PASO 7: OBTENCIÓN DE PREDICCIONES Y MÉTRICAS
# =============================================================================
def obtener_predicciones(modelo, dataloader):
    """
    Recorre el dataloader y devuelve las etiquetas reales, las predicciones
    (clase con mayor probabilidad) y las probabilidades softmax de cada
    clase, necesarias para calcular el AUC-ROC (OVR).
    """
    modelo.eval()
    etiquetas_reales = []
    predicciones_finales = []
    probabilidades_finales = []

    with torch.no_grad():
        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(DEVICE)
            salidas = modelo(imagenes)
            probabilidades = torch.softmax(salidas, dim=1)
            _, predicciones = torch.max(salidas, dim=1)

            etiquetas_reales.extend(etiquetas.numpy())
            predicciones_finales.extend(predicciones.cpu().numpy())
            probabilidades_finales.extend(probabilidades.cpu().numpy())

    return (
        np.array(etiquetas_reales),
        np.array(predicciones_finales),
        np.array(probabilidades_finales),
    )


def calcular_metricas(y_real, y_pred, y_proba):
    """
    Accuracy, Precision/Recall/F1 (macro), Balanced Accuracy y AUC-ROC
    One-vs-Rest. "macro" pondera las 5 clases por igual, para no ocultar
    el rendimiento en las minoritarias.
    """
    accuracy = accuracy_score(y_real, y_pred)
    precision_macro = precision_score(y_real, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_real, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_real, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_real, y_pred)

    y_real_binarizado = label_binarize(y_real, classes=list(range(NUM_CLASSES)))
    auc_roc = roc_auc_score(y_real_binarizado, y_proba, average="macro", multi_class="ovr")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "balanced_accuracy": balanced_acc,
        "auc_roc_ovr": auc_roc,
    }


NOMBRES_METRICAS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "auc_roc_ovr"]


def imprimir_resumen_metricas(titulo: str, resultados: dict):
    """Imprime la media ± desviación típica de cada métrica a lo largo de los folds."""
    print(f"\n-- {titulo} --")
    for nombre_metrica, valores in resultados.items():
        media = np.mean(valores)
        desviacion = np.std(valores)
        print(f"{nombre_metrica:20s}: {media:.4f} ± {desviacion:.4f}")


# =============================================================================
# PASO 8: FUNCIONES DE GRAFICADO
# =============================================================================
def graficar_historial(historial: dict, titulo: str, nombre_archivo: str):
    """Genera las gráficas de Loss y Accuracy (train vs validación) y las guarda en CARPETA_GRAFICAS."""
    epocas_range = range(1, len(historial["train_loss"]) + 1)

    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(titulo, fontsize=14)

    ejes[0].plot(epocas_range, historial["train_loss"], label="Train Loss", marker="o")
    ejes[0].plot(epocas_range, historial["val_loss"], label="Validation Loss", marker="o")
    ejes[0].set_title("Evolución de la función de pérdida (Loss)")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Loss")
    ejes[0].legend()
    ejes[0].grid(True, linestyle="--", alpha=0.5)

    ejes[1].plot(epocas_range, historial["train_acc"], label="Train Accuracy", marker="o")
    ejes[1].plot(epocas_range, historial["val_acc"], label="Validation Accuracy", marker="o")
    ejes[1].set_title("Evolución de la precisión (Accuracy)")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Accuracy")
    ejes[1].set_ylim(0.5, 1.0)  # escala fija para comparar entre gráficas
    ejes[1].legend()
    ejes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICAS, f"{nombre_archivo}.png"), dpi=150)
    plt.show()


def graficar_comparacion_da(historial_con_da: dict, historial_sin_da: dict):
    """Compara, en una misma figura, las curvas de Loss y Accuracy con y sin Data Augmentation."""
    fig, ejes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparativa: SIN vs. CON Data Augmentation", fontsize=14)

    for columna, (historial, titulo) in enumerate([
        (historial_sin_da, "SIN Data Augmentation"),
        (historial_con_da, "CON Data Augmentation"),
    ]):
        epocas_range = range(1, len(historial["train_loss"]) + 1)

        ejes[0, columna].plot(epocas_range, historial["train_loss"], label="Train Loss", marker="o")
        ejes[0, columna].plot(epocas_range, historial["val_loss"], label="Validation Loss", marker="o")
        ejes[0, columna].set_title(f"Loss — {titulo}")
        ejes[0, columna].set_xlabel("Época")
        ejes[0, columna].set_ylabel("Loss")
        ejes[0, columna].legend()
        ejes[0, columna].grid(True, linestyle="--", alpha=0.5)

        ejes[1, columna].plot(epocas_range, historial["train_acc"], label="Train Accuracy", marker="o")
        ejes[1, columna].plot(epocas_range, historial["val_acc"], label="Validation Accuracy", marker="o")
        ejes[1, columna].set_title(f"Accuracy — {titulo}")
        ejes[1, columna].set_xlabel("Época")
        ejes[1, columna].set_ylabel("Accuracy")
        ejes[1, columna].set_ylim(0.5, 1.0)  # misma escala en ambas columnas
        ejes[1, columna].legend()
        ejes[1, columna].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICAS, "comparativa_da.png"), dpi=150)
    plt.show()


# =============================================================================
# PASO 9: FUNCIÓN AUXILIAR DE ENTRENAMIENTO CON EARLY STOPPING (UN SOLO SPLIT)
# =============================================================================
def entrenar_con_early_stopping(dataset_entrenamiento, indices_train, indices_val, hparams):
    """
    Entrena un modelo con Early Stopping sobre un único split train/val (sin
    CV). Se usa solo en la comparativa Con/Sin DA del Paso 10, para ilustrar
    el efecto del augmentation, no para seleccionar hiperparámetros.
    """
    torch.manual_seed(SEED)  # mismos hiperparámetros -> mismo resultado exacto
    np.random.seed(SEED)

    subset_train = Subset(dataset_entrenamiento, indices_train)
    subset_val = Subset(val_dataset_full, indices_val)

    dataloader_train = DataLoader(subset_train, batch_size=hparams["batch_size"], shuffle=True, num_workers=0)
    dataloader_val = DataLoader(subset_val, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    modelo = crear_modelo(NUM_CLASSES, dropout=hparams["dropout"])
    criterio = nn.CrossEntropyLoss()
    optimizador = crear_optimizador(modelo, hparams)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, epoca_minima=EPOCA_MINIMA_EARLY_STOPPING)
    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoca in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = entrenar_una_epoca(modelo, dataloader_train, criterio, optimizador)
        val_loss, val_acc = evaluar_una_epoca(modelo, dataloader_val, criterio)

        print(
            f"Época {epoca:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        historial["train_loss"].append(train_loss)
        historial["train_acc"].append(train_acc)
        historial["val_loss"].append(val_loss)
        historial["val_acc"].append(val_acc)

        early_stopping(val_loss, modelo, epoca)
        if early_stopping.early_stop:
            break

    return historial, early_stopping.mejor_loss, early_stopping.mejor_epoca


# =============================================================================
# PASO 10: COMPARATIVA CON DATA AUGMENTATION VS. SIN DATA AUGMENTATION
# =============================================================================
# Mismo modelo entrenado dos veces sobre el mismo split 80/20 del pool, con
# hiperparámetros de referencia fijos, cambiando solo si hay Data Augmentation.
print("\n" + "=" * 70)
print("PASO 10: COMPARATIVA CON DATA AUGMENTATION VS. SIN DATA AUGMENTATION")
print("=" * 70)

indices_train_da, indices_val_da = train_test_split(
    indices_pool, test_size=0.2, stratify=labels_pool, random_state=SEED
)
hparams_referencia = {
    "lr": LEARNING_RATE, "weight_decay": 0.0, "batch_size": BATCH_SIZE, "optimizer_name": "adam", "dropout": 0.3,
}

print("\nEntrenando SIN Data Augmentation...")
historial_sin_da, mejor_loss_sin_da, mejor_epoca_sin_da = entrenar_con_early_stopping(
    val_dataset_full, indices_train_da, indices_val_da, hparams_referencia
)

print("Entrenando CON Data Augmentation...")
historial_con_da, mejor_loss_con_da, mejor_epoca_con_da = entrenar_con_early_stopping(
    train_dataset_full, indices_train_da, indices_val_da, hparams_referencia
)

graficar_comparacion_da(historial_con_da, historial_sin_da)

# Se compara la MEJOR Val Loss (no la última época), ya que tras activarse
# la paciencia las últimas épocas ya son peores que el mejor punto.
gap_train_val_sin_da = historial_sin_da["train_acc"][mejor_epoca_sin_da - 1] - historial_sin_da["val_acc"][mejor_epoca_sin_da - 1]
gap_train_val_con_da = historial_con_da["train_acc"][mejor_epoca_con_da - 1] - historial_con_da["val_acc"][mejor_epoca_con_da - 1]

print(
    f"\nSIN augmentation -> Mejor Val Loss: {mejor_loss_sin_da:.4f} (época {mejor_epoca_sin_da}) | "
    f"Gap Train-Val Accuracy en ese punto: {gap_train_val_sin_da:.4f}"
)
print(
    f"CON augmentation -> Mejor Val Loss: {mejor_loss_con_da:.4f} (época {mejor_epoca_con_da}) | "
    f"Gap Train-Val Accuracy en ese punto: {gap_train_val_con_da:.4f}"
)


# =============================================================================
# PASO 11: VALIDACIÓN CRUZADA ESTRATIFICADA (FUNCIÓN REUTILIZABLE)
# =============================================================================
def ejecutar_5fold_cv(
    hparams: dict, dataset_entrenamiento, prefijo: str = "cv",
    mostrar_graficas=True, mostrar_prints=True, guardar_pesos=False,
):
    """
    Ejecuta la validación cruzada de K_FOLDS folds sobre el pool de
    train+val con los hiperparámetros indicados, y devuelve las métricas
    de validación y test (test solo a título informativo) y las épocas
    óptimas de cada fold.

    mostrar_graficas/mostrar_prints se desactivan durante la búsqueda de
    Optuna y se activan en la ejecución final con los mejores hiperparámetros.
    """
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    resultados_val = {nombre: [] for nombre in NOMBRES_METRICAS}
    resultados_test = {nombre: [] for nombre in NOMBRES_METRICAS}
    epocas_optimas_por_fold = []

    for numero_fold, (pos_train, pos_val) in enumerate(skf.split(indices_pool, labels_pool), start=1):
        indices_train = indices_pool[pos_train]
        indices_val = indices_pool[pos_val]

        if mostrar_prints:
            print("\n" + "=" * 70)
            print(f"[{prefijo}] FOLD {numero_fold}/{K_FOLDS}")
            print("=" * 70)

        subset_train = Subset(dataset_entrenamiento, indices_train)
        subset_val = Subset(val_dataset_full, indices_val)

        dataloader_train = DataLoader(subset_train, batch_size=hparams["batch_size"], shuffle=True, num_workers=0)
        dataloader_val = DataLoader(subset_val, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

        # Semilla distinta por fold, pero fija entre ejecuciones
        torch.manual_seed(SEED + numero_fold)
        np.random.seed(SEED + numero_fold)

        modelo = crear_modelo(NUM_CLASSES, dropout=hparams["dropout"])
        criterio = nn.CrossEntropyLoss()
        optimizador = crear_optimizador(modelo, hparams)
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, epoca_minima=EPOCA_MINIMA_EARLY_STOPPING)
        historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        tiempo_inicio_fold = time.time()

        for epoca in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = entrenar_una_epoca(modelo, dataloader_train, criterio, optimizador)
            val_loss, val_acc = evaluar_una_epoca(modelo, dataloader_val, criterio)

            historial["train_loss"].append(train_loss)
            historial["train_acc"].append(train_acc)
            historial["val_loss"].append(val_loss)
            historial["val_acc"].append(val_acc)

            if mostrar_prints:
                print(
                    f"Época {epoca:02d}/{NUM_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            early_stopping(val_loss, modelo, epoca)
            if early_stopping.early_stop:
                if mostrar_prints:
                    print(f"Early Stopping activado en la época {epoca} (paciencia={EARLY_STOPPING_PATIENCE}).")
                break

        if mostrar_prints:
            tiempo_fold = time.time() - tiempo_inicio_fold
            print(f"Tiempo de entrenamiento del Fold {numero_fold}: {tiempo_fold / 60:.2f} minutos.")

        modelo = early_stopping.cargar_mejores_pesos(modelo)
        epocas_optimas_por_fold.append(early_stopping.mejor_epoca)

        if mostrar_graficas:
            graficar_historial(
                historial, f"[{prefijo}] Fold {numero_fold} - Curvas de entrenamiento", f"{prefijo}_fold_{numero_fold}"
            )

        indice_mejor_epoca = early_stopping.mejor_epoca - 1
        train_loss_final = historial["train_loss"][indice_mejor_epoca]
        train_acc_final = historial["train_acc"][indice_mejor_epoca]

        y_real_val, y_pred_val, y_proba_val = obtener_predicciones(modelo, dataloader_val)
        metricas_val = calcular_metricas(y_real_val, y_pred_val, y_proba_val)

        y_real_test, y_pred_test, y_proba_test = obtener_predicciones(modelo, dataloader_test)
        metricas_test = calcular_metricas(y_real_test, y_pred_test, y_proba_test)

        if mostrar_prints:
            print(f"\n--- Resultados cuantitativos del Fold {numero_fold} (mejor época: {early_stopping.mejor_epoca}) ---")
            print(f"Train      -> Loss: {train_loss_final:.4f} | Accuracy: {train_acc_final:.4f}")
            print(
                f"Validación -> Accuracy: {metricas_val['accuracy']:.4f} | "
                f"Precision: {metricas_val['precision_macro']:.4f} | "
                f"Recall: {metricas_val['recall_macro']:.4f} | "
                f"F1-macro: {metricas_val['f1_macro']:.4f} | "
                f"Balanced Acc: {metricas_val['balanced_accuracy']:.4f} | "
                f"AUC-ROC: {metricas_val['auc_roc_ovr']:.4f}"
            )
            print(
                f"Test       -> Accuracy: {metricas_test['accuracy']:.4f} | "
                f"Precision: {metricas_test['precision_macro']:.4f} | "
                f"Recall: {metricas_test['recall_macro']:.4f} | "
                f"F1-macro: {metricas_test['f1_macro']:.4f} | "
                f"Balanced Acc: {metricas_test['balanced_accuracy']:.4f} | "
                f"AUC-ROC: {metricas_test['auc_roc_ovr']:.4f}"
            )

        for nombre_metrica in resultados_val:
            resultados_val[nombre_metrica].append(metricas_val[nombre_metrica])
            resultados_test[nombre_metrica].append(metricas_test[nombre_metrica])

        if guardar_pesos:
            torch.save(modelo.state_dict(), os.path.join(CARPETA_MODELOS, f"convnext_{prefijo}_fold{numero_fold}.pth"))

    return resultados_val, resultados_test, epocas_optimas_por_fold


# =============================================================================
# PASO 12: EXPERIMENTO COMPLETO (FUNCIÓN REUTILIZABLE)
# =============================================================================
# Optuna -> 5-Fold CV de verificación -> reentrenamiento final sobre todo
# train+val -> evaluación única en test -> matrices de confusión -> métricas
# por clase. Se ejecuta una vez por escenario (Sin DA / Con DA, Paso 13).
def ejecutar_experimento_completo(nombre_experimento: str, dataset_entrenamiento, n_trials: int) -> dict:
    print("\n" + "#" * 70)
    print(f"# EXPERIMENTO: {nombre_experimento}")
    print("#" * 70)

    # Objetivo: F1-macro medio de validación entre los 5 folds. El TEST no
    # interviene en absoluto en esta búsqueda.
    def objetivo_optuna(trial: optuna.Trial) -> float:
        hparams = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "optimizer_name": trial.suggest_categorical("optimizer_name", ["adam", "adamw", "sgd"]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        }
        print(f"\n[{nombre_experimento} | Optuna] Trial {trial.number} — probando: {hparams}")
        resultados_val, _, _ = ejecutar_5fold_cv(
            hparams, dataset_entrenamiento, prefijo=f"{nombre_experimento}_trial{trial.number}",
            mostrar_graficas=False, mostrar_prints=False,
        )
        f1_macro_medio = float(np.mean(resultados_val["f1_macro"]))
        print(f"[{nombre_experimento} | Optuna] Trial {trial.number} -> F1-macro medio (validación): {f1_macro_medio:.4f}")
        return f1_macro_medio

    print(f"\n--- [{nombre_experimento}] Búsqueda de hiperparámetros con Optuna ({n_trials} trials) ---")
    # Semilla fija en el sampler: la búsqueda es reproducible entre ejecuciones
    estudio_optuna = optuna.create_study(
        direction="maximize",
        study_name=f"convnext_{nombre_experimento}",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    estudio_optuna.optimize(objetivo_optuna, n_trials=n_trials)

    mejores_hparams = estudio_optuna.best_params
    print(f"\n[{nombre_experimento}] Mejores hiperparámetros:", mejores_hparams)
    print(f"[{nombre_experimento}] Mejor F1-macro medio (validación, 5-fold): {estudio_optuna.best_value:.4f}")

    # Gráficas propias de Optuna: historial de optimización e importancia de hiperparámetros
    ejes_historial = plot_optimization_history(estudio_optuna)
    ejes_historial.figure.savefig(
        os.path.join(CARPETA_GRAFICAS, f"optuna_historial_{nombre_experimento}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(ejes_historial.figure)

    ejes_importancia = plot_param_importances(estudio_optuna)
    ejes_importancia.figure.savefig(
        os.path.join(CARPETA_GRAFICAS, f"optuna_importancia_{nombre_experimento}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(ejes_importancia.figure)

    # Repite el 5-Fold CV con la configuración ganadora, mostrando gráficas y guardando pesos
    print(f"\n--- [{nombre_experimento}] Ejecución final del 5-Fold CV ---")
    resultados_val, resultados_test, epocas_optimas_por_fold = ejecutar_5fold_cv(
        mejores_hparams, dataset_entrenamiento, prefijo=nombre_experimento,
        mostrar_graficas=True, mostrar_prints=True, guardar_pesos=True,
    )
    imprimir_resumen_metricas(f"[{nombre_experimento}] Validación (media entre los {K_FOLDS} folds)", resultados_val)
    imprimir_resumen_metricas(f"[{nombre_experimento}] Test (referencia, media de los {K_FOLDS} modelos)", resultados_test)

    # Reentrenamiento final sobre TODO train+val, evaluación única en test
    epocas_modelo_final = round(float(np.mean(epocas_optimas_por_fold)))
    print(f"\n--- [{nombre_experimento}] Reentrenamiento del modelo final sobre todo train+val ---")
    print(f"Épocas óptimas por fold: {epocas_optimas_por_fold} -> elegidas para el modelo final: {epocas_modelo_final}")

    dataloader_pool_final = DataLoader(
        Subset(dataset_entrenamiento, indices_pool), batch_size=mejores_hparams["batch_size"],
        shuffle=True, num_workers=0,
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    modelo_final = crear_modelo(NUM_CLASSES, dropout=mejores_hparams["dropout"])
    criterio_final = nn.CrossEntropyLoss()
    optimizador_final = crear_optimizador(modelo_final, mejores_hparams)

    for epoca in range(1, epocas_modelo_final + 1):
        train_loss, train_acc = entrenar_una_epoca(modelo_final, dataloader_pool_final, criterio_final, optimizador_final)
        print(
            f"[{nombre_experimento} | Modelo final] Época {epoca:02d}/{epocas_modelo_final} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

    torch.save(modelo_final.state_dict(), os.path.join(CARPETA_MODELOS, f"convnext_final_{nombre_experimento}.pth"))

    y_real_final, y_pred_final, y_proba_final = obtener_predicciones(modelo_final, dataloader_test)
    metricas_test_final = calcular_metricas(y_real_final, y_pred_final, y_proba_final)

    print(f"\n--- RESULTADO FINAL [{nombre_experimento}] - MODELO ÚNICO EVALUADO EN TEST ---")
    for nombre_metrica, valor in metricas_test_final.items():
        print(f"{nombre_metrica:20s}: {valor:.4f}")

    # Matriz de Confusión sobre TEST
    matriz_confusion = confusion_matrix(y_real_final, y_pred_final)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True
    )
    ax.set_title(f"Matriz de Confusión [{nombre_experimento}] - Test")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICAS, f"matriz_confusion_test_{nombre_experimento}.png"), dpi=150)
    plt.show()

    # Matriz de Confusión también sobre TRAIN, sin augmentation
    dataloader_pool_evaluacion = DataLoader(
        Subset(val_dataset_full, indices_pool), batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0
    )
    y_real_train, y_pred_train, _ = obtener_predicciones(modelo_final, dataloader_pool_evaluacion)
    matriz_confusion_train = confusion_matrix(y_real_train, y_pred_train)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=matriz_confusion_train, display_labels=class_names).plot(
        ax=ax, cmap="Oranges", xticks_rotation=45, colorbar=True
    )
    ax.set_title(f"Matriz de Confusión [{nombre_experimento}] - Train")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICAS, f"matriz_confusion_train_{nombre_experimento}.png"), dpi=150)
    plt.show()

    # Métricas desglosadas por clase
    precision_por_clase, recall_por_clase, f1_por_clase, soporte_por_clase = precision_recall_fscore_support(
        y_real_final, y_pred_final, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    y_real_final_binarizado = label_binarize(y_real_final, classes=list(range(NUM_CLASSES)))
    auc_por_clase = roc_auc_score(y_real_final_binarizado, y_proba_final, average=None, multi_class="ovr")

    print(f"\n--- [{nombre_experimento}] Métricas por clase - Test ---")
    print(f"{'Subtipo':<15} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'AUC-ROC':>10} {'Nº imágenes':>12}")
    for indice_clase, nombre_clase in enumerate(class_names):
        print(
            f"{nombre_clase:<15} {precision_por_clase[indice_clase]:>10.4f} {recall_por_clase[indice_clase]:>10.4f} "
            f"{f1_por_clase[indice_clase]:>10.4f} {auc_por_clase[indice_clase]:>10.4f} {soporte_por_clase[indice_clase]:>12d}"
        )

    # Curvas ROC por clase (One-vs-Rest)
    fig, ax = plt.subplots(figsize=(7, 6))
    for indice_clase, nombre_clase in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_real_final_binarizado[:, indice_clase], y_proba_final[:, indice_clase])
        ax.plot(fpr, tpr, label=f"{nombre_clase} (AUC={auc_por_clase[indice_clase]:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.5)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title(f"Curvas ROC por clase [{nombre_experimento}] - Test (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICAS, f"curvas_roc_{nombre_experimento}.png"), dpi=150)
    plt.show()

    return {
        "nombre_experimento": nombre_experimento,
        "mejores_hparams": mejores_hparams,
        "resultados_val_cv": resultados_val,
        "resultados_test_cv": resultados_test,
        "epocas_modelo_final": epocas_modelo_final,
        "modelo_final": modelo_final,
        "metricas_test_final": metricas_test_final,
        "y_real_final": y_real_final,
        "y_pred_final": y_pred_final,
    }


# =============================================================================
# PASO 13: EJECUTAR LOS DOS EXPERIMENTOS COMPLETOS (SIN DA Y CON DA)
# =============================================================================
resultado_sin_da = ejecutar_experimento_completo("sinDA", val_dataset_full, OPTUNA_N_TRIALS_POR_EXPERIMENTO)
resultado_con_da = ejecutar_experimento_completo("conDA", train_dataset_full, OPTUNA_N_TRIALS_POR_EXPERIMENTO)


# =============================================================================
# PASO 14: COMPARACIÓN FINAL ENTRE EXPERIMENTOS Y MODELO GANADOR
# =============================================================================
print("\n" + "=" * 70)
print("PASO 14: COMPARACIÓN FINAL — SIN DATA AUGMENTATION vs. CON DATA AUGMENTATION")
print("=" * 70)
print(f"{'Métrica':<20} {'Sin DA':>12} {'Con DA':>12}")
for nombre_metrica in NOMBRES_METRICAS:
    print(
        f"{nombre_metrica:<20} "
        f"{resultado_sin_da['metricas_test_final'][nombre_metrica]:>12.4f} "
        f"{resultado_con_da['metricas_test_final'][nombre_metrica]:>12.4f}"
    )

# Ganador: mejor F1-macro en el test independiente (Rafa, para la inferencia final)
resultado_ganador = (
    resultado_con_da
    if resultado_con_da["metricas_test_final"]["f1_macro"] >= resultado_sin_da["metricas_test_final"]["f1_macro"]
    else resultado_sin_da
)
print(f"\nExperimento ganador (mejor F1-macro en test): {resultado_ganador['nombre_experimento']}")

# Copia en la raíz y en CARPETA_MODELOS. Nombre no genérico ("convnext_final.pth"):
# app.py todavía solo carga MobileNetV2, así que este archivo es de referencia/backup.
torch.save(resultado_ganador["modelo_final"].state_dict(), "convnext_final.pth")
torch.save(resultado_ganador["modelo_final"].state_dict(), os.path.join(CARPETA_MODELOS, "convnext_final.pth"))
print(f'Pesos del modelo ganador guardados en "convnext_final.pth" (raíz) y en "{CARPETA_MODELOS}".')

# Resumen numérico en JSON, para no tener que rebuscar en el log de texto
resumen_resultados = {
    "fecha_ejecucion": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dispositivo": str(DEVICE),
    "experimento_ganador": resultado_ganador["nombre_experimento"],
    "experimentos": {
        resultado["nombre_experimento"]: {
            "mejores_hiperparametros": resultado["mejores_hparams"],
            "epocas_modelo_final": resultado["epocas_modelo_final"],
            "metricas_test_final": resultado["metricas_test_final"],
        }
        for resultado in (resultado_sin_da, resultado_con_da)
    },
}
RUTA_RESUMEN_JSON = os.path.join(CARPETA_RESULTADOS, "resumen_resultados.json")
with open(RUTA_RESUMEN_JSON, "w", encoding="utf-8") as archivo_resumen:
    json.dump(resumen_resultados, archivo_resumen, indent=2, ensure_ascii=False)
print(f"Resumen numérico completo guardado en: {RUTA_RESUMEN_JSON}")


# =============================================================================
# PASO 15: EXPLICABILIDAD CON GRAD-CAM
# =============================================================================
# Mapa de calor con los gradientes de la clase predicha respecto a las
# activaciones de features[-1][-1] (la capa entrenable en el Fine-Tuning).
class GradCAM:
    """Implementación mínima de Grad-CAM para una capa convolucional dada."""

    def __init__(self, modelo: nn.Module, capa_objetivo: nn.Module):
        self.modelo = modelo
        self.activaciones = None
        self.gradientes = None
        capa_objetivo.register_forward_hook(self._guardar_activaciones)
        capa_objetivo.register_full_backward_hook(self._guardar_gradientes)

    def _guardar_activaciones(self, modulo, entrada, salida):
        self.activaciones = salida.detach()

    def _guardar_gradientes(self, modulo, grad_entrada, grad_salida):
        self.gradientes = grad_salida[0].detach()

    def generar_mapa(self, tensor_imagen: torch.Tensor, indice_clase: int) -> np.ndarray:
        self.modelo.zero_grad()
        salidas = self.modelo(tensor_imagen)
        salidas[0, indice_clase].backward()

        pesos = self.gradientes.mean(dim=(2, 3), keepdim=True)  # media espacial por canal
        mapa = torch.relu((pesos * self.activaciones).sum(dim=1)).squeeze()
        mapa = mapa - mapa.min()
        mapa = mapa / (mapa.max() + 1e-8)
        return mapa.cpu().numpy()


def superponer_gradcam(imagen_pil: Image.Image, mapa_calor: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Redimensiona el mapa de calor al tamaño de la imagen y lo superpone con un colormap tipo 'jet'."""
    mapa_uint8 = Image.fromarray(np.uint8(mapa_calor * 255)).resize(imagen_pil.size, Image.BILINEAR)
    colormap = plt.get_cmap("jet")
    mapa_color = colormap(np.array(mapa_uint8) / 255.0)[:, :, :3]
    imagen_array = np.array(imagen_pil.convert("RGB")) / 255.0
    superpuesta = (1 - alpha) * imagen_array + alpha * mapa_color
    return np.clip(superpuesta, 0, 1)


print("\n" + "=" * 70)
print(f"PASO 15: EXPLICABILIDAD CON GRAD-CAM (modelo ganador: {resultado_ganador['nombre_experimento']})")
print("=" * 70)

modelo_gradcam = resultado_ganador["modelo_final"]
modelo_gradcam.eval()
gradcam = GradCAM(modelo_gradcam, modelo_gradcam.features[-1][-1])

# Hasta 2 imágenes de test por clase (10 como máximo)
indices_por_clase = {i: [] for i in range(NUM_CLASSES)}
for indice in indices_test:
    etiqueta = int(all_labels[indice])
    if len(indices_por_clase[etiqueta]) < 2:
        indices_por_clase[etiqueta].append(indice)
indices_gradcam = [indice for lista in indices_por_clase.values() for indice in lista]

fig, ejes = plt.subplots(2, len(indices_gradcam), figsize=(3 * len(indices_gradcam), 6.5))
for columna, indice in enumerate(indices_gradcam):
    ruta_imagen, etiqueta_real = val_dataset_full.samples[indice]
    imagen_pil = Image.open(ruta_imagen).convert("RGB")
    tensor_imagen = val_transforms(imagen_pil).unsqueeze(0).to(DEVICE)

    salidas = modelo_gradcam(tensor_imagen)
    indice_predicho = int(torch.argmax(salidas, dim=1).item())
    mapa_calor = gradcam.generar_mapa(tensor_imagen, indice_predicho)

    imagen_redimensionada = imagen_pil.resize((IMG_SIZE, IMG_SIZE))
    superpuesta = superponer_gradcam(imagen_redimensionada, mapa_calor)

    ejes[0, columna].imshow(imagen_redimensionada)
    ejes[0, columna].set_title(f"Real: {class_names[etiqueta_real]}", fontsize=9)
    ejes[0, columna].axis("off")

    ejes[1, columna].imshow(superpuesta)
    ejes[1, columna].set_title(f"Pred: {class_names[indice_predicho]}", fontsize=9)
    ejes[1, columna].axis("off")

fig.suptitle("Grad-CAM: zonas en las que se fija el modelo para decidir", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICAS, "gradcam.png"), dpi=150)
plt.show()
print(f"Grad-CAM generado para {len(indices_gradcam)} imágenes de test (hasta 2 por clase).")


# =============================================================================
# PASO 16: TABLA Y GRÁFICA COMPARATIVA ENTRE MODELOS
# =============================================================================
# Este CSV vive en la raíz (no en resultados/, que se sobrescribe en cada
# ejecución) y va acumulando una fila por cada arquitectura que se pruebe.
# Si se repite la ejecución del mismo modelo, se actualiza su fila.
print("\n" + "=" * 70)
print("PASO 16: TABLA COMPARATIVA ENTRE MODELOS")
print("=" * 70)

RUTA_CSV_COMPARATIVA = "comparativa_modelos.csv"
CAMPOS_CSV_COMPARATIVA = ["modelo", "fecha", "experimento_ganador"] + NOMBRES_METRICAS

filas_existentes = {}
if os.path.exists(RUTA_CSV_COMPARATIVA):
    with open(RUTA_CSV_COMPARATIVA, newline="", encoding="utf-8") as archivo_csv:
        for fila in csv.DictReader(archivo_csv):
            filas_existentes[fila["modelo"]] = fila

filas_existentes[NOMBRE_MODELO] = {
    "modelo": NOMBRE_MODELO,
    "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
    "experimento_ganador": resultado_ganador["nombre_experimento"],
    **{nombre: f"{resultado_ganador['metricas_test_final'][nombre]:.4f}" for nombre in NOMBRES_METRICAS},
}

with open(RUTA_CSV_COMPARATIVA, "w", newline="", encoding="utf-8") as archivo_csv:
    escritor_csv = csv.DictWriter(archivo_csv, fieldnames=CAMPOS_CSV_COMPARATIVA)
    escritor_csv.writeheader()
    escritor_csv.writerows(filas_existentes.values())

# Copia también dentro de resultados/, como foto fija de esta ejecución
shutil.copy(RUTA_CSV_COMPARATIVA, os.path.join(CARPETA_RESULTADOS, "comparativa_modelos.csv"))

print(f"Fila de '{NOMBRE_MODELO}' añadida/actualizada en: {RUTA_CSV_COMPARATIVA}")
print(f"Copia también guardada en: {os.path.join(CARPETA_RESULTADOS, 'comparativa_modelos.csv')}")
print(f"Modelos presentes en la comparativa hasta ahora: {list(filas_existentes.keys())}")

nombres_modelos = list(filas_existentes.keys())
valores_f1 = [float(filas_existentes[modelo]["f1_macro"]) for modelo in nombres_modelos]
valores_auc = [float(filas_existentes[modelo]["auc_roc_ovr"]) for modelo in nombres_modelos]

posiciones = np.arange(len(nombres_modelos))
ancho_barra = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(posiciones - ancho_barra / 2, valores_f1, ancho_barra, label="F1-macro")
ax.bar(posiciones + ancho_barra / 2, valores_auc, ancho_barra, label="AUC-ROC")
ax.set_xticks(posiciones)
ax.set_xticklabels(nombres_modelos)
ax.set_ylim(0, 1)
ax.set_ylabel("Valor de la métrica (Test)")
ax.set_title("Comparativa entre modelos")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICAS, "comparativa_modelos.png"), dpi=150)
plt.show()
