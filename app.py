"""
TFG - Ingeniería de la Salud
Prototipo de interfaz web: sube una imagen histopatológica y el modelo
MobileNetV2 (ya entrenado por mobileNet.py) predice el subtipo de cáncer
de ovario.

Ejecutar con:  streamlit run app.py
Requiere que "mobilenetv2_final.pth" (generado por mobileNet.py, Paso 14)
esté en la misma carpeta que este script.
"""

import base64
import csv
import datetime
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Mismo orden que usó ImageFolder al entrenar (alfabético por nombre de carpeta)
CLASS_NAMES = ["Clear_Cell", "Endometri", "Mucinous", "Non_Cancerous", "Serous"]
OPCIONES_CLASE_REAL = CLASS_NAMES + ["No lo sé"]
RUTA_PESOS = "mobilenetv2_final.pth"

# Métricas del modelo final (Paso 14 de mobileNet.py, experimento ganador: Con DA)
METRICAS_MODELO = {
    "Accuracy": "0.760",
    "F1-macro": "0.756",
    "Balanced Accuracy": "0.760",
    "AUC-ROC": "0.947",
}

# Galería opcional (ejemplos/Clear_Cell/..., ejemplos/Serous/...); se omite si no existe
CARPETA_EJEMPLOS = Path("ejemplos")

# Registro de imágenes subidas y predicciones, para revisión experta futura
# (ver Conclusiones y Trabajo Futuro). No actualiza el modelo automáticamente.
CARPETA_LOGS = Path("logs")
CARPETA_IMAGENES_LOG = CARPETA_LOGS / "imagenes_subidas"
RUTA_CSV_LOG = CARPETA_LOGS / "predicciones.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocesamiento = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================================================================
# CARGA DEL MODELO (se cachea para no recargarlo en cada interacción)
# =============================================================================
@st.cache_resource
def cargar_modelo():
    """Reconstruye la arquitectura de MobileNetV2 usada en el entrenamiento y le carga los pesos finales."""
    modelo = mobilenet_v2(weights=None)
    num_features = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, len(CLASS_NAMES)),
    )
    modelo.load_state_dict(torch.load(RUTA_PESOS, map_location=DEVICE))
    modelo.to(DEVICE)
    modelo.eval()
    return modelo


def predecir(modelo, imagen: Image.Image):
    """Preprocesa la imagen y devuelve las probabilidades softmax para cada clase."""
    tensor = preprocesamiento(imagen.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        salidas = modelo(tensor)
        probabilidades = torch.softmax(salidas, dim=1).cpu().numpy().flatten()
    return probabilidades


def registrar_prediccion(
    imagen: Image.Image, clase_predicha: str, confianza: float, probabilidades, clase_real: str
):
    """Guarda la imagen y una fila de resultados en el CSV de predicciones; no modifica el modelo."""
    CARPETA_IMAGENES_LOG.mkdir(parents=True, exist_ok=True)

    momento = datetime.datetime.now()
    identificador = uuid.uuid4().hex[:8]
    nombre_archivo = f"{momento:%Y-%m-%d_%H%M%S}_{identificador}.jpg"
    imagen.convert("RGB").save(CARPETA_IMAGENES_LOG / nombre_archivo)

    es_archivo_nuevo = not RUTA_CSV_LOG.exists()
    with open(RUTA_CSV_LOG, mode="a", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.writer(archivo_csv)
        if es_archivo_nuevo:
            escritor.writerow(
                ["fecha_hora", "archivo_imagen", "clase_predicha", "clase_real", "confianza"] + CLASS_NAMES
            )
        escritor.writerow(
            [momento.isoformat(timespec="seconds"), nombre_archivo, clase_predicha, clase_real, f"{confianza:.4f}"]
            + [f"{p:.4f}" for p in probabilidades]
        )


def miniatura_data_uri(imagen: Image.Image, tamano: int = 64) -> str:
    """Convierte una imagen en una miniatura codificada en base64, para incrustarla en la tabla resumen."""
    copia = imagen.convert("RGB").copy()
    copia.thumbnail((tamano, tamano))
    buffer = BytesIO()
    copia.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def analizar_imagen(nombre: str, imagen: Image.Image, clase_real: str) -> dict:
    """Ejecuta la predicción sobre una imagen y registra el resultado. Devuelve un resumen para la tabla."""
    modelo = cargar_modelo()
    probabilidades = predecir(modelo, imagen)

    indice_predicho = int(np.argmax(probabilidades))
    clase_predicha = CLASS_NAMES[indice_predicho]
    confianza = float(probabilidades[indice_predicho])

    registrar_prediccion(imagen, clase_predicha, confianza, probabilidades, clase_real)

    if clase_real == "No lo sé":
        acierto = "—"
    elif clase_real == clase_predicha:
        acierto = "✓"
    else:
        acierto = "✗"

    return {
        "nombre": nombre,
        "imagen": imagen,
        "clase_predicha": clase_predicha,
        "clase_real": clase_real,
        "acierto": acierto,
        "confianza": confianza,
        "probabilidades": probabilidades,
    }


def mostrar_resultados(imagenes: list):
    """Muestra una tabla resumen (con casilla para quitar, miniatura, predicción, clase real,
    confianza) y el desglose por imagen."""
    resultados = [
        {**analizar_imagen(img["nombre"], img["imagen"], img["clase_real"]), "origen": img["origen"], "id": img["id"]}
        for img in imagenes
    ]

    st.subheader(f"Resultados ({len(resultados)} imagen{'es' if len(resultados) != 1 else ''})")
    tabla = pd.DataFrame([
        {
            "Quitar": False,
            "Imagen": miniatura_data_uri(r["imagen"]),
            "Archivo": r["nombre"],
            "Predicción": r["clase_predicha"],
            "Clase Real": r["clase_real"],
            "¿Acierto?": r["acierto"],
            "Confianza": r["confianza"] * 100,
        }
        for r in resultados
    ])
    tabla_editada = st.data_editor(
        tabla,
        column_config={
            "Quitar": st.column_config.CheckboxColumn("Quitar", help="Márcala para quitar esta imagen de la lista"),
            "Imagen": st.column_config.ImageColumn("Imagen"),
            "Confianza": st.column_config.ProgressColumn("Confianza", format="%.1f%%", min_value=0, max_value=100),
        },
        disabled=["Imagen", "Archivo", "Predicción", "Clase Real", "¿Acierto?", "Confianza"],
        hide_index=True,
        use_container_width=True,
        key="tabla_resultados",
    )

    # Marcar "Quitar" solo selecciona la fila; no se elimina nada hasta pulsar
    # el botón, para poder marcar varias antes de confirmar.
    marcadas_para_quitar = [
        resultado for marcar, resultado in zip(tabla_editada["Quitar"], resultados) if marcar
    ]
    if st.button(
        f"🗑️ Eliminar seleccionadas ({len(marcadas_para_quitar)})",
        disabled=not marcadas_para_quitar,
    ):
        for resultado in marcadas_para_quitar:
            if resultado["origen"] == "ejemplo":
                st.session_state.ejemplos_seleccionados.pop(resultado["id"], None)
            else:
                st.session_state.archivos_subidos_excluidos.add(resultado["id"])
        st.rerun()

    st.caption("Desglose detallado por imagen:")
    for resultado in resultados:
        etiqueta_expander = (
            f"{resultado['nombre']} → Predicción: {resultado['clase_predicha']} ({resultado['confianza']:.1%})"
            f" | Real: {resultado['clase_real']} {resultado['acierto']}"
        )
        with st.expander(etiqueta_expander):
            columna_imagen, columna_resultado = st.columns([1, 1])
            with columna_imagen:
                st.image(resultado["imagen"], caption=resultado["nombre"], use_container_width=True)
            with columna_resultado:
                for nombre_clase, probabilidad in sorted(
                    zip(CLASS_NAMES, resultado["probabilidades"]), key=lambda item: item[1], reverse=True
                ):
                    etiqueta = f"**{nombre_clase}**" if nombre_clase == resultado["clase_predicha"] else nombre_clase
                    st.progress(float(probabilidad), text=f"{etiqueta} — {probabilidad:.1%}")


# =============================================================================
# BARRA LATERAL: INFORMACIÓN DEL MODELO
# =============================================================================
st.set_page_config(page_title="Clasificador de Cáncer de Ovario", page_icon="🔬", layout="wide")

with st.sidebar:
    st.header("Sobre el modelo")
    st.markdown(
        "**Arquitectura:** MobileNetV2 (Transfer Learning + Fine-Tuning del último bloque)\n\n"
        "**Dataset:** ~500 imágenes histopatológicas, 5 subtipos de cáncer de ovario\n\n"
        "**Entrenamiento:** 5-Fold CV estratificado + Optuna (búsqueda de hiperparámetros) "
        "+ evaluación final sobre un conjunto de test independiente (15%)"
    )
    st.subheader("Métricas en test independiente")
    for nombre_metrica, valor in METRICAS_MODELO.items():
        st.metric(nombre_metrica, valor)
    st.divider()
    st.caption(
        "⚠️ Prototipo académico (TFG). No es una herramienta de diagnóstico clínico "
        "ni ha sido validada para uso médico real."
    )


# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
st.title("Clasificador de subtipos de cáncer de ovario")
st.write(
    "Sube una o varias imágenes histopatológicas y el modelo predice a cuál de los 5 "
    "subtipos pertenece cada una: Clear Cell, Endometrioide, Mucinoso, No Canceroso o Seroso."
)

archivos_subidos = st.file_uploader(
    "Sube una o varias imágenes (JPG o PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Nombres de archivos subidos que el usuario ha quitado desde la tabla de
# resultados (ver mostrar_resultados). El propio widget de subida no permite
# quitar un archivo por código, así que se filtra aquí en cada ejecución.
if "archivos_subidos_excluidos" not in st.session_state:
    st.session_state.archivos_subidos_excluidos = set()

imagenes_a_analizar = []
if archivos_subidos:
    st.caption("Si conoces el subtipo real de cada imagen, indícalo para comparar con la predicción:")
    for archivo in archivos_subidos:
        if archivo.name in st.session_state.archivos_subidos_excluidos:
            continue
        clase_real = st.selectbox(
            f"Clase real — {archivo.name}",
            OPCIONES_CLASE_REAL,
            index=len(OPCIONES_CLASE_REAL) - 1,  # "No lo sé" por defecto
            key=f"clase_real_{archivo.name}",
        )
        imagenes_a_analizar.append({
            "nombre": archivo.name,
            "imagen": Image.open(archivo),
            "clase_real": clase_real,
            "origen": "subida",
            "id": archivo.name,
        })

# Galería opcional de ejemplos (solo si existe la carpeta "ejemplos/"). Las
# selecciones se guardan en session_state: al pulsar un botón, Streamlit
# vuelve a ejecutar todo el script desde el principio, así que sin esto solo
# sobrevivía la del último clic y las anteriores se perdían.
if "ejemplos_seleccionados" not in st.session_state:
    st.session_state.ejemplos_seleccionados = {}  # ruta (str) -> clase elegida

if CARPETA_EJEMPLOS.is_dir():
    with st.expander("O prueba con una imagen de ejemplo"):
        clase_elegida = st.selectbox("Subtipo de ejemplo", CLASS_NAMES)
        carpeta_clase = CARPETA_EJEMPLOS / clase_elegida
        rutas_ejemplo = sorted(carpeta_clase.glob("*.jpg")) + sorted(carpeta_clase.glob("*.png")) if carpeta_clase.is_dir() else []
        if rutas_ejemplo:
            columnas = st.columns(min(len(rutas_ejemplo), 5))
            for columna, ruta_ejemplo in zip(columnas, rutas_ejemplo):
                with columna:
                    st.image(str(ruta_ejemplo), use_container_width=True)
                    clave_ejemplo = str(ruta_ejemplo)
                    ya_seleccionada = clave_ejemplo in st.session_state.ejemplos_seleccionados
                    if st.button("Quitar" if ya_seleccionada else "Usar esta", key=clave_ejemplo):
                        if ya_seleccionada:
                            del st.session_state.ejemplos_seleccionados[clave_ejemplo]
                        else:
                            st.session_state.ejemplos_seleccionados[clave_ejemplo] = clase_elegida
                        st.rerun()
        else:
            st.caption("No hay imágenes de ejemplo para este subtipo todavía.")

        if st.session_state.ejemplos_seleccionados:
            st.caption(f"Ejemplos seleccionados: {len(st.session_state.ejemplos_seleccionados)}")

for ruta_str, clase in st.session_state.ejemplos_seleccionados.items():
    ruta_ejemplo = Path(ruta_str)
    imagenes_a_analizar.append({
        "nombre": ruta_ejemplo.name,
        "imagen": Image.open(ruta_ejemplo),
        "clase_real": clase,
        "origen": "ejemplo",
        "id": ruta_str,
    })

if imagenes_a_analizar:
    mostrar_resultados(imagenes_a_analizar)
else:
    st.info("Sube una o varias imágenes (o elige un ejemplo) para obtener una predicción.")
