import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------
# CONFIGURACIÓN
# -----------------------------------------

st.set_page_config(
    page_title="Clasificador de Imágenes",
    page_icon="🤖"
)

# -----------------------------------------
# INFORMACIÓN DEL ESTUDIANTE
# -----------------------------------------

NOMBRE = "Marco Argueta"

# -----------------------------------------
# CLASES
# -----------------------------------------

class_names = [
    "Camiseta",
    "Pantalón",
    "Jersey",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Bota"
]

# -----------------------------------------
# CARGAR MODELO
# -----------------------------------------

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_fashion.keras")

model = cargar_modelo()

# -----------------------------------------
# INTERFAZ
# -----------------------------------------

st.title("🤖 Clasificador de Imágenes con IA")

st.subheader("Clasificación mediante Inteligencia Artificial")

st.write(f"**Estudiante:** {NOMBRE}")

st.write(
    "Esta aplicación utiliza una red neuronal convolucional "
    "entrenada con imágenes de Fashion-MNIST."
)

st.divider()

# -----------------------------------------
# SELECCIONAR MÉTODO
# -----------------------------------------

st.subheader("📷 Selecciona una imagen")

opcion = st.radio(
    "¿Cómo deseas proporcionar la imagen?",
    ["Subir una imagen", "Tomar una foto"]
)

imagen = None

# -----------------------------------------
# SUBIR IMAGEN
# -----------------------------------------

if opcion == "Subir una imagen":

    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:
        imagen = Image.open(archivo)

# -----------------------------------------
# TOMAR FOTO
# -----------------------------------------

else:

    archivo = st.camera_input("Toma una fotografía")

    if archivo is not None:
        imagen = Image.open(archivo)

# -----------------------------------------
# REALIZAR PREDICCIÓN
# -----------------------------------------

if imagen is not None:

    st.divider()

    st.subheader("🖼️ Imagen")

    st.image(
        imagen,
        caption="Imagen seleccionada",
        use_container_width=True
    )

    # Convertir a escala de grises
    imagen = imagen.convert("L")

    # Redimensionar a 28x28
    imagen = imagen.resize((28, 28))

    # Convertir a numpy
    imagen_array = np.array(imagen)

    # Normalizar
    imagen_array = imagen_array.astype("float32") / 255.0

    # Agregar dimensiones
    imagen_array = np.expand_dims(imagen_array, axis=0)
    imagen_array = np.expand_dims(imagen_array, axis=-1)

    # Predicción
    prediccion = model.predict(
        imagen_array,
        verbose=0
    )

    indice = np.argmax(prediccion[0])

    clase = class_names[indice]

    confianza = prediccion[0][indice] * 100

    # -----------------------------------------
    # RESULTADO
    # -----------------------------------------

    st.subheader("🔍 Resultado")

    st.success(
        f"Predicción: {clase}"
    )

    st.metric(
        "Confianza",
        f"{confianza:.2f}%"
    )

    # -----------------------------------------
    # TODAS LAS PROBABILIDADES
    # -----------------------------------------

    st.subheader("📊 Probabilidades")

    for i, nombre in enumerate(class_names):

        porcentaje = float(prediccion[0][i] * 100)

        st.write(
            f"**{nombre}:** {porcentaje:.2f}%"
        )

        st.progress(
            min(int(porcentaje), 100)
        )

# -----------------------------------------
# PIE DE PÁGINA
# -----------------------------------------

st.divider()

st.caption(
    f"Proyecto desarrollado por {NOMBRE} "
    "con TensorFlow y Streamlit."
)