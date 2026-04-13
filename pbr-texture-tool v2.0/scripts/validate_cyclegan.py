from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------
# Rutas del proyecto.
# ------------------------------------------------------------------
# Este script vive en:
#   C:\pbr_tool\scripts\validate_cyclegan.py
# Por eso parents[1] sube a:
#   C:\pbr_tool
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Archivo de pesos final que vamos a validar.
WEIGHTS_PATH = PROJECT_ROOT / "models" / "cyclegan" / "gen_AB_epoca80_FINAL.weights.h5"

# Imagen de prueba para esta fase.
INPUT_IMAGE_PATH = PROJECT_ROOT / "sample_inputs" / "cyclegan_test_input.png"

# Carpeta donde guardaremos los resultados de esta validación.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3_cyclegan_validation"

# Intensidad por defecto confirmada para el modelo.
DEFAULT_INTENSITY = 0.7

# Resolución fija usada en esta prueba inicial.
# OJO: esto es solo para validar carga + inferencia.
# El pipeline final por parches se implementará más adelante.
TEST_SIZE = 256


# ------------------------------------------------------------------
# Capas personalizadas del modelo.
# ------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class AdaIN(tf.keras.layers.Layer):
    """
    Adaptive Instance Normalization.

    Entradas:
    - x: tensor de características
    - intensity: escalar de control del desgaste

    Salida:
    - tensor modulado por la intensidad
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, intensity = inputs

        # Media y varianza espaciales por canal.
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        std = tf.sqrt(var + 1e-5)

        # Normalización estilo InstanceNorm.
        x_norm = (x - mean) / std

        # Convertimos intensity a forma compatible con x:
        # de (batch, 1) a (batch, 1, 1, 1)
        intensity = tf.reshape(tf.cast(intensity, x.dtype), (-1, 1, 1, 1))

        return x_norm * (1.0 + intensity)


@tf.keras.utils.register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    """
    Bloque de autoatención espacial.

    Ayuda a que el modelo relacione zonas alejadas de la textura.
    """
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

        # Proyecciones Q, K, V
        self.m_q = tf.keras.layers.Conv2D(channels // 8, 1)
        self.m_k = tf.keras.layers.Conv2D(channels // 8, 1)
        self.m_v = tf.keras.layers.Conv2D(channels, 1)

        # Escalar entrenable que regula cuánto pesa la atención.
        self.gamma = self.add_weight(
            name="gamma",
            shape=[1],
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        f = self.m_q(x)
        g = self.m_k(x)
        h = self.m_v(x)

        b = tf.shape(x)[0]

        # Reorganizamos los tensores para construir la matriz de atención.
        s = tf.matmul(
            tf.reshape(g, [b, -1, g.shape[-1]]),
            tf.reshape(f, [b, -1, f.shape[-1]]),
            transpose_b=True,
        )

        beta = tf.nn.softmax(s, axis=-1)

        o = tf.reshape(
            tf.matmul(beta, tf.reshape(h, [b, -1, h.shape[-1]])),
            tf.shape(x),
        )

        return x + self.gamma * o

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


# ------------------------------------------------------------------
# Reconstrucción exacta del generador.
# ------------------------------------------------------------------
def crear_generador() -> tf.keras.Model:
    """
    Reconstruye el generador AB de la CycleGAN.

    Entrada:
    - tensor de 7 canales (256, 256, 7)
      [RGB(3) + Normal(3) + Roughness(1)]
    - escalar de intensidad

    Salida:
    - tensor de 7 canales
      [Albedo(3) + Normal(3) + Roughness(1)]
    """
    img_in = tf.keras.layers.Input(shape=(256, 256, 7), name="input_image")
    int_in = tf.keras.layers.Input(shape=(1,), name="intensity")

    x = tf.keras.layers.Conv2D(64, 7, padding="same")(img_in)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    for f in [128, 256]:
        x = tf.keras.layers.Conv2D(f, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.GroupNormalization(groups=-1)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    for i in range(9):
        res = x

        x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
        x = AdaIN()([x, int_in])
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        if i == 4:
            x = SelfAttention(256)(x)

        x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
        x = AdaIN()([x, int_in])
        x = tf.keras.layers.Add()([res, x])

    for f in [128, 64]:
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same")(x)
        x = tf.keras.layers.GroupNormalization(groups=-1)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    albedo = tf.keras.layers.Conv2D(3, 7, padding="same", activation="sigmoid")(x)
    norm_raw = tf.keras.layers.Conv2D(3, 7, padding="same", activation="tanh")(x)

    # Normalizamos el vector normal por píxel para que tenga norma 1.
    normal = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1),
        name="l2_normalize_normal",
    )(norm_raw)

    roughness = tf.keras.layers.Conv2D(1, 7, padding="same", activation="sigmoid")(x)

    out = tf.keras.layers.Concatenate()([albedo, normal, roughness])

    return tf.keras.Model([img_in, int_in], out, name="CycleGAN_Generador_AB")


# ------------------------------------------------------------------
# Utilidades de imagen.
# ------------------------------------------------------------------
def cargar_imagen_rgb(path: Path) -> np.ndarray:
    """
    Lee una imagen desde disco y la devuelve en RGB.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def preparar_entrada_7_canales(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB en una entrada 7ch válida para la CycleGAN.

    IMPORTANTE:
    En esta prueba de humo:
    - usamos la imagen RGB real,
    - añadimos un normal map neutro,
    - añadimos roughness neutra,
    - y normalizamos todo a [-1, 1].

    Esto sigue la lógica del script de integración que ya tenéis.
    """
    img_res = cv2.resize(img_rgb, (TEST_SIZE, TEST_SIZE), interpolation=cv2.INTER_AREA)

    # Normal map neutro tipo OpenGL:
    # X ~ 128, Y ~ 128, Z ~ 255
    normal_neutra = np.full((TEST_SIZE, TEST_SIZE, 3), 128, dtype=np.uint8)
    normal_neutra[..., 2] = 255

    # Roughness neutra al nivel medio.
    roughness_neutra = np.full((TEST_SIZE, TEST_SIZE, 1), 128, dtype=np.uint8)

    pbr_in = np.concatenate([img_res, normal_neutra, roughness_neutra], axis=-1).astype(np.float32)

    # Normalización a [-1, 1]
    pbr_in = (pbr_in / 127.5) - 1.0

    return pbr_in


def guardar_resultados(pred: np.ndarray, output_dir: Path) -> None:
    """
    Convierte la salida del generador a imágenes PNG y las guarda.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # El script de integración actual remapea así la salida.
    pred_vis = (pred * 0.5 + 0.5).clip(0.0, 1.0)

    albedo = (pred_vis[..., :3] * 255).astype(np.uint8)
    normal = (pred_vis[..., 3:6] * 255).astype(np.uint8)
    roughness = (pred_vis[..., 6] * 255).astype(np.uint8)

    cv2.imwrite(str(output_dir / "cyclegan_albedo.png"), cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "cyclegan_normal.png"), cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "cyclegan_roughness.png"), roughness)


def main() -> None:
    """
    Flujo principal:
    1) comprobar archivos,
    2) reconstruir modelo,
    3) cargar pesos,
    4) preparar entrada,
    5) inferir,
    6) guardar resultados.
    """
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo de pesos: {WEIGHTS_PATH}")

    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"No existe la imagen de entrada: {INPUT_IMAGE_PATH}")

    print(f"Pesos detectados en: {WEIGHTS_PATH}")
    print(f"Imagen de prueba detectada en: {INPUT_IMAGE_PATH}")

    print("Reconstruyendo el generador...")
    model = crear_generador()

    print("Cargando pesos...")
    model.load_weights(WEIGHTS_PATH)
    print("Pesos cargados correctamente.")

    print("Leyendo imagen de prueba...")
    img_rgb = cargar_imagen_rgb(INPUT_IMAGE_PATH)

    print("Preparando entrada de 7 canales...")
    entrada = preparar_entrada_7_canales(img_rgb)

    print("Ejecutando inferencia...")
    pred = model.predict(
        [np.expand_dims(entrada, 0), np.array([[DEFAULT_INTENSITY]], dtype=np.float32)],
        verbose=0,
    )[0]

    print("Guardando resultados...")
    guardar_resultados(pred, OUTPUT_DIR)

    print(f"Validación completada. Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
