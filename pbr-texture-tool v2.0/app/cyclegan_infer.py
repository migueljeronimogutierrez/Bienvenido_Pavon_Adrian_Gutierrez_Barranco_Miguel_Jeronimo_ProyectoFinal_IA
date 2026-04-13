from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from app.config import CYCLEGAN_DEFAULT_INTENSITY, CYCLEGAN_WEIGHTS, MODEL_SIZE


@tf.keras.utils.register_keras_serializable()
class AdaIN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, intensity = inputs

        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        std = tf.sqrt(var + 1e-5)
        x_norm = (x - mean) / std

        intensity = tf.reshape(tf.cast(intensity, x.dtype), (-1, 1, 1, 1))
        return x_norm * (1.0 + intensity)


@tf.keras.utils.register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.m_q = tf.keras.layers.Conv2D(channels // 8, 1)
        self.m_k = tf.keras.layers.Conv2D(channels // 8, 1)
        self.m_v = tf.keras.layers.Conv2D(channels, 1)
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


def create_generator() -> tf.keras.Model:
    img_in = tf.keras.layers.Input(shape=(MODEL_SIZE, MODEL_SIZE, 7), name="input_image")
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
    normal = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1),
        name="l2_normalize_normal",
    )(norm_raw)
    roughness = tf.keras.layers.Conv2D(1, 7, padding="same", activation="sigmoid")(x)

    out = tf.keras.layers.Concatenate()([albedo, normal, roughness])
    return tf.keras.Model([img_in, int_in], out, name="CycleGAN_Generador_AB")


# -----------------------------------------------------------------
# Corrección 1 — Intensidad real mediante blend en post-proceso.
#
# El modelo fue entrenado con intensidad fija 0.7 en AdaIN, por lo
# que pasar valores distintos no produce efecto visual significativo.
# La solución es ignorar el parámetro como entrada al modelo
# (siempre inferimos con CYCLEGAN_DEFAULT_INTENSITY) y usarlo
# después para interpolar linealmente entre la imagen original
# y la salida del modelo:
#
#   resultado = original * (1 - t) + aged * t
#
# Esto da al usuario control real y perceptualmente correcto sobre
# el nivel de envejecimiento visible.
# -----------------------------------------------------------------

def _blend_images(
    original: np.ndarray,
    aged: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """
    Interpola linealmente entre original y aged según intensity.

    Parámetros
    ----------
    original : np.ndarray
        Imagen original en uint8.
    aged : np.ndarray
        Imagen envejecida en uint8.
    intensity : float
        Factor de mezcla en [0.0, 1.0].
        0.0 → original puro. 1.0 → aged puro.

    Devuelve
    --------
    np.ndarray
        Imagen mezclada en uint8, misma forma que las entradas.
    """
    t = float(np.clip(intensity, 0.0, 1.0))
    blended = (1.0 - t) * original.astype(np.float32) + t * aged.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------
# Corrección 2 — Normal map aged: corrección por histogram matching.
#
# La CycleGAN genera normal maps con el canal Z (azul) casi a cero
# y los canales X/Y disparados, produciendo el color rosa visible
# en la interfaz. El normal map base generado por DeepPBR sí es
# correcto (espacio tangente OpenGL: azul dominante).
#
# La corrección transfiere la distribución estadística del normal
# base al normal aged, canal por canal:
#
#   corrected = (aged - mean_aged) / std_aged * std_base + mean_base
#
# Esto preserva toda la variación de detalle añadida por la CycleGAN
# (desgaste, grietas) pero la "ancla" al espacio de color correcto
# del normal base.
# -----------------------------------------------------------------

def _correct_normal_map(
    normal_aged: np.ndarray,
    normal_base: np.ndarray,
) -> np.ndarray:
    """
    Corrige el espacio de color del normal aged mediante histogram
    matching estadístico por canal respecto al normal base.

    Parámetros
    ----------
    normal_aged : np.ndarray
        Normal map aged en uint8 (H, W, 3). Típicamente rosa/incorrecto.
    normal_base : np.ndarray
        Normal map base en uint8 (H, W, 3). Generado por DeepPBR, correcto.

    Devuelve
    --------
    np.ndarray
        Normal map corregido en uint8 (H, W, 3) con la paleta de
        colores del espacio tangente correcto.
    """
    # Trabajamos en float32 para evitar overflow/underflow.
    aged_f = normal_aged.astype(np.float32)
    base_f = normal_base.astype(np.float32)

    corrected = np.empty_like(aged_f)

    for ch in range(3):
        mean_aged = aged_f[..., ch].mean()
        std_aged = aged_f[..., ch].std() + 1e-6  # evitar división por cero

        mean_base = base_f[..., ch].mean()
        std_base = base_f[..., ch].std() + 1e-6

        # Reescalar el canal aged para que tenga la misma distribución
        # que el canal base.
        corrected[..., ch] = (
            (aged_f[..., ch] - mean_aged) / std_aged * std_base + mean_base
        )

    return np.clip(corrected, 0, 255).astype(np.uint8)


class CycleGANInferencer:
    def __init__(
        self,
        weights_path: Path = CYCLEGAN_WEIGHTS,
        default_intensity: float = CYCLEGAN_DEFAULT_INTENSITY,
        crop_top: int = 2,
        crop_left: int = 2,
        crop_right: int = 2,
        crop_bottom: int = 10,
    ):
        self.weights_path = Path(weights_path)
        self.default_intensity = float(default_intensity)
        self.crop_top = int(crop_top)
        self.crop_left = int(crop_left)
        self.crop_right = int(crop_right)
        self.crop_bottom = int(crop_bottom)
        self.model = None

    def load(self) -> None:
        self.model = create_generator()
        self.model.load_weights(self.weights_path)

    @staticmethod
    def preprocess_pbr(
        rgb_uint8: np.ndarray,
        normal_uint8: np.ndarray,
        roughness_uint8: np.ndarray,
    ) -> np.ndarray:
        rgb_res = cv2.resize(rgb_uint8, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
        normal_res = cv2.resize(normal_uint8, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)

        if roughness_uint8.ndim == 2:
            roughness_in = roughness_uint8
        else:
            roughness_in = roughness_uint8.squeeze(-1)

        rough_res = cv2.resize(roughness_in, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
        rough_res = np.expand_dims(rough_res, axis=-1)

        pbr = np.concatenate([rgb_res, normal_res, rough_res], axis=-1).astype(np.float32)
        pbr = (pbr / 127.5) - 1.0
        return pbr

    def apply_asymmetric_crop_and_resize(
        self,
        image: np.ndarray,
        interpolation: int,
    ) -> np.ndarray:
        """
        Recorta de forma asimétrica para atacar mejor el artefacto inferior.
        Luego reescala al tamaño original.
        """
        h, w = image.shape[:2]

        top = max(0, self.crop_top)
        left = max(0, self.crop_left)
        right = max(0, self.crop_right)
        bottom = max(0, self.crop_bottom)

        if top + bottom >= h:
            return image
        if left + right >= w:
            return image

        cropped = image[top:h - bottom, left:w - right]

        if cropped.size == 0:
            return image

        resized = cv2.resize(cropped, (w, h), interpolation=interpolation)
        return resized

    def postprocess_prediction(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_vis = (pred * 0.5 + 0.5).clip(0.0, 1.0)

        albedo = (pred_vis[..., :3] * 255).astype(np.uint8)
        normal = (pred_vis[..., 3:6] * 255).astype(np.uint8)
        roughness = (pred_vis[..., 6] * 255).astype(np.uint8)

        albedo = self.apply_asymmetric_crop_and_resize(
            albedo,
            interpolation=cv2.INTER_CUBIC,
        )
        normal = self.apply_asymmetric_crop_and_resize(
            normal,
            interpolation=cv2.INTER_CUBIC,
        )
        roughness = self.apply_asymmetric_crop_and_resize(
            roughness,
            interpolation=cv2.INTER_CUBIC,
        )

        return albedo, normal, roughness

    # -----------------------------------------------------------------
    # Método legacy: procesa toda la imagen a 256x256.
    # Se mantiene como fallback por si se necesita el modo rápido.
    #
    # CAMBIOS respecto a la versión anterior:
    # 1. Siempre inferimos con CYCLEGAN_DEFAULT_INTENSITY (0.7).
    #    El parámetro intensity del usuario se usa solo para el blend.
    # 2. Se aplica corrección de normal map por histogram matching.
    # 3. Se aplica blend de intensidad sobre los tres canales de salida.
    # -----------------------------------------------------------------

    def predict_from_pbr(
        self,
        rgb_uint8: np.ndarray,
        normal_uint8: np.ndarray,
        roughness_uint8: np.ndarray,
        intensity: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("El modelo CycleGAN no está cargado. Llama antes a load().")

        if intensity is None:
            intensity = self.default_intensity

        # Siempre inferimos con el valor de entrenamiento (0.7).
        # La intensidad del usuario se aplica en post-proceso.
        entrada = self.preprocess_pbr(rgb_uint8, normal_uint8, roughness_uint8)

        pred = self.model.predict(
            [
                np.expand_dims(entrada, 0),
                np.array([[self.default_intensity]], dtype=np.float32),
            ],
            verbose=0,
        )[0]

        albedo_aged, normal_aged, roughness_aged = self.postprocess_prediction(pred)

        # Corrección 2: arreglar el espacio de color del normal aged.
        # normal_uint8 ya está a tamaño original; lo redimensionamos
        # al tamaño de salida del modelo para que coincidan.
        normal_base_resized = cv2.resize(
            normal_uint8,
            (normal_aged.shape[1], normal_aged.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        normal_aged = _correct_normal_map(normal_aged, normal_base_resized)

        # Corrección 1: blend de intensidad en post-proceso.
        # Redimensionamos los originales al tamaño de salida del modelo.
        rgb_resized = cv2.resize(
            rgb_uint8,
            (albedo_aged.shape[1], albedo_aged.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        roughness_orig = roughness_uint8
        if roughness_orig.ndim == 3:
            roughness_orig = roughness_orig.squeeze(-1)
        roughness_resized = cv2.resize(
            roughness_orig,
            (roughness_aged.shape[1], roughness_aged.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        albedo_aged = _blend_images(rgb_resized, albedo_aged, intensity)
        normal_aged = _blend_images(normal_base_resized, normal_aged, intensity)
        roughness_aged = _blend_images(roughness_resized, roughness_aged, intensity)

        return albedo_aged, normal_aged, roughness_aged

    # -----------------------------------------------------------------
    # Método nuevo para patching a escala real.
    #
    # CAMBIOS respecto a la versión anterior:
    # 1. Siempre inferimos con CYCLEGAN_DEFAULT_INTENSITY (0.7).
    # 2. Se aplica corrección de normal map por histogram matching.
    # 3. Se aplica blend de intensidad sobre los tres canales de salida.
    # -----------------------------------------------------------------

    def predict_tile(
        self,
        tile_7ch_uint8: np.ndarray,
        intensity: float | None = None,
    ) -> np.ndarray:
        """
        Ejecuta CycleGAN sobre un parche individual de 7 canales.

        Parámetros
        ----------
        tile_7ch_uint8 : np.ndarray
            Parche uint8 con forma (256, 256, 7).
            Canales: RGB(3) + Normal(3) + Roughness(1).
        intensity : float
            Factor de mezcla visible en [0.0, 1.0].
            La inferencia siempre se realiza con CYCLEGAN_DEFAULT_INTENSITY.

        Devuelve
        --------
        np.ndarray
            Parche uint8 con forma (256, 256, 7).
            Canales: Albedo_aged(3) + Normal_aged(3) + Roughness_aged(1).
        """
        if self.model is None:
            raise RuntimeError("El modelo CycleGAN no está cargado. Llama antes a load().")

        if intensity is None:
            intensity = self.default_intensity

        # Separar los canales originales para el blend posterior.
        rgb_orig = tile_7ch_uint8[..., :3]
        normal_orig = tile_7ch_uint8[..., 3:6]
        roughness_orig = tile_7ch_uint8[..., 6]

        # Normalizar de uint8 [0, 255] a float32 [-1, 1].
        tile_float = tile_7ch_uint8.astype(np.float32)
        tile_norm = (tile_float / 127.5) - 1.0

        # Inferencia siempre con el valor de entrenamiento (0.7).
        pred = self.model.predict(
            [
                np.expand_dims(tile_norm, 0),
                np.array([[self.default_intensity]], dtype=np.float32),
            ],
            verbose=0,
        )[0]

        # Convertir de [-1, 1] a [0, 255] uint8.
        pred_vis = (pred * 0.5 + 0.5).clip(0.0, 1.0)
        tile_out = (pred_vis * 255).astype(np.uint8)

        albedo_aged = tile_out[..., :3]
        normal_aged = tile_out[..., 3:6]
        roughness_aged = tile_out[..., 6]

        # Corrección 2: arreglar espacio de color del normal aged.
        normal_aged = _correct_normal_map(normal_aged, normal_orig)

        # Corrección 1: blend de intensidad en post-proceso.
        albedo_aged = _blend_images(rgb_orig, albedo_aged, intensity)
        normal_aged = _blend_images(normal_orig, normal_aged, intensity)
        roughness_aged = _blend_images(roughness_orig, roughness_aged, intensity)

        # Recomponer el parche de 7 canales.
        roughness_3d = roughness_aged[..., np.newaxis]
        tile_final = np.concatenate([albedo_aged, normal_aged, roughness_3d], axis=-1)

        return tile_final