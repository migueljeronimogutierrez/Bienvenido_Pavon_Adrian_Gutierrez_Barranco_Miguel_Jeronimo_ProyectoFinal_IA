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

        # Evita recortes inválidos.
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

        entrada = self.preprocess_pbr(rgb_uint8, normal_uint8, roughness_uint8)

        pred = self.model.predict(
            [np.expand_dims(entrada, 0), np.array([[float(intensity)]], dtype=np.float32)],
            verbose=0,
        )[0]

        return self.postprocess_prediction(pred)

    # -----------------------------------------------------------------
    # Método nuevo para patching a escala real.
    #
    # A diferencia de predict_from_pbr (que recibe las 3 imágenes
    # por separado, las reduce a 256x256 y aplica crop anti-artefacto),
    # predict_tile recibe directamente un parche de 256x256x7 en uint8,
    # lo normaliza, ejecuta la inferencia y devuelve uint8 de 7 canales.
    #
    # NO se aplica crop asimétrico aquí, porque el blending por
    # ventana de Hann ya de-pondera los bordes de cada parche.
    # Los artefactos de borde quedan naturalmente atenuados al
    # mezclarse con las zonas centrales de los parches vecinos.
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
            Intensidad del envejecimiento (0.0 a 1.0).

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

        # Normalizar de uint8 [0, 255] a float32 [-1, 1].
        tile_float = tile_7ch_uint8.astype(np.float32)
        tile_norm = (tile_float / 127.5) - 1.0

        # Inferencia.
        pred = self.model.predict(
            [
                np.expand_dims(tile_norm, 0),
                np.array([[float(intensity)]], dtype=np.float32),
            ],
            verbose=0,
        )[0]

        # Convertir de [-1, 1] a [0, 255] uint8.
        pred_vis = (pred * 0.5 + 0.5).clip(0.0, 1.0)
        tile_out = (pred_vis * 255).astype(np.uint8)

        return tile_out