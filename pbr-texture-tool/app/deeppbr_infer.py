from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from app.config import DEEPPBR_CKPT_PREFIX, MODEL_SIZE


# -------------------------------------------------------------
# Política global de precisión.
# -------------------------------------------------------------
# El entrenamiento original utilizó mixed_float16, así que la
# activamos ANTES de construir el modelo.
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)


def cbam_block(input_feature, ratio=8):
    """
    Bloque CBAM (Convolutional Block Attention Module).

    Hace dos cosas:
    1) atención de canal
    2) atención espacial

    input_feature:
        Tensor 4D con forma aproximada (batch, H, W, C)

    ratio:
        Factor de reducción usado en la atención de canal.

    Devuelve:
        Tensor refinado por atención.
    """
    channels = input_feature.shape[-1]

    shared_layer_one = layers.Dense(
        channels // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
    )
    shared_layer_two = layers.Dense(
        channels,
        kernel_initializer="he_normal",
        use_bias=True,
    )

    # Atención de canal basada en media global.
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
    avg_pool = shared_layer_two(shared_layer_one(avg_pool))

    # Atención de canal basada en máximo global.
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channels))(max_pool)
    max_pool = shared_layer_two(shared_layer_one(max_pool))

    channel_attention = layers.Activation("sigmoid")(layers.Add()([avg_pool, max_pool]))
    cbam_feature = layers.Multiply()([input_feature, channel_attention])

    # Atención espacial: reduce canales a dos mapas descriptivos.
    spatial_avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
    )(cbam_feature)

    spatial_max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
    )(cbam_feature)

    concat = layers.Concatenate(axis=-1)([spatial_avg_pool, spatial_max_pool])

    spatial_attention = layers.Activation("sigmoid")(
        layers.Conv2D(1, (7, 7), padding="same")(concat)
    )

    cbam_feature = layers.Multiply()([cbam_feature, spatial_attention])
    return cbam_feature


def upsample_block(filters, size, apply_dropout=False):
    """
    Bloque de subida de resolución del decoder.

    filters:
        Número de filtros de la Conv2DTranspose.
    size:
        Tamaño del kernel.
    apply_dropout:
        Si True, aplica Dropout(0.5) tras BatchNorm.

    Devuelve:
        tf.keras.Sequential reutilizable.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())
    return result


def build_deep_pbr_generator() -> tf.keras.Model:
    """
    Reconstruye el generador DeepPBR-Net.

    Arquitectura:
    - entrada RGB 256x256x3
    - encoder ResNet50 sin top
    - skip connections con CBAM
    - dos cabezas de salida:
        * normal map (3 canales)
        * roughness (1 canal)

    IMPORTANTE:
    weights=None evita depender de Internet para ImageNet.
    El checkpoint restaurará los pesos entrenados.
    """
    inputs = layers.Input(shape=[MODEL_SIZE, MODEL_SIZE, 3], name="entrada_rgb")

    base_model = tf.keras.applications.ResNet50(
        input_tensor=inputs,
        include_top=False,
        weights=None,
    )

    base_model.trainable = True

    # Reproducimos la misma idea estructural del entrenamiento final:
    # una parte inicial congelada.
    for layer in base_model.layers[:80]:
        layer.trainable = False

    skip_names = [
        "conv1_relu",
        "conv2_block3_out",
        "conv3_block4_out",
        "conv4_block6_out",
    ]
    skips = [base_model.get_layer(name).output for name in skip_names]
    bottleneck = base_model.get_layer("conv5_block3_out").output

    def build_decoder_head(x, head_name, output_channels):
        up_filters = [1024, 512, 256, 64]

        for skip, filters in zip(reversed(skips), up_filters):
            x = upsample_block(filters, 4)(x)
            skip_attended = cbam_block(skip)
            x = layers.Concatenate()([x, skip_attended])

        initializer = tf.random_normal_initializer(0.0, 0.02)

        last = layers.Conv2DTranspose(
            output_channels,
            4,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            activation="tanh",
            name=head_name,
            dtype="float32",
        )

        return last(x)

    normal_output = build_decoder_head(bottleneck, "salida_normal", 3)
    roughness_output = build_decoder_head(bottleneck, "salida_roughness", 1)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[normal_output, roughness_output],
        name="DeepPBR_Generator",
    )


class DeepPBRInferencer:
    """
    Encapsula la carga e inferencia del modelo DeepPBR.

    Flujo esperado:
        inferencer = DeepPBRInferencer()
        inferencer.load()
        normal, roughness = inferencer.predict_from_rgb(image_rgb)
    """

    def __init__(self, checkpoint_prefix: Path = DEEPPBR_CKPT_PREFIX):
        self.checkpoint_prefix = Path(checkpoint_prefix)
        self.model = None

    def load(self) -> None:
        """
        Reconstruye el generador y restaura el checkpoint.
        """
        self.model = build_deep_pbr_generator()

        ckpt = tf.train.Checkpoint(generador_pbr=self.model)
        status = ckpt.restore(str(self.checkpoint_prefix))

        # El checkpoint original seguramente tenía más objetos
        # (optimizadores, discriminador, etc.). Aquí solo nos interesa
        # el generador.
        status.expect_partial()

    @staticmethod
    def preprocess_rgb(image_rgb: np.ndarray) -> np.ndarray:
        """
        Convierte una imagen RGB arbitraria en una entrada válida
        para la prueba base del modelo.

        Pasos:
        1) resize a 256x256
        2) convertir a float32
        3) normalizar a [-1, 1]

        NOTA:
        Esto todavía NO es el pipeline final de parches.
        """
        img_res = cv2.resize(image_rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
        img_res = img_res.astype(np.float32)
        img_norm = (img_res / 127.5) - 1.0
        return img_norm

    @staticmethod
    def postprocess_outputs(pred_normal: np.ndarray, pred_rough: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convierte salidas del modelo desde [-1, 1] a uint8 visualizable.

        pred_normal:
            Array float con forma (H, W, 3)
        pred_rough:
            Array float con forma (H, W, 1)

        Devuelve:
            (normal_uint8, roughness_uint8)
        """
        normal_vis = ((pred_normal + 1.0) * 0.5).clip(0.0, 1.0)
        rough_vis = ((pred_rough + 1.0) * 0.5).clip(0.0, 1.0)

        normal_uint8 = (normal_vis * 255).astype(np.uint8)
        rough_uint8 = (rough_vis.squeeze(-1) * 255).astype(np.uint8)
        return normal_uint8, rough_uint8

    def predict_from_rgb(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta inferencia a partir de una imagen RGB.

        image_rgb:
            Array uint8 con forma (H, W, 3)

        Devuelve:
            (normal_uint8, roughness_uint8)
        """
        if self.model is None:
            raise RuntimeError("El modelo DeepPBR no está cargado. Llama antes a load().")

        entrada = self.preprocess_rgb(image_rgb)

        pred_normal, pred_rough = self.model(
            tf.convert_to_tensor(np.expand_dims(entrada, axis=0), dtype=tf.float32),
            training=False,
        )

        pred_normal = pred_normal[0].numpy()
        pred_rough = pred_rough[0].numpy()

        return self.postprocess_outputs(pred_normal, pred_rough)
