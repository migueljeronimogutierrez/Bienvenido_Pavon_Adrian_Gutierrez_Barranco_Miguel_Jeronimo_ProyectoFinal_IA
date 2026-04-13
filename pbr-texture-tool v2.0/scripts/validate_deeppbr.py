from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PREFIX = PROJECT_ROOT / "models" / "deeppbr" / "ckpt-42"
INPUT_IMAGE_PATH = PROJECT_ROOT / "sample_inputs" / "deeppbr_test_input.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3_deeppbr_validation"
TEST_SIZE = 256


policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)


def cbam_block(input_feature, ratio=8):
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

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
    avg_pool = shared_layer_two(shared_layer_one(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channels))(max_pool)
    max_pool = shared_layer_two(shared_layer_one(max_pool))

    channel_attention = layers.Activation("sigmoid")(layers.Add()([avg_pool, max_pool]))
    cbam_feature = layers.Multiply()([input_feature, channel_attention])

    spatial_avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
    )(cbam_feature)

    spatial_max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
    )(cbam_feature)

    concat = layers.Concatenate(axis=-1)([spatial_avg_pool, spatial_max_pool])

    spatial_attention = layers.Activation("sigmoid")(
        layers.Conv2D(1, (7, 7), padding="same", activation=None)(concat)
    )

    cbam_feature = layers.Multiply()([cbam_feature, spatial_attention])
    return cbam_feature


def upsample_block(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    # Sin nombre fijo para evitar colisiones entre los dos decoders.
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
    inputs = layers.Input(shape=[256, 256, 3], name="entrada_rgb")

    base_model = tf.keras.applications.ResNet50(
        input_tensor=inputs,
        include_top=False,
        weights=None,
    )

    base_model.trainable = True
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

    normal_output = build_decoder_head(
        bottleneck,
        head_name="salida_normal",
        output_channels=3,
    )

    roughness_output = build_decoder_head(
        bottleneck,
        head_name="salida_roughness",
        output_channels=1,
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[normal_output, roughness_output],
        name="DeepPBR_Generator",
    )

    return model


def cargar_imagen_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def preparar_entrada_rgb(img_rgb: np.ndarray) -> np.ndarray:
    img_res = cv2.resize(img_rgb, (TEST_SIZE, TEST_SIZE), interpolation=cv2.INTER_AREA)
    img_res = img_res.astype(np.float32)
    img_norm = (img_res / 127.5) - 1.0
    return img_norm


def guardar_resultados(pred_normal: np.ndarray, pred_rough: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_vis = ((pred_normal + 1.0) * 0.5).clip(0.0, 1.0)
    rough_vis = ((pred_rough + 1.0) * 0.5).clip(0.0, 1.0)

    normal_uint8 = (normal_vis * 255).astype(np.uint8)
    rough_uint8 = (rough_vis.squeeze(-1) * 255).astype(np.uint8)

    cv2.imwrite(str(output_dir / "deeppbr_normal.png"), cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "deeppbr_roughness.png"), rough_uint8)


def main() -> None:
    checkpoint_dir = CHECKPOINT_PREFIX.parent
    ckpt_index = checkpoint_dir / "ckpt-42.index"
    ckpt_data = checkpoint_dir / "ckpt-42.data-00000-of-00001"

    if not ckpt_index.exists():
        raise FileNotFoundError(f"Falta el archivo: {ckpt_index}")

    if not ckpt_data.exists():
        raise FileNotFoundError(f"Falta el archivo: {ckpt_data}")

    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"No existe la imagen de entrada: {INPUT_IMAGE_PATH}")

    print(f"Política de precisión activa: {tf.keras.mixed_precision.global_policy()}")
    print(f"Checkpoint detectado con prefijo: {CHECKPOINT_PREFIX}")
    print(f"Imagen de prueba detectada en: {INPUT_IMAGE_PATH}")

    print("Reconstruyendo el generador DeepPBR...")
    generator = build_deep_pbr_generator()

    print("Creando objeto Checkpoint...")
    ckpt = tf.train.Checkpoint(generador_pbr=generator)

    print("Restaurando checkpoint...")
    status = ckpt.restore(str(CHECKPOINT_PREFIX))
    status.expect_partial()

    print("Leyendo imagen de prueba...")
    img_rgb = cargar_imagen_rgb(INPUT_IMAGE_PATH)

    print("Preparando entrada RGB...")
    entrada = preparar_entrada_rgb(img_rgb)

    print("Ejecutando inferencia...")
    pred_normal, pred_rough = generator(
        tf.convert_to_tensor(np.expand_dims(entrada, axis=0), dtype=tf.float32),
        training=False,
    )

    pred_normal = pred_normal[0].numpy()
    pred_rough = pred_rough[0].numpy()

    print("Guardando resultados...")
    guardar_resultados(pred_normal, pred_rough, OUTPUT_DIR)

    print(f"Validación completada. Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
