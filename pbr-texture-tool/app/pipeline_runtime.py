from __future__ import annotations

import numpy as np

from app.deeppbr_infer import DeepPBRInferencer
from app.cyclegan_infer import CycleGANInferencer
from app.tiling import (
    TileItem,
    apply_zoom,
    crop_back_to_original_size,
    extract_tiles,
    pad_image_for_tiling,
    resize_back_to_target,
    stitch_tiles,
)


def process_large_image_with_deeppbr(
    image_rgb: np.ndarray,
    inferencer: DeepPBRInferencer,
    tile_size: int = 256,
    overlap: int = 64,
    zoom_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Procesa una imagen RGB arbitraria con DeepPBR usando patching.

    Flujo:
    1) aplicar zoom
    2) pad geométrico para encajar tiles
    3) extraer parches
    4) inferir tile a tile
    5) reconstruir con Hann blending
    6) recortar padding
    7) devolver al tamaño objetivo tras zoom
    """
    zoomed = apply_zoom(image_rgb, zoom_factor=zoom_factor)
    zoomed_h, zoomed_w = zoomed.shape[:2]

    padded, pad_info = pad_image_for_tiling(
        zoomed,
        tile_size=tile_size,
        overlap=overlap,
    )

    tiles = extract_tiles(
        padded,
        tile_size=tile_size,
        overlap=overlap,
    )

    normal_tiles: list[TileItem] = []
    roughness_tiles: list[TileItem] = []

    for item in tiles:
        normal_uint8, roughness_uint8 = inferencer.predict_from_rgb(item.tile)
        normal_tiles.append(TileItem(y=item.y, x=item.x, tile=normal_uint8))
        roughness_tiles.append(TileItem(y=item.y, x=item.x, tile=roughness_uint8))

    reconstructed_normal = stitch_tiles(
        tile_predictions=normal_tiles,
        output_shape=(pad_info.padded_h, pad_info.padded_w, 3),
        tile_size=tile_size,
        overlap=overlap,
    )

    reconstructed_roughness = stitch_tiles(
        tile_predictions=roughness_tiles,
        output_shape=(pad_info.padded_h, pad_info.padded_w),
        tile_size=tile_size,
        overlap=overlap,
    )

    reconstructed_normal = np.clip(reconstructed_normal, 0, 255).astype(np.uint8)
    reconstructed_roughness = np.clip(reconstructed_roughness, 0, 255).astype(np.uint8)

    cropped_normal = crop_back_to_original_size(reconstructed_normal, pad_info)
    cropped_roughness = crop_back_to_original_size(reconstructed_roughness, pad_info)

    final_normal = resize_back_to_target(
        cropped_normal,
        target_size=(zoomed_w, zoomed_h),
    )
    final_roughness = resize_back_to_target(
        cropped_roughness,
        target_size=(zoomed_w, zoomed_h),
    )

    return final_normal, final_roughness


# -----------------------------------------------------------------
# CycleGAN a escala real mediante patching.
#
# La lógica es idéntica a DeepPBR, pero:
# - la entrada es un tensor de 7 canales (RGB + Normal + Roughness)
# - la salida también es de 7 canales (Albedo + Normal + Roughness aged)
# - cada parche de 256x256x7 pasa por CycleGAN de forma individual
# - la reconstrucción usa Hann blending igual que DeepPBR
#
# Al procesar por parches, el artefacto de borde que aparecía
# cuando CycleGAN procesaba la imagen completa reducida queda
# muy mitigado: la ventana de Hann de-pondera los bordes de cada
# parche, así que las zonas problemáticas reciben menos peso en
# la mezcla final.
# -----------------------------------------------------------------


def _build_7ch_image(
    rgb_uint8: np.ndarray,
    normal_uint8: np.ndarray,
    roughness_uint8: np.ndarray,
) -> np.ndarray:
    """
    Construye un tensor de 7 canales (H, W, 7) apilando
    RGB (3) + Normal (3) + Roughness (1).

    Todos los arrays de entrada deben tener la misma altura y anchura.
    roughness_uint8 puede ser (H, W) o (H, W, 1).

    Devuelve:
        np.ndarray uint8 con forma (H, W, 7)
    """
    if roughness_uint8.ndim == 2:
        rough_3d = roughness_uint8[..., np.newaxis]
    else:
        rough_3d = roughness_uint8

    return np.concatenate([rgb_uint8, normal_uint8, rough_3d], axis=-1)


def _split_7ch_output(
    output_uint8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa un tensor de 7 canales en sus componentes:
    - albedo aged  (canales 0-2, RGB)
    - normal aged  (canales 3-5, RGB)
    - roughness aged (canal 6, escala de grises)

    Devuelve:
        (albedo_uint8, normal_uint8, roughness_uint8)
    """
    albedo = output_uint8[..., :3]
    normal = output_uint8[..., 3:6]
    roughness = output_uint8[..., 6]
    return albedo, normal, roughness


def process_large_image_with_cyclegan(
    rgb_uint8: np.ndarray,
    normal_uint8: np.ndarray,
    roughness_uint8: np.ndarray,
    inferencer: CycleGANInferencer,
    intensity: float = 0.7,
    tile_size: int = 256,
    overlap: int = 64,
    zoom_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Procesa mapas PBR con CycleGAN a escala real usando patching.

    Parámetros
    ----------
    rgb_uint8 : np.ndarray
        Imagen RGB original (H, W, 3) uint8.
    normal_uint8 : np.ndarray
        Normal map generado por DeepPBR (H, W, 3) uint8.
    roughness_uint8 : np.ndarray
        Roughness map generado por DeepPBR (H, W) o (H, W, 1) uint8.
    inferencer : CycleGANInferencer
        Modelo CycleGAN ya cargado.
    intensity : float
        Intensidad del envejecimiento (0.0 a 1.0).
    tile_size : int
        Tamaño del parche (debe coincidir con MODEL_SIZE = 256).
    overlap : int
        Solape entre parches en píxeles.
    zoom_factor : float
        Factor de zoom previo al patching.

    Flujo
    -----
    1) Construir tensor 7ch a resolución real
    2) Aplicar zoom
    3) Pad geométrico
    4) Extraer parches 256x256x7
    5) Inferir cada parche con CycleGAN
    6) Reconstruir 7ch con Hann blending
    7) Recortar padding
    8) Separar en albedo / normal / roughness aged

    Devuelve
    --------
    (albedo_aged_uint8, normal_aged_uint8, roughness_aged_uint8)
    """
    # 1) Construimos la imagen de 7 canales.
    image_7ch = _build_7ch_image(rgb_uint8, normal_uint8, roughness_uint8)

    # 2) Zoom (igual que en DeepPBR).
    zoomed = apply_zoom(image_7ch, zoom_factor=zoom_factor)
    zoomed_h, zoomed_w = zoomed.shape[:2]

    # 3) Padding geométrico.
    padded, pad_info = pad_image_for_tiling(
        zoomed,
        tile_size=tile_size,
        overlap=overlap,
    )

    # 4) Extraer parches de 7 canales.
    tiles = extract_tiles(
        padded,
        tile_size=tile_size,
        overlap=overlap,
    )

    # 5) Inferir cada parche.
    #    predict_tile espera un parche uint8 de 7 canales y
    #    devuelve un parche uint8 de 7 canales ya postprocesado.
    aged_tiles: list[TileItem] = []

    for item in tiles:
        aged_tile = inferencer.predict_tile(
            tile_7ch_uint8=item.tile,
            intensity=intensity,
        )
        aged_tiles.append(TileItem(y=item.y, x=item.x, tile=aged_tile))

    # 6) Reconstruir con Hann blending (7 canales).
    reconstructed = stitch_tiles(
        tile_predictions=aged_tiles,
        output_shape=(pad_info.padded_h, pad_info.padded_w, 7),
        tile_size=tile_size,
        overlap=overlap,
    )

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    # 7) Recortar padding.
    cropped = crop_back_to_original_size(reconstructed, pad_info)

    # Volver al tamaño pre-zoom.
    final = resize_back_to_target(
        cropped,
        target_size=(zoomed_w, zoomed_h),
    )

    # 8) Separar en 3 mapas.
    albedo_aged, normal_aged, roughness_aged = _split_7ch_output(final)

    return albedo_aged, normal_aged, roughness_aged