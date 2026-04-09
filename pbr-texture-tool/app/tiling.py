from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PaddingInfo:
    """
    Guarda la información necesaria para deshacer el padding al final.

    original_h, original_w:
        Tamaño original de la imagen antes de paddear.

    padded_h, padded_w:
        Tamaño final después del padding.
    """
    original_h: int
    original_w: int
    padded_h: int
    padded_w: int


@dataclass
class TileItem:
    """
    Representa un parche extraído de la imagen.

    y, x:
        Coordenadas de la esquina superior izquierda del parche
        dentro de la imagen paddeada o escalada.

    tile:
        Array de imagen con forma (tile_size, tile_size, C) o
        (tile_size, tile_size) según el caso.
    """
    y: int
    x: int
    tile: np.ndarray


def apply_zoom(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Reescala la imagen según un factor de zoom.

    zoom_factor:
        - 1.0  -> no cambia tamaño
        - 0.5  -> reduce a la mitad
        - 2.0  -> duplica

    IMPORTANTE:
    En vuestro caso, el uso típico será <= 1.0 para "alejar" el contenido
    antes de mandarlo a la IA, y luego reconstruir al tamaño deseado.

    Devuelve:
        imagen reescalada
    """
    if zoom_factor <= 0:
        raise ValueError("zoom_factor debe ser mayor que 0.")

    h, w = image.shape[:2]

    new_w = max(1, int(round(w * zoom_factor)))
    new_h = max(1, int(round(h * zoom_factor)))

    # INTER_AREA suele ser buena opción al reducir tamaño.
    # INTER_CUBIC suele ir bien al ampliar.
    interpolation = cv2.INTER_AREA if zoom_factor <= 1.0 else cv2.INTER_CUBIC

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized


def reflection_pad_to_min_size(image: np.ndarray, min_size: int = 256) -> tuple[np.ndarray, PaddingInfo]:
    """
    Aplica reflection padding hasta asegurar que la imagen tenga
    al menos min_size x min_size.

    Esto se usa para imágenes pequeñas, evitando deformarlas
    con un resize forzado.

    Devuelve:
        (imagen_paddeada, info_de_padding)
    """
    h, w = image.shape[:2]

    pad_bottom = max(0, min_size - h)
    pad_right = max(0, min_size - w)

    # Solo añadimos padding por abajo y derecha en esta versión base.
    # Es una decisión simple y estable para el pipeline.
    padded = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    info = PaddingInfo(
        original_h=h,
        original_w=w,
        padded_h=padded.shape[0],
        padded_w=padded.shape[1],
    )

    return padded, info


def crop_back_to_original_size(image: np.ndarray, padding_info: PaddingInfo) -> np.ndarray:
    """
    Recorta la imagen al tamaño original previo al reflection padding.

    Esto deshace el padding una vez que el modelo ya ha procesado
    la versión expandida.
    """
    return image[:padding_info.original_h, :padding_info.original_w]


def compute_stride(tile_size: int, overlap: int) -> int:
    """
    Calcula el stride entre parches.

    stride = tile_size - overlap

    Debe ser positivo.
    """
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap debe ser menor que tile_size.")
    return stride


def pad_image_for_tiling(image: np.ndarray, tile_size: int = 256, overlap: int = 64) -> tuple[np.ndarray, PaddingInfo]:
    """
    Amplía la imagen lo mínimo necesario para que pueda cubrirse
    completamente con una cuadrícula de tiles de tamaño fijo y stride fijo.

    Esto NO es el padding para imágenes pequeñas.
    Esto es padding geométrico para que el patching "encaje" sin dejar bordes.

    Devuelve:
        (imagen_paddeada, info_de_padding)
    """
    h, w = image.shape[:2]
    stride = compute_stride(tile_size, overlap)

    # Si la imagen es menor que un tile, forzamos al menos un tile.
    target_h = max(h, tile_size)
    target_w = max(w, tile_size)

    # Queremos que:
    # target = tile_size + n * stride
    # para algún n entero >= 0
    if target_h > tile_size:
        n_h = math.ceil((target_h - tile_size) / stride)
        target_h = tile_size + n_h * stride

    if target_w > tile_size:
        n_w = math.ceil((target_w - tile_size) / stride)
        target_w = tile_size + n_w * stride

    pad_bottom = target_h - h
    pad_right = target_w - w

    padded = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    info = PaddingInfo(
        original_h=h,
        original_w=w,
        padded_h=target_h,
        padded_w=target_w,
    )

    return padded, info


def extract_tiles(image: np.ndarray, tile_size: int = 256, overlap: int = 64) -> list[TileItem]:
    """
    Extrae una lista de parches con solape a partir de una imagen ya preparada.

    image:
        Imagen paddeada lo suficiente como para que el tiling encaje.

    Devuelve:
        lista de TileItem
    """
    h, w = image.shape[:2]
    stride = compute_stride(tile_size, overlap)

    tiles: list[TileItem] = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size].copy()
            tiles.append(TileItem(y=y, x=x, tile=tile))

    return tiles


def make_hann_window_2d(tile_size: int = 256, epsilon: float = 1e-6) -> np.ndarray:
    """
    Crea una ventana de Hann bidimensional para mezclar parches.

    La idea:
    - los bordes pesan menos,
    - el centro pesa más,
    - al solapar parches, las costuras se suavizan.

    epsilon:
        Evita tener pesos exactamente cero, lo cual ayuda a no dejar
        regiones sin contribución por redondeos numéricos.
    """
    hann_1d = np.hanning(tile_size).astype(np.float32)

    # Producto externo para construir la ventana 2D.
    window_2d = np.outer(hann_1d, hann_1d).astype(np.float32)

    # Evitamos ceros absolutos.
    window_2d = np.maximum(window_2d, epsilon)
    return window_2d


def stitch_tiles(
    tile_predictions: list[TileItem],
    output_shape: tuple[int, int, int] | tuple[int, int],
    tile_size: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Reconstruye una imagen completa a partir de tiles ya inferidos.

    tile_predictions:
        lista de TileItem donde cada tile ya contiene la salida del modelo

    output_shape:
        forma final de la imagen paddeada antes del recorte
        por ejemplo:
        - (H, W, 3) para normal
        - (H, W) para mapas de un canal
        - (H, W, 1) si quieres mantener canal explícito

    Devuelve:
        imagen reconstruida float32

    Estrategia
    ----------
    1) acumulamos suma ponderada de tiles
    2) acumulamos suma de pesos
    3) dividimos ambos al final
    """
    if len(output_shape) not in (2, 3):
        raise ValueError("output_shape debe tener longitud 2 o 3.")

    if len(output_shape) == 2:
        acc = np.zeros(output_shape, dtype=np.float32)
        weight_acc = np.zeros(output_shape, dtype=np.float32)
        single_channel = True
    else:
        acc = np.zeros(output_shape, dtype=np.float32)
        weight_acc = np.zeros(output_shape[:2], dtype=np.float32)
        single_channel = False

    window_2d = make_hann_window_2d(tile_size=tile_size)

    for item in tile_predictions:
        y, x, tile = item.y, item.x, item.tile

        if single_channel:
            tile_f = tile.astype(np.float32)
            acc[y:y + tile_size, x:x + tile_size] += tile_f * window_2d
            weight_acc[y:y + tile_size, x:x + tile_size] += window_2d
        else:
            tile_f = tile.astype(np.float32)
            acc[y:y + tile_size, x:x + tile_size, :] += tile_f * window_2d[..., None]
            weight_acc[y:y + tile_size, x:x + tile_size] += window_2d

    if single_channel:
        result = acc / np.maximum(weight_acc, 1e-8)
    else:
        result = acc / np.maximum(weight_acc[..., None], 1e-8)

    return result


def resize_back_to_target(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Reescala una imagen al tamaño objetivo (width, height).

    target_size:
        (target_w, target_h)
    """
    target_w, target_h = target_size

    if image.ndim == 2:
        interpolation = cv2.INTER_CUBIC
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    interpolation = cv2.INTER_CUBIC
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
