from __future__ import annotations

import numpy as np


def make_tile_preview_2x2(image_rgb: np.ndarray) -> np.ndarray:
    """
    Construye un mosaico 2x2 repitiendo la textura dada.

    Parámetros
    ----------
    image_rgb : np.ndarray
        Imagen RGB con forma (H, W, 3).

    Devuelve
    --------
    np.ndarray
        Imagen RGB con forma (2H, 2W, 3).

    Explicación
    -----------
    Esta vista sirve para comprobar de forma visual si la textura puede
    repetirse sin que se note el corte.

    No "arregla" nada:
    solo multiplica la imagen en una cuadrícula 2x2.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb debe tener forma (H, W, 3).")

    top_row = np.concatenate([image_rgb, image_rgb], axis=1)
    bottom_row = np.concatenate([image_rgb, image_rgb], axis=1)
    preview = np.concatenate([top_row, bottom_row], axis=0)

    return preview
