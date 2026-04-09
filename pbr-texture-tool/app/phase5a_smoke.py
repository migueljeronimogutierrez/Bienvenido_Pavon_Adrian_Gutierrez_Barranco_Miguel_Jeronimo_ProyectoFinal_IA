from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import OUTPUTS_DIR, SAMPLE_INPUTS_DIR
from app.geometry import warp_perspective_from_points
from app.preview import make_tile_preview_2x2


def load_rgb(path: Path) -> np.ndarray:
    """
    Lee una imagen desde disco y la devuelve en RGB.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """
    Guarda una imagen RGB en disco.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def main():
    """
    Prueba de humo de la fase 5A.

    Flujo:
    1) cargar imagen,
    2) aplicar corrección de perspectiva usando 4 puntos manuales,
    3) generar mosaico 2x2,
    4) guardar resultados.
    """
    input_path = SAMPLE_INPUTS_DIR / "deeppbr_test_input.png"
    output_dir = OUTPUTS_DIR / "phase5a_geometry"

    print(f"Cargando imagen desde: {input_path}")
    image_rgb = load_rgb(input_path)

    # ==========================================================
    # AJUSTA ESTOS 4 PUNTOS SEGÚN TU IMAGEN
    #
    # Formato: [x, y]
    # Orden libre: el módulo geometry.py ya los reordena.
    #
    # IMPORTANTE:
    # Estos valores son SOLO de ejemplo.
    # Tendrás que cambiarlos para que coincidan con el material que
    # quieres rectificar en tu imagen real.
    # ==========================================================
    points = np.array(
        [
            [120, 120],
            [900, 100],
            [920, 900],
            [140, 920],
        ],
        dtype=np.float32,
    )

    print("Aplicando corrección de perspectiva...")
    warped = warp_perspective_from_points(image_rgb, points)

    print("Construyendo preview tileable 2x2...")
    tile_preview = make_tile_preview_2x2(warped)

    print("Guardando resultados...")
    save_rgb(output_dir / "01_original.png", image_rgb)
    save_rgb(output_dir / "02_warped.png", warped)
    save_rgb(output_dir / "03_tile_preview_2x2.png", tile_preview)

    print(f"Fase 5A completada. Resultados en: {output_dir}")


if __name__ == "__main__":
    main()
