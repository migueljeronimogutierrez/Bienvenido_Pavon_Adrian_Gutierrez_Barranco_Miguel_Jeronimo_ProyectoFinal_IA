from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import MODEL_SIZE, OUTPUTS_DIR, SAMPLE_INPUTS_DIR
from app.deeppbr_infer import DeepPBRInferencer
from app.tiling import (
    apply_zoom,
    crop_back_to_original_size,
    extract_tiles,
    pad_image_for_tiling,
    resize_back_to_target,
    stitch_tiles,
    TileItem,
)


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


def save_gray(path: Path, image_gray: np.ndarray) -> None:
    """
    Guarda una imagen de un canal en disco.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_gray)


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
    1) guardar tamaño original
    2) aplicar zoom
    3) preparar padding geométrico para que encaje el tiling
    4) extraer tiles
    5) inferir tile a tile
    6) recomponer normal y roughness
    7) recortar padding
    8) reescalar al tamaño objetivo final

    target final:
        tamaño de la imagen tras zoom, no del original bruto.
        Eso es intencional en esta prueba.
    """
    # ---------------------------------------------------------
    # 1. Aplicar zoom.
    # ---------------------------------------------------------
    zoomed = apply_zoom(image_rgb, zoom_factor=zoom_factor)
    zoomed_h, zoomed_w = zoomed.shape[:2]

    # ---------------------------------------------------------
    # 2. Pad geométrico para que el tiling encaje.
    # ---------------------------------------------------------
    padded, pad_info = pad_image_for_tiling(zoomed, tile_size=tile_size, overlap=overlap)

    # ---------------------------------------------------------
    # 3. Extraer tiles.
    # ---------------------------------------------------------
    tiles = extract_tiles(padded, tile_size=tile_size, overlap=overlap)

    normal_tiles: list[TileItem] = []
    roughness_tiles: list[TileItem] = []

    # ---------------------------------------------------------
    # 4. Inferencia tile a tile.
    # ---------------------------------------------------------
    total_tiles = len(tiles)
    for idx, item in enumerate(tiles, start=1):
        print(f"Procesando tile {idx}/{total_tiles} en posición (y={item.y}, x={item.x})")

        normal_uint8, roughness_uint8 = inferencer.predict_from_rgb(item.tile)

        normal_tiles.append(TileItem(y=item.y, x=item.x, tile=normal_uint8))
        roughness_tiles.append(TileItem(y=item.y, x=item.x, tile=roughness_uint8))

    # ---------------------------------------------------------
    # 5. Reconstrucción en la imagen paddeada completa.
    # ---------------------------------------------------------
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

    # Convertimos a uint8 tras la reconstrucción.
    reconstructed_normal = np.clip(reconstructed_normal, 0, 255).astype(np.uint8)
    reconstructed_roughness = np.clip(reconstructed_roughness, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # 6. Recorte al tamaño zoomed previo al padding.
    # ---------------------------------------------------------
    cropped_normal = crop_back_to_original_size(reconstructed_normal, pad_info)
    cropped_roughness = crop_back_to_original_size(reconstructed_roughness, pad_info)

    # ---------------------------------------------------------
    # 7. En esta prueba, el tamaño objetivo final es el tamaño
    #    tras aplicar zoom.
    #    Si zoom_factor = 1.0, esto coincide con la imagen original.
    # ---------------------------------------------------------
    final_normal = resize_back_to_target(cropped_normal, target_size=(zoomed_w, zoomed_h))
    final_roughness = resize_back_to_target(cropped_roughness, target_size=(zoomed_w, zoomed_h))

    return final_normal, final_roughness


def main():
    """
    Prueba de humo del patching real con DeepPBR.
    """
    input_path = SAMPLE_INPUTS_DIR / "deeppbr_test_input.png"
    output_dir = OUTPUTS_DIR / "phase5b_tiling"

    # ---------------------------------------------------------
    # Parámetros principales de esta prueba.
    # ---------------------------------------------------------
    tile_size = MODEL_SIZE
    overlap = 64
    zoom_factor = 1.0

    print(f"Cargando imagen desde: {input_path}")
    image_rgb = load_rgb(input_path)

    print("Cargando inferencer DeepPBR...")
    inferencer = DeepPBRInferencer()
    inferencer.load()

    print("Procesando imagen completa con tiling...")
    final_normal, final_roughness = process_large_image_with_deeppbr(
        image_rgb=image_rgb,
        inferencer=inferencer,
        tile_size=tile_size,
        overlap=overlap,
        zoom_factor=zoom_factor,
    )

    print("Guardando resultados...")
    save_rgb(output_dir / "01_input_rgb.png", image_rgb)
    save_rgb(output_dir / "02_normal_tiled.png", final_normal)
    save_gray(output_dir / "03_roughness_tiled.png", final_roughness)

    print(f"Fase 5B completada. Resultados en: {output_dir}")


if __name__ == "__main__":
    main()
