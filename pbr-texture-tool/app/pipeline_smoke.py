from pathlib import Path

import cv2

from app.config import OUTPUTS_DIR, SAMPLE_INPUTS_DIR
from app.deeppbr_infer import DeepPBRInferencer
from app.cyclegan_infer import CycleGANInferencer


def load_rgb(path: Path):
    """
    Lee una imagen desde disco y la devuelve en RGB.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb):
    """
    Guarda una imagen RGB en disco como PNG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, image_gray):
    """
    Guarda una imagen de un canal en disco.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_gray)


def main():
    """
    Prueba de humo del pipeline interno.

    Flujo:
    1) cargar una imagen de entrada,
    2) generar normal y roughness con DeepPBR,
    3) generar PBR envejecida con CycleGAN,
    4) guardar todo.
    """
    input_path = SAMPLE_INPUTS_DIR / "deeppbr_test_input.png"
    output_dir = OUTPUTS_DIR / "phase4_pipeline_smoke"

    print(f"Cargando imagen base desde: {input_path}")
    image_rgb = load_rgb(input_path)

    print("Cargando DeepPBR...")
    deeppbr = DeepPBRInferencer()
    deeppbr.load()

    print("Generando normal y roughness base...")
    normal_base, roughness_base = deeppbr.predict_from_rgb(image_rgb)

    print("Cargando CycleGAN...")
    cyclegan = CycleGANInferencer()
    cyclegan.load()

    print("Generando versión envejecida del PBR...")
    albedo_old, normal_old, roughness_old = cyclegan.predict_from_pbr(
        rgb_uint8=image_rgb,
        normal_uint8=normal_base,
        roughness_uint8=roughness_base,
    )

    print("Guardando resultados...")
    save_rgb(output_dir / "01_input_rgb.png", image_rgb)
    save_rgb(output_dir / "02_normal_base.png", normal_base)
    save_gray(output_dir / "03_roughness_base.png", roughness_base)
    save_rgb(output_dir / "04_albedo_aged.png", albedo_old)
    save_rgb(output_dir / "05_normal_aged.png", normal_old)
    save_gray(output_dir / "06_roughness_aged.png", roughness_old)

    print(f"Prueba completada. Resultados en: {output_dir}")


if __name__ == "__main__":
    main()
