from __future__ import annotations

import numpy as np
import cv2


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """
    Ordena 4 puntos 2D en el orden:

    1) top-left
    2) top-right
    3) bottom-right
    4) bottom-left

    Parámetros
    ----------
    points : np.ndarray
        Array con forma (4, 2). Cada fila representa un punto [x, y].

    Devuelve
    --------
    np.ndarray
        Array float32 con forma (4, 2) en el orden correcto.

    Explicación
    -----------
    OpenCV espera que los 4 puntos de entrada sigan un orden consistente.
    Si el orden cambia, la perspectiva puede salir girada, invertida o rota.

    Estrategia usada
    ----------------
    - La suma x+y suele ser mínima en la esquina superior izquierda.
    - La suma x+y suele ser máxima en la esquina inferior derecha.
    - La resta x-y ayuda a distinguir superior derecha e inferior izquierda.
    """
    if points.shape != (4, 2):
        raise ValueError(f"Se esperaban 4 puntos con forma (4, 2), recibido: {points.shape}")

    pts = points.astype(np.float32)

    # top-left: punto con menor suma
    # bottom-right: punto con mayor suma
    s = pts.sum(axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]

    # top-right: menor diferencia (x - y) o, equivalente, mínimo en diff = y - x
    # bottom-left: mayor diferencia
    diff = np.diff(pts, axis=1)  # shape (4,1): y - x
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return ordered


def compute_output_size(ordered_points: np.ndarray) -> tuple[int, int]:
    """
    Calcula el tamaño de salida apropiado para la textura corregida.

    ordered_points debe venir ya en orden:
    [top-left, top-right, bottom-right, bottom-left]

    Devuelve
    --------
    tuple[int, int]
        (width, height) enteros.

    Explicación
    -----------
    El plano de entrada puede ser un cuadrilátero arbitrario.
    Para rectificarlo, estimamos:
    - ancho superior e inferior,
    - alto izquierdo y derecho,
    y usamos el máximo en cada dimensión.
    """
    if ordered_points.shape != (4, 2):
        raise ValueError("ordered_points debe tener forma (4, 2).")

    (tl, tr, br, bl) = ordered_points

    # Distancias horizontales aproximadas.
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(round(max(width_top, width_bottom)))

    # Distancias verticales aproximadas.
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(round(max(height_left, height_right)))

    # Seguridad mínima para evitar tamaños degenerados.
    width = max(width, 1)
    height = max(height, 1)

    return width, height


def warp_perspective_from_points(image_rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Corrige la perspectiva de una imagen RGB a partir de 4 puntos.

    Parámetros
    ----------
    image_rgb : np.ndarray
        Imagen RGB con forma (H, W, 3), dtype uint8.
    points : np.ndarray
        Array con forma (4, 2), puntos del plano material en coordenadas
        de la imagen original.

    Devuelve
    --------
    np.ndarray
        Imagen RGB corregida en perspectiva.

    Flujo interno
    -------------
    1) ordenar puntos
    2) calcular tamaño destino
    3) construir cuadrilátero destino rectangular
    4) obtener matriz de homografía
    5) aplicar cv2.warpPerspective
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb debe tener forma (H, W, 3).")

    ordered = order_points_clockwise(points)
    width, height = compute_output_size(ordered)

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, destination)

    warped = cv2.warpPerspective(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
    )

    return warped
