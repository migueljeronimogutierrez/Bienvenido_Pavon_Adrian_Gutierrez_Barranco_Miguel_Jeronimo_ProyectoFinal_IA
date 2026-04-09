from __future__ import annotations

import base64
import hashlib
import math
from io import BytesIO

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from app.cyclegan_infer import CycleGANInferencer
from app.deeppbr_infer import DeepPBRInferencer
from app.io_utils import (
    gray_numpy_to_pil,
    pil_to_png_bytes,
    pil_to_rgb_numpy,
    rgb_numpy_to_pil,
)
from app.pipeline_runtime import (
    process_large_image_with_cyclegan,
    process_large_image_with_deeppbr,
)
from app.preview import make_tile_preview_2x2
from app.zip_utils import build_results_zip


# -----------------------------------------------------------------
# Page configuration.
# Must be the first Streamlit call in the script.
# -----------------------------------------------------------------

st.set_page_config(
    page_title="PBR Texture Tool",
    page_icon="🧱",
    layout="wide",
)


# -----------------------------------------------------------------
# Session-state helpers.
#
# Streamlit reruns the script from top to bottom every time the user
# interacts with widgets such as selectboxes or sliders.  Therefore,
# any inference results stored only in local variables would be lost
# after each rerun.
#
# To prevent that, we persist all generated maps in st.session_state.
# This allows the app to keep the latest results visible until the
# user uploads a different image or explicitly generates new outputs.
# -----------------------------------------------------------------

RESULT_STATE_KEYS = [
    "results_ready",
    "source_image_hash",
    "last_run_settings",
    "normal_base",
    "roughness_base",
    "albedo_aged",
    "normal_aged",
    "roughness_aged",
]


def init_session_state() -> None:
    """Initializes all session-state keys used by the app."""
    defaults = {
        "results_ready": False,
        "source_image_hash": None,
        "last_run_settings": None,
        "normal_base": None,
        "roughness_base": None,
        "albedo_aged": None,
        "normal_aged": None,
        "roughness_aged": None,
        "compare_option": "Albedo: original vs aged",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def clear_inference_results() -> None:
    """
    Clears the latest inference outputs from the current Streamlit
    session.  This is called automatically when the uploaded image
    changes, so outdated results are never shown for a different file.
    """
    for key in RESULT_STATE_KEYS:
        if key == "source_image_hash":
            st.session_state[key] = None
        elif key == "results_ready":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

    # Reset comparison selector to a safe default for aged outputs.
    st.session_state["compare_option"] = "Albedo: original vs aged"



def get_current_settings(
    use_real_tiling: bool,
    zoom_factor: float,
    overlap: int,
    apply_aging: bool,
    aging_intensity: float,
    show_tile_preview: bool,
) -> dict[str, object]:
    """Builds a serializable snapshot of the current UI settings."""
    return {
        "use_real_tiling": use_real_tiling,
        "zoom_factor": float(zoom_factor),
        "overlap": int(overlap),
        "apply_aging": bool(apply_aging),
        "aging_intensity": float(aging_intensity),
        "show_tile_preview": bool(show_tile_preview),
    }


init_session_state()


# -----------------------------------------------------------------
# Model loading with Streamlit cache.
#
# @st.cache_resource ensures the models are loaded only once
# per session and reused across reruns (button clicks, slider
# changes, etc.).  This avoids reloading heavy weights every
# time the user interacts with the app.
# -----------------------------------------------------------------

@st.cache_resource

def load_deeppbr_model() -> DeepPBRInferencer:
    model = DeepPBRInferencer()
    model.load()
    return model


@st.cache_resource

def load_cyclegan_model() -> CycleGANInferencer:
    model = CycleGANInferencer()
    model.load()
    return model


# -----------------------------------------------------------------
# Utility: convert a numpy image to a base64-encoded PNG string.
# Used by the before/after comparison slider (HTML component).
# -----------------------------------------------------------------


def numpy_to_base64_png(img: np.ndarray) -> str:
    """
    Converts a NumPy image array to a base64-encoded PNG string
    suitable for embedding in an HTML <img> src attribute.

    Supports:
    - (H, W, 3) RGB uint8  -> saved as RGB PNG
    - (H, W) grayscale uint8 -> saved as grayscale PNG
    """
    if img.ndim == 2:
        pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(img.astype(np.uint8), mode="RGB")

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# -----------------------------------------------------------------
# Utility: estimate processing time before running inference.
#
# Based on benchmarks with a 1024x1024 image on an i5 + GTX 1650:
#   - DeepPBR:  ~1.3 seconds per tile
#   - CycleGAN: ~2.8 seconds per tile
#
# These are rough estimates.  Actual times will vary depending on
# the user's hardware, but the order of magnitude should be close.
# -----------------------------------------------------------------


def estimate_processing_time(
    h: int,
    w: int,
    tile_size: int,
    overlap: int,
    zoom_factor: float,
    apply_aging: bool,
) -> tuple[int, float]:
    """
    Estimates the total number of tiles and approximate processing
    time in seconds for a given image size and settings.

    Returns:
        (n_tiles, estimated_seconds)
    """
    # Apply zoom to get effective dimensions.
    zh = max(1, int(round(h * zoom_factor)))
    zw = max(1, int(round(w * zoom_factor)))

    stride = tile_size - overlap

    # Compute padded dimensions (same logic as pad_image_for_tiling).
    target_h = max(zh, tile_size)
    target_w = max(zw, tile_size)

    if target_h > tile_size:
        n_h = math.ceil((target_h - tile_size) / stride)
        target_h = tile_size + n_h * stride

    if target_w > tile_size:
        n_w = math.ceil((target_w - tile_size) / stride)
        target_w = tile_size + n_w * stride

    # Count tiles.
    tiles_y = 1 + (target_h - tile_size) // stride if target_h > tile_size else 1
    tiles_x = 1 + (target_w - tile_size) // stride if target_w > tile_size else 1
    n_tiles = tiles_y * tiles_x

    # Benchmark-based estimates (seconds per tile).
    DEEPPBR_SPT = 1.3
    CYCLEGAN_SPT = 2.8

    seconds = n_tiles * DEEPPBR_SPT
    if apply_aging:
        seconds += n_tiles * CYCLEGAN_SPT

    return n_tiles, seconds


# -----------------------------------------------------------------
# Utility: render a before/after comparison slider.
#
# This creates a self-contained HTML widget that shows two images
# side by side with a draggable divider.  The user can drag the
# slider left and right to compare the original and the aged
# version (or any two images).
#
# Everything runs in the browser -- no extra Python dependencies.
# -----------------------------------------------------------------


def render_comparison_slider(
    img_left: np.ndarray,
    img_right: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Aged",
    height: int = 450,
) -> None:
    """
    Renders an interactive before/after comparison slider using
    Streamlit's HTML component system.

    img_left, img_right:
        NumPy arrays (H, W, 3) or (H, W) with the two images.
    label_left, label_right:
        Text labels shown on each side of the slider.
    height:
        Pixel height of the comparison widget.
    """
    b64_left = numpy_to_base64_png(img_left)
    b64_right = numpy_to_base64_png(img_right)

    html = f"""
    <div id="comp-container" style="
        position: relative;
        width: 100%;
        height: {height}px;
        overflow: hidden;
        border-radius: 8px;
        cursor: col-resize;
        user-select: none;
        background: #111;
    ">
        <img src="data:image/png;base64,{b64_right}"
             style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain;"
             draggable="false">

        <div id="clip-left" style="
            position: absolute;
            top: 0; left: 0;
            width: 50%;
            height: 100%;
            overflow: hidden;
        ">
            <img src="data:image/png;base64,{b64_left}"
                 id="img-left"
                 style="position:absolute; top:0; left:0; height:100%; object-fit:contain;"
                 draggable="false">
        </div>

        <div id="slider-handle" style="
            position: absolute;
            top: 0;
            left: 50%;
            width: 3px;
            height: 100%;
            background: white;
            transform: translateX(-50%);
            z-index: 10;
            pointer-events: none;
        ">
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                color: #333;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                pointer-events: none;
            ">&#x27FA;</div>
        </div>

        <div style="
            position:absolute; top:10px; left:10px;
            background:rgba(0,0,0,0.65); color:white;
            padding:3px 10px; border-radius:4px;
            font-size:12px; font-family:sans-serif;
            pointer-events:none; z-index:11;
        ">{label_left}</div>

        <div style="
            position:absolute; top:10px; right:10px;
            background:rgba(0,0,0,0.65); color:white;
            padding:3px 10px; border-radius:4px;
            font-size:12px; font-family:sans-serif;
            pointer-events:none; z-index:11;
        ">{label_right}</div>
    </div>

    <script>
    (function() {{
        var container = document.getElementById('comp-container');
        var clipLeft  = document.getElementById('clip-left');
        var imgLeft   = document.getElementById('img-left');
        var handle    = document.getElementById('slider-handle');
        var dragging  = false;

        // Set left image width to match container (not the clip div).
        function syncWidth() {{
            imgLeft.style.width = container.offsetWidth + 'px';
        }}
        syncWidth();
        window.addEventListener('resize', syncWidth);

        function update(clientX) {{
            var rect = container.getBoundingClientRect();
            var pct  = ((clientX - rect.left) / rect.width) * 100;
            pct = Math.max(1, Math.min(99, pct));
            clipLeft.style.width = pct + '%';
            handle.style.left   = pct + '%';
        }}

        container.addEventListener('mousedown',  function() {{ dragging = true; }});
        document.addEventListener('mouseup',      function() {{ dragging = false; }});
        container.addEventListener('mousemove',   function(e) {{ if (dragging) update(e.clientX); }});
        container.addEventListener('click',       function(e) {{ update(e.clientX); }});

        container.addEventListener('touchstart',  function() {{ dragging = true; }});
        document.addEventListener('touchend',     function() {{ dragging = false; }});
        container.addEventListener('touchmove',   function(e) {{
            if (dragging) {{ update(e.touches[0].clientX); e.preventDefault(); }}
        }});
    }})();
    </script>
    """

    components.html(html, height=height + 10)


# -----------------------------------------------------------------
# Title and description.
# -----------------------------------------------------------------

st.title("🧱 PBR Texture Tool")
st.caption(
    "Generate PBR maps (Normal + Roughness) from a photograph, "
    "with optional synthetic aging.  "
    "Domain: stone, brick, concrete, asphalt."
)


# -----------------------------------------------------------------
# Tips banner.
# -----------------------------------------------------------------

st.info(
    "**Tips for best results:**  \n"
    "• Use frontal shots with even, diffuse lighting.  \n"
    "• Avoid extreme shadows, reflections or out-of-focus areas.  \n"
    "• Stick to stone-family materials (brick, concrete, asphalt, stone).  \n"
    "• Very high resolutions will increase processing time significantly."
)


# -----------------------------------------------------------------
# Sidebar: settings panel.
# -----------------------------------------------------------------

st.sidebar.header("Settings")

use_real_tiling = st.sidebar.checkbox(
    "Real patching mode",
    value=True,
    help=(
        "Splits the image into overlapping 256x256 tiles, runs "
        "inference on each tile individually, and reconstructs the "
        "full-resolution output with Hann-window blending.  "
        "Recommended for any image larger than 256x256."
    ),
)

zoom_factor = st.sidebar.slider(
    "Zoom factor",
    min_value=0.25,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help=(
        "Downscales the image before patching.  "
        "1.0 = original resolution.  "
        "Lower values speed up inference but reduce detail."
    ),
)

overlap = st.sidebar.selectbox(
    "Tile overlap (px)",
    options=[64, 96, 128],
    index=0,
    help=(
        "Number of pixels shared between adjacent tiles.  "
        "Higher overlap gives smoother blending but increases "
        "the total number of tiles and processing time."
    ),
)

st.sidebar.markdown("---")

apply_aging = st.sidebar.checkbox(
    "Apply synthetic aging (CycleGAN)",
    value=False,
    help=(
        "Runs CycleGAN to generate weathered versions of all maps.  "
        "When real patching is enabled, aging also runs at full "
        "resolution using the same tile-based pipeline."
    ),
)

aging_intensity = st.sidebar.slider(
    "Aging intensity",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help=(
        "Controls how strong the aging effect is.  "
        "0.1 = subtle weathering, 1.0 = heavy deterioration."
    ),
)

st.sidebar.markdown("---")

show_tile_preview = st.sidebar.checkbox(
    "Show tileable 2x2 preview",
    value=True,
    help=(
        "Displays a 2x2 mosaic of the uploaded texture so you can "
        "visually check if the pattern tiles seamlessly."
    ),
)

current_settings = get_current_settings(
    use_real_tiling=use_real_tiling,
    zoom_factor=zoom_factor,
    overlap=overlap,
    apply_aging=apply_aging,
    aging_intensity=aging_intensity,
    show_tile_preview=show_tile_preview,
)


# -----------------------------------------------------------------
# Image upload.
# -----------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a texture image",
    type=["png", "jpg", "jpeg"],
    help="PNG or JPG photograph of a stone-family material.",
)

if uploaded_file is None:
    st.markdown(
        "_Upload an image to get started.  "
        "The tool will generate Normal and Roughness maps, "
        "and optionally apply synthetic aging._"
    )
    st.stop()


# -----------------------------------------------------------------
# Read and convert the uploaded image.
#
# We hash the raw file bytes so the app can detect when the user has
# switched to a different image.  In that case, old results are
# cleared automatically from session_state.
# -----------------------------------------------------------------

uploaded_bytes = uploaded_file.getvalue()
current_image_hash = hashlib.sha256(uploaded_bytes).hexdigest()

if st.session_state["source_image_hash"] != current_image_hash:
    clear_inference_results()
    st.session_state["source_image_hash"] = current_image_hash

input_pil = Image.open(BytesIO(uploaded_bytes))
input_rgb = pil_to_rgb_numpy(input_pil)
img_h, img_w = input_rgb.shape[:2]


# -----------------------------------------------------------------
# Image validation warnings.
#
# These help the user understand potential issues before running
# inference, preventing wasted time and unexpected results.
# -----------------------------------------------------------------

if img_h < 256 or img_w < 256:
    st.warning(
        f"**Small image detected** ({img_w} x {img_h}).  "
        "The model works on 256x256 tiles. Images smaller than this "
        "will be padded, which may reduce quality."
    )

if img_h > 2048 or img_w > 2048:
    st.warning(
        f"**Large image detected** ({img_w} x {img_h}).  "
        "Processing will take significantly longer. Consider reducing "
        "the Zoom factor in Settings to speed things up."
    )

if max(img_h, img_w) / max(min(img_h, img_w), 1) > 2.0:
    st.warning(
        f"**Unusual aspect ratio** ({img_w} x {img_h}).  "
        "The tool works best with square or near-square textures. "
        "Extreme aspect ratios may produce uneven results."
    )


# -----------------------------------------------------------------
# Display original image and tileable preview.
# -----------------------------------------------------------------

if show_tile_preview:
    col_orig, col_tile = st.columns(2)

    with col_orig:
        st.subheader("Original image")
        st.image(input_rgb, channels="RGB", use_container_width=True)

    with col_tile:
        tile_preview = make_tile_preview_2x2(input_rgb)
        st.subheader("Tileable 2x2 preview")
        st.image(tile_preview, channels="RGB", use_container_width=True)
else:
    st.subheader("Original image")
    st.image(input_rgb, channels="RGB", use_container_width=True)


# -----------------------------------------------------------------
# Time estimation.
#
# Shown right before the "Generate" button so the user knows
# what to expect.  The estimate is based on benchmarks with a
# 1024x1024 image on an i5 + GTX 1650 (CPU inference).
# -----------------------------------------------------------------

if use_real_tiling:
    n_tiles, est_seconds = estimate_processing_time(
        h=img_h,
        w=img_w,
        tile_size=256,
        overlap=overlap,
        zoom_factor=zoom_factor,
        apply_aging=apply_aging,
    )

    est_minutes = int(est_seconds // 60)
    est_secs = int(est_seconds % 60)

    if est_minutes > 0:
        time_str = f"~{est_minutes} min {est_secs} s"
    else:
        time_str = f"~{est_secs} s"

    pipeline_label = "DeepPBR + CycleGAN" if apply_aging else "DeepPBR only"

    st.caption(
        f"Estimated processing time: **{time_str}** "
        f"({n_tiles} tiles, {pipeline_label}).  "
        f"Times may vary depending on your hardware."
    )


# -----------------------------------------------------------------
# Main inference button.
# -----------------------------------------------------------------

generate_clicked = st.button("Generate PBR maps", type="primary", use_container_width=True)

if generate_clicked:

    # --- DeepPBR inference -----------------------------------------

    deeppbr = load_deeppbr_model()

    if use_real_tiling:
        with st.spinner("Generating base maps with real patching (this may take a few minutes)..."):
            normal_base, roughness_base = process_large_image_with_deeppbr(
                image_rgb=input_rgb,
                inferencer=deeppbr,
                tile_size=256,
                overlap=overlap,
                zoom_factor=zoom_factor,
            )
    else:
        with st.spinner("Generating base maps (fast mode — single 256x256 pass)..."):
            normal_base, roughness_base = deeppbr.predict_from_rgb(input_rgb)

    # --- CycleGAN inference (optional) -----------------------------

    albedo_aged = None
    normal_aged = None
    roughness_aged = None

    if apply_aging:
        cyclegan = load_cyclegan_model()

        if use_real_tiling:
            with st.spinner(
                "Applying aging with real patching (this will take a while — "
                "each tile passes through CycleGAN individually)..."
            ):
                albedo_aged, normal_aged, roughness_aged = process_large_image_with_cyclegan(
                    rgb_uint8=input_rgb,
                    normal_uint8=normal_base,
                    roughness_uint8=roughness_base,
                    inferencer=cyclegan,
                    intensity=aging_intensity,
                    tile_size=256,
                    overlap=overlap,
                    zoom_factor=zoom_factor,
                )
        else:
            with st.spinner("Applying aging (fast mode — single 256x256 pass)..."):
                albedo_aged, normal_aged, roughness_aged = cyclegan.predict_from_pbr(
                    rgb_uint8=input_rgb,
                    normal_uint8=normal_base,
                    roughness_uint8=roughness_base,
                    intensity=aging_intensity,
                )

    # Persist all outputs so they survive widget-triggered reruns.
    st.session_state["normal_base"] = normal_base.copy()
    st.session_state["roughness_base"] = roughness_base.copy()
    st.session_state["albedo_aged"] = None if albedo_aged is None else albedo_aged.copy()
    st.session_state["normal_aged"] = None if normal_aged is None else normal_aged.copy()
    st.session_state["roughness_aged"] = None if roughness_aged is None else roughness_aged.copy()
    st.session_state["last_run_settings"] = current_settings
    st.session_state["results_ready"] = True


# -----------------------------------------------------------------
# Results rendering.
#
# Results are displayed whenever they exist in session_state, not
# only during the button-click run.  This is the key fix that keeps
# the outputs visible when the user changes the comparison selectbox
# or any other widget that triggers a rerun.
# -----------------------------------------------------------------

if st.session_state["results_ready"]:
    normal_base = st.session_state["normal_base"]
    roughness_base = st.session_state["roughness_base"]
    albedo_aged = st.session_state["albedo_aged"]
    normal_aged = st.session_state["normal_aged"]
    roughness_aged = st.session_state["roughness_aged"]

    last_run_settings = st.session_state["last_run_settings"] or {}
    settings_changed_since_last_run = current_settings != last_run_settings

    if settings_changed_since_last_run:
        st.info(
            "The displayed outputs correspond to the **last completed run**. "
            "You changed one or more settings afterwards.  "
            "Click **Generate PBR maps** again to refresh the results with the new configuration."
        )

    # --- Display base results --------------------------------------

    st.subheader("Base PBR results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Normal map**")
        st.image(normal_base, channels="RGB", use_container_width=True)

    with col2:
        st.markdown("**Roughness map**")
        st.image(roughness_base, clamp=True, use_container_width=True)

    # --- Display aged results (if enabled in the last run) ---------

    has_aged_results = (
        albedo_aged is not None
        and normal_aged is not None
        and roughness_aged is not None
    )

    if has_aged_results:
        st.subheader("Aged results (CycleGAN)")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Albedo aged**")
            st.image(albedo_aged, channels="RGB", use_container_width=True)

        with c2:
            st.markdown("**Normal aged**")
            st.image(normal_aged, channels="RGB", use_container_width=True)

        with c3:
            st.markdown("**Roughness aged**")
            st.image(roughness_aged, clamp=True, use_container_width=True)

        # --- Before / after comparison slider ----------------------
        #
        # The selectbox now uses a persistent session_state key,
        # while the images themselves are also stored in session_state.
        # Therefore, switching comparison mode no longer destroys the
        # previously computed results.
        # -----------------------------------------------------------

        st.subheader("Before / after comparison")

        compare_option = st.selectbox(
            "Compare:",
            options=[
                "Albedo: original vs aged",
                "Normal: base vs aged",
                "Roughness: base vs aged",
            ],
            key="compare_option",
        )

        if compare_option == "Albedo: original vs aged":
            render_comparison_slider(
                img_left=input_rgb,
                img_right=albedo_aged,
                label_left="Original",
                label_right="Aged albedo",
            )
        elif compare_option == "Normal: base vs aged":
            render_comparison_slider(
                img_left=normal_base,
                img_right=normal_aged,
                label_left="Normal base",
                label_right="Normal aged",
            )
        else:
            render_comparison_slider(
                img_left=roughness_base,
                img_right=roughness_aged,
                label_left="Roughness base",
                label_right="Roughness aged",
            )

    # --- Downloads -------------------------------------------------

    st.subheader("Downloads")

    # Convert base maps to PIL / PNG bytes.
    normal_base_pil = rgb_numpy_to_pil(normal_base)
    roughness_base_pil = gray_numpy_to_pil(roughness_base)

    normal_base_bytes = pil_to_png_bytes(normal_base_pil)
    roughness_base_bytes = pil_to_png_bytes(roughness_base_pil)

    # Individual download buttons for base maps.
    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Download normal_base.png",
            data=normal_base_bytes,
            file_name="normal_base.png",
            mime="image/png",
        )

    with d2:
        st.download_button(
            label="Download roughness_base.png",
            data=roughness_base_bytes,
            file_name="roughness_base.png",
            mime="image/png",
        )

    # Build the ZIP contents (always includes base maps).
    zip_files = {
        "normal_base.png": normal_base_bytes,
        "roughness_base.png": roughness_base_bytes,
    }

    # Individual download buttons for aged maps + add them to ZIP.
    if has_aged_results:
        albedo_aged_pil = rgb_numpy_to_pil(albedo_aged)
        normal_aged_pil = rgb_numpy_to_pil(normal_aged)
        roughness_aged_pil = gray_numpy_to_pil(roughness_aged)

        albedo_aged_bytes = pil_to_png_bytes(albedo_aged_pil)
        normal_aged_bytes = pil_to_png_bytes(normal_aged_pil)
        roughness_aged_bytes = pil_to_png_bytes(roughness_aged_pil)

        a1, a2, a3 = st.columns(3)

        with a1:
            st.download_button(
                label="Download albedo_aged.png",
                data=albedo_aged_bytes,
                file_name="albedo_aged.png",
                mime="image/png",
            )

        with a2:
            st.download_button(
                label="Download normal_aged.png",
                data=normal_aged_bytes,
                file_name="normal_aged.png",
                mime="image/png",
            )

        with a3:
            st.download_button(
                label="Download roughness_aged.png",
                data=roughness_aged_bytes,
                file_name="roughness_aged.png",
                mime="image/png",
            )

        zip_files["albedo_aged.png"] = albedo_aged_bytes
        zip_files["normal_aged.png"] = normal_aged_bytes
        zip_files["roughness_aged.png"] = roughness_aged_bytes

    # Full ZIP download.
    zip_bytes = build_results_zip(zip_files)

    st.download_button(
        label="Download all results (.zip)",
        data=zip_bytes,
        file_name="pbr_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
