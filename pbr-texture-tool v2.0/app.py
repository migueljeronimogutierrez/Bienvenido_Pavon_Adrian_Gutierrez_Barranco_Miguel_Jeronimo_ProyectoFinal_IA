from __future__ import annotations

import base64
import hashlib
import math
from io import BytesIO

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2 as _cv2
from PIL import Image

from app.cyclegan_infer import CycleGANInferencer
from app.deeppbr_infer import DeepPBRInferencer
from app.geometry import warp_perspective_from_points
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
# -----------------------------------------------------------------

st.set_page_config(
    page_title="PBR Texture Tool",
    page_icon="🧱",
    layout="wide",
)


# -----------------------------------------------------------------
# Session-state helpers.
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
        # Perspective correction state
        "warp_points": None,        # list of 4 [x,y] in original image coords
        "warped_image": None,       # np.ndarray result of warp, or None
        "warp_confirmed": False,    # True once user clicks "Apply"
        "pending_points_json": "",  # raw JSON string coming from JS canvas
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_inference_results() -> None:
    # NOTE: intentionally does NOT touch source_image_hash,
    # warp_confirmed or warped_image — those must survive reruns.
    inference_keys = [
        "results_ready",
        "normal_base",
        "roughness_base",
        "albedo_aged",
        "normal_aged",
        "roughness_aged",
        "last_run_settings",
    ]
    for key in inference_keys:
        if key == "results_ready":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

    st.session_state["compare_option"] = "Albedo: original vs aged"


def clear_warp_state() -> None:
    st.session_state["warp_points"] = None
    st.session_state["warped_image"] = None
    st.session_state["warp_confirmed"] = False
    st.session_state["pending_points_json"] = ""


def get_current_settings(
    use_real_tiling: bool,
    zoom_factor: float,
    overlap: int,
    apply_aging: bool,
    aging_intensity: float,
) -> dict[str, object]:
    return {
        "use_real_tiling": use_real_tiling,
        "zoom_factor": float(zoom_factor),
        "overlap": int(overlap),
        "apply_aging": bool(apply_aging),
        "aging_intensity": float(aging_intensity),
    }


init_session_state()


# -----------------------------------------------------------------
# Model loading.
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
# Utilities.
# -----------------------------------------------------------------

def numpy_to_base64_png(img: np.ndarray) -> str:
    if img.ndim == 2:
        pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(img.astype(np.uint8), mode="RGB")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def estimate_processing_time(
    h: int,
    w: int,
    tile_size: int,
    overlap: int,
    zoom_factor: float,
    apply_aging: bool,
) -> tuple[int, float]:
    zh = max(1, int(round(h * zoom_factor)))
    zw = max(1, int(round(w * zoom_factor)))
    stride = tile_size - overlap
    target_h = max(zh, tile_size)
    target_w = max(zw, tile_size)
    if target_h > tile_size:
        n_h = math.ceil((target_h - tile_size) / stride)
        target_h = tile_size + n_h * stride
    if target_w > tile_size:
        n_w = math.ceil((target_w - tile_size) / stride)
        target_w = tile_size + n_w * stride
    tiles_y = 1 + (target_h - tile_size) // stride if target_h > tile_size else 1
    tiles_x = 1 + (target_w - tile_size) // stride if target_w > tile_size else 1
    n_tiles = tiles_y * tiles_x
    DEEPPBR_SPT = 1.3
    CYCLEGAN_SPT = 2.8
    seconds = n_tiles * DEEPPBR_SPT
    if apply_aging:
        seconds += n_tiles * CYCLEGAN_SPT
    return n_tiles, seconds


def render_comparison_slider(
    img_left: np.ndarray,
    img_right: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Aged",
    height: int = 450,
) -> None:
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
# Perspective correction canvas.
#
# Communication JS → Python via st.query_params:
#   On mouseup/touchend the JS sets  ?warp=x0,y0,x1,y1,x2,y2,x3,y3
#   in the parent window URL using history.replaceState (no reload).
#   Streamlit detects the query param change on the next interaction
#   and Python reads it with st.query_params.get("warp").
#
#   Because history.replaceState alone doesn't trigger a Streamlit
#   rerun, we also POST to /_stcore/stream via fetch to force one.
#   The cleanest alternative that works without hacks is to use a
#   small "Apply" button flow: the JS stores the points in
#   sessionStorage and a Streamlit st.button triggers the read.
#   We use that approach here as it is 100 % reliable across all
#   Streamlit versions.
# -----------------------------------------------------------------

def render_perspective_canvas(
    image_rgb: np.ndarray,
    canvas_height: int = 480,
) -> None:
    """
    Renders the interactive 4-point perspective correction canvas.

    On every mouseup/touchend the JS writes the current handle
    coordinates into the parent window URL as ?warp=x0,y0,...
    using history.replaceState (synchronous, no page reload).
    Python reads st.query_params["warp"] on the next rerun,
    triggered by the "Update preview" st.button outside the iframe.

    Coordinate string format: "x0,y0,x1,y1,x2,y2,x3,y3"  (integers, image-pixel space)
    """
    b64 = numpy_to_base64_png(image_rgb)
    ih, iw = image_rgb.shape[:2]

    margin_x = int(iw * 0.10)
    margin_y = int(ih * 0.10)
    default_pts = [
        [margin_x,        margin_y],
        [iw - margin_x,   margin_y],
        [iw - margin_x,   ih - margin_y],
        [margin_x,        ih - margin_y],
    ]

    pts_init = st.session_state["warp_points"] if st.session_state["warp_points"] is not None else default_pts
    pts_json = str(pts_init).replace(" ", "")

    html = f"""
<style>
  #wc {{ position:relative; width:100%; user-select:none; }}
  #cv {{ width:100%; height:{canvas_height}px; display:block; border-radius:8px; cursor:crosshair; }}
  #wh {{ margin-top:6px; font-size:12px; color:#aaa; font-family:sans-serif; }}
</style>
<div id="wc"><canvas id="cv"></canvas>
<p id="wh">Drag the handles to frame the surface, then click <strong>↩ Update preview</strong> below.</p></div>
<script>
(function(){{
  const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
  const IW={iw},IH={ih};
  const img=new Image(); img.src='data:image/png;base64,{b64}';
  let pts={pts_json}, drag=-1;
  const R=11;
  const i2c=(x,y)=>[x*cv.width/IW, y*cv.height/IH];
  const c2i=(x,y)=>[x*IW/cv.width, y*IH/cv.height];
  function rsz(){{ const r=cv.getBoundingClientRect(); cv.width=r.width||800; cv.height=r.height||{canvas_height}; draw(); }}
  function draw(){{
    if(!img.complete) return;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(img,0,0,cv.width,cv.height);
    const [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]=pts.map(p=>i2c(...p));
    ctx.save(); ctx.beginPath();
    ctx.rect(0,0,cv.width,cv.height);
    ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.lineTo(x2,y2);ctx.lineTo(x3,y3);ctx.closePath();
    ctx.fillStyle='rgba(0,0,0,0.5)'; ctx.fill('evenodd'); ctx.restore();
    ctx.beginPath(); ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.lineTo(x2,y2);ctx.lineTo(x3,y3);ctx.closePath();
    ctx.strokeStyle='#ff4b4b'; ctx.lineWidth=2; ctx.stroke();
    [[x0,y0],[x1,y1],[x2,y2],[x3,y3]].forEach(([cx,cy])=>{{
      ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2);
      ctx.fillStyle='#ff4b4b'; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();
    }});
  }}
  function hit(cx,cy){{ for(let i=0;i<4;i++){{ const [hx,hy]=i2c(...pts[i]); if(Math.hypot(cx-hx,cy-hy)<=R+5) return i; }} return -1; }}
  function epos(e){{ const r=cv.getBoundingClientRect(); return e.touches?[e.touches[0].clientX-r.left,e.touches[0].clientY-r.top]:[e.clientX-r.left,e.clientY-r.top]; }}
  function push(){{
    const flat=pts.map(p=>[Math.round(p[0]),Math.round(p[1])].join(',')).join(',');
    try{{ const u=new URL(window.parent.location.href); u.searchParams.set('warp',flat); window.parent.history.replaceState(null,'',u.toString()); }}catch(e){{}}
  }}
  cv.addEventListener('mousedown',e=>{{ drag=hit(...epos(e)); }});
  window.addEventListener('mouseup',()=>{{ if(drag>=0) push(); drag=-1; }});
  cv.addEventListener('mousemove',e=>{{
    if(drag<0) return;
    const [cx,cy]=epos(e);
    pts[drag]=c2i(Math.max(0,Math.min(cv.width,cx)),Math.max(0,Math.min(cv.height,cy)));
    draw();
  }});
  cv.addEventListener('touchstart',e=>{{e.preventDefault();drag=hit(...epos(e));}},{{passive:false}});
  window.addEventListener('touchend',()=>{{ if(drag>=0) push(); drag=-1; }});
  cv.addEventListener('touchmove',e=>{{
    e.preventDefault(); if(drag<0) return;
    const [cx,cy]=epos(e);
    pts[drag]=c2i(Math.max(0,Math.min(cv.width,cx)),Math.max(0,Math.min(cv.height,cy)));
    draw();
  }},{{passive:false}});
  img.onload=rsz; window.addEventListener('resize',rsz); if(img.complete) rsz();
}})();
</script>"""
    components.html(html, height=canvas_height + 40, scrolling=False)


def parse_points_from_string(raw: str) -> list[list[int]] | None:
    """
    Parses "x0,y0,x1,y1,x2,y2,x3,y3" into [[x0,y0],[x1,y1],[x2,y2],[x3,y3]].
    Returns None if the string is empty or malformed.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        values = [int(float(v)) for v in raw.split(",")]
        if len(values) != 8:
            return None
        return [[values[i * 2], values[i * 2 + 1]] for i in range(4)]
    except (ValueError, IndexError):
        return None



# -----------------------------------------------------------------
# PBR 3D Viewer.
#
# Renders a Three.js scene inside a components.html iframe.
# A sphere with MeshStandardMaterial displays the PBR maps:
#   - albedo  → material.map
#   - normal  → material.normalMap  (Three.js expects OpenGL convention)
#   - roughness → material.roughnessMap
#
# The user can orbit the light (drag) and zoom (scroll).
# Textures are passed as base64 PNG strings embedded in the HTML.
# -----------------------------------------------------------------

def render_pbr_viewer(
    albedo: np.ndarray,
    normal: np.ndarray,
    roughness: np.ndarray,
    viewer_height: int = 500,
    label: str = "Base",
) -> None:
    """
    Renders an interactive 3D PBR sphere viewer using Three.js r128.

    Parameters
    ----------
    albedo : np.ndarray  (H, W, 3) uint8  — colour map
    normal : np.ndarray  (H, W, 3) uint8  — tangent-space normal map
    roughness : np.ndarray  (H, W) or (H, W, 1) uint8  — roughness map
    viewer_height : int  — pixel height of the iframe
    label : str  — shown in the top-left corner of the viewer
    """
    b64_albedo   = numpy_to_base64_png(albedo)
    b64_normal   = numpy_to_base64_png(normal)

    # Roughness must be RGB for Three.js texture loader.
    if roughness.ndim == 2:
        rough_rgb = np.stack([roughness, roughness, roughness], axis=-1)
    else:
        rough_rgb = np.concatenate([roughness, roughness, roughness], axis=-1)
    b64_roughness = numpy_to_base64_png(rough_rgb)

    html = f"""
<!DOCTYPE html>
<html style="margin:0;padding:0;background:#1a1a1a;">
<body style="margin:0;padding:0;overflow:hidden;">

<canvas id="c" style="display:block;width:100%;height:{viewer_height}px;"></canvas>

<div id="lbl" style="
  position:absolute;top:10px;left:12px;
  background:rgba(0,0,0,0.55);color:#fff;
  padding:3px 10px;border-radius:4px;
  font-size:12px;font-family:sans-serif;
  pointer-events:none;">
  {label} &nbsp;·&nbsp; drag to orbit light &nbsp;·&nbsp; scroll to zoom
</div>

<div id="controls" style="
  position:absolute;bottom:14px;left:12px;right:12px;
  display:flex;gap:20px;align-items:center;
  background:rgba(0,0,0,0.55);
  padding:8px 14px;border-radius:6px;
  font-size:12px;font-family:sans-serif;color:#ccc;">
  <span style="white-space:nowrap;">Normal strength</span>
  <input id="sl-normal" type="range" min="0" max="400" value="100"
    style="flex:1;accent-color:#ff4b4b;cursor:pointer;">
  <span id="val-normal" style="width:35px;text-align:right;">1.0×</span>
  <span style="white-space:nowrap;margin-left:10px;">Roughness mult.</span>
  <input id="sl-rough" type="range" min="0" max="300" value="100"
    style="flex:1;accent-color:#61afef;cursor:pointer;">
  <span id="val-rough" style="width:35px;text-align:right;">1.0×</span>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){{
  // ── Renderer ────────────────────────────────────────────────
  const canvas = document.getElementById('c');
  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(canvas.clientWidth, {viewer_height});
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  // ── Scene & Camera ──────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / {viewer_height}, 0.1, 100);
  camera.position.set(0, 0, 3);

  // ── Textures ────────────────────────────────────────────────
  const loader = new THREE.TextureLoader();

  function loadB64(b64, encoding) {{
    const tex = loader.load('data:image/png;base64,' + b64);
    tex.encoding  = encoding || THREE.LinearEncoding;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(2, 2);   // tile twice for better detail on sphere
    return tex;
  }}

  const texAlbedo    = loadB64('{b64_albedo}',   THREE.sRGBEncoding);
  const texNormal    = loadB64('{b64_normal}',   THREE.LinearEncoding);
  const texRoughness = loadB64('{b64_roughness}', THREE.LinearEncoding);

  // ── Material & Sphere ───────────────────────────────────────
  const material = new THREE.MeshStandardMaterial({{
    map:          texAlbedo,
    normalMap:    texNormal,
    normalScale:  new THREE.Vector2(1, 1),
    roughnessMap: texRoughness,
    roughness:    1.0,
    metalness:    0.0,
  }});

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(1, 64, 64),
    material
  );
  scene.add(sphere);

  // ── Lights ──────────────────────────────────────────────────
  // Ambient: soft fill so shadowed areas aren't pitch black.
  const ambient = new THREE.AmbientLight(0xffffff, 0.25);
  scene.add(ambient);

  // Main point light — orbits with mouse drag.
  const pointLight = new THREE.PointLight(0xffffff, 2.5, 20);
  pointLight.position.set(2, 2, 2);
  scene.add(pointLight);

  // Small rim light from the back for depth.
  const rimLight = new THREE.DirectionalLight(0x8899ff, 0.8);
  rimLight.position.set(-2, -1, -2);
  scene.add(rimLight);

  // ── Material controls (sliders) ─────────────────────────────
  const slNormal = document.getElementById('sl-normal');
  const slRough  = document.getElementById('sl-rough');
  const valNormal = document.getElementById('val-normal');
  const valRough  = document.getElementById('val-rough');

  slNormal.addEventListener('input', () => {{
    const v = slNormal.value / 100;
    material.normalScale.set(v, v);
    material.needsUpdate = true;
    valNormal.textContent = v.toFixed(1) + '×';
  }});

  slRough.addEventListener('input', () => {{
    const v = slRough.value / 100;
    material.roughness = v;
    material.needsUpdate = true;
    valRough.textContent = v.toFixed(1) + '×';
  }});

  // ── Light orbit (drag) ──────────────────────────────────────
  let lightTheta = Math.PI / 4;   // horizontal angle
  let lightPhi   = Math.PI / 4;   // vertical angle
  const LIGHT_R  = 3.5;

  function updateLight() {{
    pointLight.position.set(
      LIGHT_R * Math.sin(lightPhi) * Math.cos(lightTheta),
      LIGHT_R * Math.cos(lightPhi),
      LIGHT_R * Math.sin(lightPhi) * Math.sin(lightTheta)
    );
  }}
  updateLight();

  let dragging = false, lastX = 0, lastY = 0;

  canvas.addEventListener('mousedown',  e => {{ dragging=true;  lastX=e.clientX; lastY=e.clientY; }});
  window.addEventListener('mouseup',    ()  => {{ dragging=false; }});
  window.addEventListener('mousemove',  e  => {{
    if (!dragging) return;
    lightTheta -= (e.clientX - lastX) * 0.01;
    lightPhi    = Math.max(0.1, Math.min(Math.PI-0.1, lightPhi + (e.clientY - lastY) * 0.01));
    lastX=e.clientX; lastY=e.clientY;
    updateLight();
  }});

  // Touch support for orbit.
  let lastTX=0, lastTY=0;
  canvas.addEventListener('touchstart', e=>{{ lastTX=e.touches[0].clientX; lastTY=e.touches[0].clientY; }},{{passive:true}});
  canvas.addEventListener('touchmove',  e=>{{
    e.preventDefault();
    lightTheta -= (e.touches[0].clientX - lastTX) * 0.01;
    lightPhi    = Math.max(0.1, Math.min(Math.PI-0.1, lightPhi + (e.touches[0].clientY - lastTY) * 0.01));
    lastTX=e.touches[0].clientX; lastTY=e.touches[0].clientY;
    updateLight();
  }},{{passive:false}});

  // ── Zoom (scroll) ───────────────────────────────────────────
  canvas.addEventListener('wheel', e => {{
    camera.position.z = Math.max(1.5, Math.min(6, camera.position.z + e.deltaY * 0.005));
    e.preventDefault();
  }}, {{passive:false}});

  // ── Render loop ─────────────────────────────────────────────
  function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }}
  animate();

  // ── Resize ──────────────────────────────────────────────────
  window.addEventListener('resize', () => {{
    const w = canvas.clientWidth;
    renderer.setSize(w, {viewer_height});
    camera.aspect = w / {viewer_height};
    camera.updateProjectionMatrix();
  }});
}})();
</script>
</body>
</html>
"""
    components.html(html, height=viewer_height + 4, scrolling=False)

# -----------------------------------------------------------------
# Title and description.
# -----------------------------------------------------------------

st.title("🧱 PBR Texture Tool")
st.caption(
    "Generate PBR maps (Normal + Roughness) from a photograph, "
    "with optional synthetic aging.  "
    "Domain: Stone-family materials."
)


# -----------------------------------------------------------------
# Tips banner.
# -----------------------------------------------------------------

st.info(
    "**Tips for best results:**  \n"
    "• Use frontal shots with even, diffuse lighting.  \n"
    "• Avoid extreme shadows, reflections or out-of-focus areas.  \n"
    "• Stick to stone-family materials (brick, concrete, asphalt, stone, plaster, polished marble, etc).  \n"
    "• Very high resolutions will increase processing time significantly."
)


# -----------------------------------------------------------------
# Sidebar.
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
        "Zoom factor controls how much of the image each 256×256 tile 'sees' before inference.  \n"
        "The models were trained on 256×256 patches cropped from 1K textures, so each tile is expected to contain recognisable material structure — full bricks, visible pores, clear joints.  \n"
        "If you upload a very high-resolution image (e.g. 4K) without reducing zoom, each tile will cover only a tiny fragment of the surface. The model may not recognise the material at all, producing incoherent maps.  \n"
        "Reducing the zoom downscales the image before tiling, so each patch sees a wider, more complete region — closer to what the model learned during training.  \n"
        "• As a rule of thumb: if your image is 2K, try 0.5. If it is 4K, try 0.25. At 1K, keep it at 1.0.  \n"
        "Note: lower zoom also speeds up inference, but that is a side effect, not the main purpose."
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
        "Controls how visible the aging effect is.  "
        "0.1 = subtle weathering blended with the original.  "
        "1.0 = full aging output from the model."
    ),
)

current_settings = get_current_settings(
    use_real_tiling=use_real_tiling,
    zoom_factor=zoom_factor,
    overlap=overlap,
    apply_aging=apply_aging,
    aging_intensity=aging_intensity,
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
# Read and hash uploaded image.
# -----------------------------------------------------------------

uploaded_bytes = uploaded_file.getvalue()
current_image_hash = hashlib.sha256(uploaded_bytes).hexdigest()

if st.session_state["source_image_hash"] != current_image_hash:
    clear_inference_results()
    clear_warp_state()
    st.session_state["source_image_hash"] = current_image_hash

input_pil = Image.open(BytesIO(uploaded_bytes))
input_rgb = pil_to_rgb_numpy(input_pil)
img_h, img_w = input_rgb.shape[:2]


# -----------------------------------------------------------------
# Validation warnings.
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
# Section 1 — Perspective correction (optional).
#
# The user can drag 4 corner handles to frame the material region.
# Once happy, clicking "Apply perspective correction" warps the
# image and uses the result as input for the PBR pipeline instead
# of the original upload.
#
# The section is collapsible so users who don't need it can skip it
# without visual clutter.
# -----------------------------------------------------------------

with st.expander("✂️ Perspective correction (optional)", expanded=False):
    st.markdown(
        "Drag the **four red handles** to frame the flat surface of the material.  \n"
        "Click **↩ Update preview** to see the corrected result and its tileable preview.  \n"
        "Then click **✅ Apply perspective correction** to use it as input for the pipeline."
    )

    # ------------------------------------------------------------------
    # The JS canvas writes ?warp=x0,y0,... into the URL on every
    # mouseup via history.replaceState.  The button below triggers
    # a Streamlit rerun so Python can read that param.
    if st.button("↩ Update preview", use_container_width=True, key="btn_update_preview"):
        warp_param = st.query_params.get("warp", "")
        if warp_param:
            pts = parse_points_from_string(warp_param)
            if pts is not None:
                st.session_state["warp_points"] = pts
                st.session_state["pending_points_json"] = warp_param
                pts_array = np.array(pts, dtype=np.float32)
                try:
                    st.session_state["warped_image"] = warp_perspective_from_points(
                        input_rgb, pts_array
                    )
                except Exception:
                    st.session_state["warped_image"] = None
            st.query_params.clear()

    # Draw the interactive canvas (after the button so the button
    # is rendered above the canvas in the page order).
    render_perspective_canvas(input_rgb, canvas_height=460)

    # Live preview.
    if st.session_state["warped_image"] is not None:
        warped = st.session_state["warped_image"]
        col_w, col_t = st.columns(2)
        with col_w:
            st.markdown("**Corrected crop**")
            st.image(warped, channels="RGB", use_container_width=True)
        with col_t:
            st.markdown("**Tileable 2×2 preview**")
            st.image(make_tile_preview_2x2(warped), channels="RGB", use_container_width=True)

    # Confirmation / reset buttons.
    col_apply, col_reset = st.columns([2, 1])
    with col_apply:
        apply_warp = st.button(
            "✅ Apply perspective correction",
            use_container_width=True,
            disabled=st.session_state["warped_image"] is None,
            help="Use the corrected crop as input for the PBR pipeline.",
        )
    with col_reset:
        reset_warp = st.button(
            "↺ Reset",
            use_container_width=True,
            help="Discard the correction and go back to the original image.",
        )

    if apply_warp and st.session_state["warped_image"] is not None:
        st.session_state["warp_confirmed"] = True
        clear_inference_results()
        st.rerun()

    if reset_warp:
        clear_warp_state()
        clear_inference_results()
        st.rerun()

    if st.session_state["warp_confirmed"]:
        st.info("✅ **Correction active** — the pipeline is using the corrected crop.")


# -----------------------------------------------------------------
# Determine the actual pipeline input:
# corrected image if warp was confirmed, original otherwise.
# -----------------------------------------------------------------

if st.session_state["warp_confirmed"] and st.session_state["warped_image"] is not None:
    pipeline_input_rgb = st.session_state["warped_image"]
    pipeline_source_label = "corrected"
else:
    pipeline_input_rgb = input_rgb
    pipeline_source_label = "original"

pipe_h, pipe_w = pipeline_input_rgb.shape[:2]


# -----------------------------------------------------------------
# Section 2 — Original image and tileable preview.
# -----------------------------------------------------------------

col_orig, col_tile = st.columns(2)
with col_orig:
    st.subheader("Original image")
    st.image(input_rgb, channels="RGB", use_container_width=True)
with col_tile:
    tile_preview = make_tile_preview_2x2(pipeline_input_rgb)
    label = "Tileable 2×2 preview"
    if pipeline_source_label == "corrected":
        label += " (corrected)"
    st.subheader(
        label,
        help=(
            "A 2×2 mosaic of the texture. Use it to visually check "
            "whether the pattern repeats seamlessly — if the edges "
            "don't match, the material may produce visible seams "
            "when tiled in a 3D engine."
        ),
    )
    st.image(tile_preview, channels="RGB", use_container_width=True)


# -----------------------------------------------------------------
# Time estimation.
# -----------------------------------------------------------------

if use_real_tiling:
    n_tiles, est_seconds = estimate_processing_time(
        h=pipe_h,
        w=pipe_w,
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
# Generate button.
# -----------------------------------------------------------------

generate_clicked = st.button("Generate PBR maps", type="primary", use_container_width=True)

if generate_clicked:

    deeppbr = load_deeppbr_model()

    if use_real_tiling:
        with st.spinner("Generating base maps with real patching (this may take a few minutes)..."):
            normal_base, roughness_base = process_large_image_with_deeppbr(
                image_rgb=pipeline_input_rgb,
                inferencer=deeppbr,
                tile_size=256,
                overlap=overlap,
                zoom_factor=zoom_factor,
            )
    else:
        with st.spinner("Generating base maps (fast mode — single 256x256 pass)..."):
            normal_base, roughness_base = deeppbr.predict_from_rgb(pipeline_input_rgb)

    albedo_aged = None
    normal_aged = None
    roughness_aged = None

    if apply_aging:
        cyclegan = load_cyclegan_model()

        # When zoom_factor < 1.0 DeepPBR outputs maps at the zoomed
        # resolution, so the RGB input must be resized to match before
        # building the 7-channel stack for CycleGAN.
        _zh = max(1, int(round(pipe_h * zoom_factor)))
        _zw = max(1, int(round(pipe_w * zoom_factor)))
        _rgb_for_cyclegan = _cv2.resize(
            pipeline_input_rgb, (_zw, _zh), interpolation=_cv2.INTER_AREA
        )

        if use_real_tiling:
            with st.spinner(
                "Applying aging with real patching (this will take a while — "
                "each tile passes through CycleGAN individually)..."
            ):

                albedo_aged, normal_aged, roughness_aged = process_large_image_with_cyclegan(
                    rgb_uint8=_rgb_for_cyclegan,
                    normal_uint8=normal_base,
                    roughness_uint8=roughness_base,
                    inferencer=cyclegan,
                    intensity=aging_intensity,
                    tile_size=256,
                    overlap=overlap,
                    zoom_factor=1.0,  # zoom already applied above
                )
        else:
            with st.spinner("Applying aging (fast mode — single 256x256 pass)..."):
                albedo_aged, normal_aged, roughness_aged = cyclegan.predict_from_pbr(
                    rgb_uint8=_rgb_for_cyclegan,
                    normal_uint8=normal_base,
                    roughness_uint8=roughness_base,
                    intensity=aging_intensity,
                )

    st.session_state["normal_base"] = normal_base.copy()
    st.session_state["roughness_base"] = roughness_base.copy()
    st.session_state["albedo_aged"] = None if albedo_aged is None else albedo_aged.copy()
    st.session_state["normal_aged"] = None if normal_aged is None else normal_aged.copy()
    st.session_state["roughness_aged"] = None if roughness_aged is None else roughness_aged.copy()
    st.session_state["last_run_settings"] = current_settings
    st.session_state["results_ready"] = True


# -----------------------------------------------------------------
# Results.
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

    st.subheader("Base PBR results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Normal map**")
        st.image(normal_base, channels="RGB", use_container_width=True)

    with col2:
        st.markdown("**Roughness map**")
        st.image(roughness_base, clamp=True, use_container_width=True)

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

        st.subheader("Before / after comparison")

        if zoom_factor < 1.0:
            st.caption(
                "Zoom factor is below 1.0 — the aged maps were generated "
                "at a lower resolution than the original image. The comparison "
                "may not be perfectly pixel-aligned, but the visual result is correct."
            )

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
                img_left=pipeline_input_rgb,
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


    # --- 3D PBR Viewer ---

    st.subheader("3D PBR Viewer")
    st.caption(
        "Drag to orbit the light · Scroll to zoom · "
        "The sphere uses your generated maps as a real PBR material · Use the sliders to TEST the map exaggeration  \n"
        "Note: The normals and roughness map modifications will NOT be applied to the download."
    )

    viewer_tabs = ["Base maps"]
    if has_aged_results:
        viewer_tabs.append("Aged maps")

    active_tab = st.radio(
        "Show in viewer:",
        options=viewer_tabs,
        horizontal=True,
        key="viewer_tab",
        label_visibility="collapsed",
    ) if has_aged_results else "Base maps"

    if active_tab == "Base maps":
        render_pbr_viewer(
            albedo=pipeline_input_rgb,
            normal=normal_base,
            roughness=roughness_base,
            viewer_height=500,
            label="Base PBR",
        )
    else:
        render_pbr_viewer(
            albedo=albedo_aged,
            normal=normal_aged,
            roughness=roughness_aged,
            viewer_height=500,
            label="Aged PBR",
        )

    # --- Downloads ---

    st.subheader("Downloads")

    normal_base_pil = rgb_numpy_to_pil(normal_base)
    roughness_base_pil = gray_numpy_to_pil(roughness_base)
    normal_base_bytes = pil_to_png_bytes(normal_base_pil)
    roughness_base_bytes = pil_to_png_bytes(roughness_base_pil)

    zip_files = {
        "normal_base.png": normal_base_bytes,
        "roughness_base.png": roughness_base_bytes,
    }

    if has_aged_results:
        albedo_aged_pil   = rgb_numpy_to_pil(albedo_aged)
        normal_aged_pil   = rgb_numpy_to_pil(normal_aged)
        roughness_aged_pil = gray_numpy_to_pil(roughness_aged)
        albedo_aged_bytes   = pil_to_png_bytes(albedo_aged_pil)
        normal_aged_bytes   = pil_to_png_bytes(normal_aged_pil)
        roughness_aged_bytes = pil_to_png_bytes(roughness_aged_pil)

    # 3-column layout: Albedo | Normal | Roughness
    # Each column shows base button (always) + aged button (if available).
    ca, cn, cr = st.columns(3)

    with ca:
        st.markdown("**Albedo**")
        st.download_button(
            label="Original",
            data=pil_to_png_bytes(rgb_numpy_to_pil(pipeline_input_rgb)),
            file_name="albedo_original.png",
            mime="image/png",
            use_container_width=True,
        )
        if has_aged_results:
            st.download_button(
                label="Aged",
                data=albedo_aged_bytes,
                file_name="albedo_aged.png",
                mime="image/png",
                use_container_width=True,
            )

    with cn:
        st.markdown("**Normal map**")
        st.download_button(
            label="Base",
            data=normal_base_bytes,
            file_name="normal_base.png",
            mime="image/png",
            use_container_width=True,
        )
        if has_aged_results:
            st.download_button(
                label="Aged",
                data=normal_aged_bytes,
                file_name="normal_aged.png",
                mime="image/png",
                use_container_width=True,
            )

    with cr:
        st.markdown("**Roughness map**")
        st.download_button(
            label="Base",
            data=roughness_base_bytes,
            file_name="roughness_base.png",
            mime="image/png",
            use_container_width=True,
        )
        if has_aged_results:
            st.download_button(
                label="Aged",
                data=roughness_aged_bytes,
                file_name="roughness_aged.png",
                mime="image/png",
                use_container_width=True,
            )

    zip_files = {
        "albedo_original.png": pil_to_png_bytes(rgb_numpy_to_pil(pipeline_input_rgb)),
        "normal_base.png":     normal_base_bytes,
        "roughness_base.png":  roughness_base_bytes,
    }
    if has_aged_results:
        zip_files["albedo_aged.png"]    = albedo_aged_bytes
        zip_files["normal_aged.png"]    = normal_aged_bytes
        zip_files["roughness_aged.png"] = roughness_aged_bytes

    zip_bytes = build_results_zip(zip_files)

    st.download_button(
        label="Download all results (.zip)",
        data=zip_bytes,
        file_name="pbr_results.zip",
        mime="application/zip",
        use_container_width=True,
    )

# --- Technical inspection (optional) ---

    with st.expander("📊 Technical inspection — map histograms", expanded=False):
        st.caption(
            "Pixel value distribution for each generated map. "
            "Useful to detect clipping, flat outputs or unexpected shifts."
        )

        import matplotlib.pyplot as plt

        def plot_histogram(ax, data: np.ndarray, title: str, color: str) -> None:
            flat = data.flatten().astype(np.float32)
            ax.hist(flat, bins=128, color=color, alpha=0.85, edgecolor="none")
            ax.set_title(title, fontsize=9, color="#cccccc")
            ax.set_xlim(0, 255)
            ax.tick_params(colors="#888888", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.set_facecolor("#1e1e1e")

        maps_to_plot = [
            (normal_base[..., 0],   "Normal R (X)",  "#e06c75"),
            (normal_base[..., 1],   "Normal G (Y)",  "#98c379"),
            (normal_base[..., 2],   "Normal B (Z)",  "#61afef"),
            (roughness_base,        "Roughness base","#c678dd"),
        ]
        if has_aged_results:
            maps_to_plot += [
                (normal_aged[..., 0], "Normal aged R (X)", "#e06c75"),
                (normal_aged[..., 1], "Normal aged G (Y)", "#98c379"),
                (normal_aged[..., 2],  "Normal aged B (Z)", "#56b6c2"),
                (roughness_aged,       "Roughness aged",    "#e5c07b"),
            ]

        n_cols = 4
        n_rows = math.ceil(len(maps_to_plot) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 3, n_rows * 2.2),
            facecolor="#151515",
        )
        axes = np.array(axes).flatten()

        for i, (data, title, color) in enumerate(maps_to_plot):
            plot_histogram(axes[i], data, title, color)

        # Hide unused axes.
        for j in range(len(maps_to_plot), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)