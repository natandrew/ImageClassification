from pathlib import Path
import os
import io
import json
import tempfile
import requests
import streamlit as st
from PIL import Image
import numpy as np

# -------------------------
# Basic config / constants
# -------------------------
DEMO_DIR = Path("demo_assets")
DEMO_JSON = DEMO_DIR / "demo_preds.json"
DEFAULT_CKPT = Path("./ckpt/best.pth")

# CIFAR-10 class names (so we don't need torchvision just to show labels)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -------------------------
# Safe conditional imports
# -------------------------
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision
    from torchvision import models
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    # only show warning once UI initialized (below) via st.warning

# -------------------------
# Utility helpers
# -------------------------
def humanize_probs(lst):
    return [f"{p*100:.2f}%" for (_, p) in lst]

def save_uploaded_file(uploaded_file) -> Path:
    """Save a streamlit uploaded file to a temporary path and return it."""
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    tf.close()
    return Path(tf.name)

# -------------------------
# Model-related functions (only defined if torch available)
# -------------------------
if TORCH_AVAILABLE:
    # model factory matches the notebook: resnet18 adapted for 32x32
    def make_model(num_classes=10, pretrained=False):
        model = models.resnet18(pretrained=pretrained)
        # adapt first conv and remove maxpool for CIFAR-10 32x32 images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    @st.cache_resource
    def load_model_cached(checkpoint_path: str | None = None, use_pretrained_backbone: bool = False):
        """
        Load and cache the model in the Streamlit instance. Returns tuple (model, device).
        If checkpoint_path is None or doesn't exist, returns a freshly initialized model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make_model(num_classes=10, pretrained=use_pretrained_backbone)
        model = model.to(device)
        model.eval()
        if checkpoint_path:
            try:
                ckpt = torch.load(checkpoint_path, map_location=device)
                state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state)
                st.info(f"Loaded checkpoint from: {checkpoint_path}")
            except Exception as e:
                st.warning(f"Failed to load checkpoint: {e}")
        else:
            st.info("No checkpoint path provided. Using untrained model (or ImageNet backbone if requested).")
        return model, device

    def predict_with_model(model, device, image_pil: Image.Image, topk: int = 3):
        preprocess = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        img_t = preprocess(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
            probs = torch.softmax(out, dim=1).cpu().squeeze(0).numpy()
        topk_idx = probs.argsort()[::-1][:topk]
        return [(CLASS_NAMES[int(i)], float(probs[int(i)])) for i in topk_idx]

else:
    # placeholders to avoid NameError if referenced
    def load_model_cached(*a, **kw):
        return None, None
    def predict_with_model(*a, **kw):
        raise RuntimeError("Torch not available")

# -------------------------
# Demo-mode helpers
# -------------------------
def load_demo_preds():
    if DEMO_JSON.exists():
        try:
            with open(DEMO_JSON, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to read demo predictions JSON: {e}")
            return {}
    else:
        return {}

def list_demo_images():
    """Return list of image filenames in demo assets (ordered)."""
    if not DEMO_DIR.exists():
        return []
    imgs = [p.name for p in sorted(DEMO_DIR.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return imgs

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="ImageClassification — Streamlit demo", layout="centered")
st.title("ImageClassification — Streamlit demo")

if not TORCH_AVAILABLE:
    st.warning("PyTorch / torchvision not available in this environment — running in DEMO mode with precomputed predictions. "
               "To enable real model inference, run the app locally or deploy to an environment that can install PyTorch.")

st.markdown(
    """
This demo shows predictions for the ResNet-18 model used in the notebook.
- If a compatible PyTorch is available, the app runs real model inference.
- Otherwise, the app uses precomputed demo assets (images + predictions) committed to `demo_assets/`.
"""
)

# Sidebar - model & checkpoint controls
st.sidebar.header("Model / checkpoint")

use_imagenet_backbone = st.sidebar.checkbox("Use ImageNet backbone weights (pretrained)", value=False)
use_default_ckpt = st.sidebar.checkbox(f"Load default checkpoint at {DEFAULT_CKPT}", value=True)

uploaded_ckpt = st.sidebar.file_uploader("Or upload a checkpoint (.pth)", type=["pt", "pth", "bin"])
# optional: a public release URL to try downloading from
release_ckpt_url = st.sidebar.text_input("Optional: download checkpoint from URL", value="")

# Try to determine which checkpoint path to use (local path or uploaded)
checkpoint_path = None
if uploaded_ckpt is not None:
    uploaded_path = save_uploaded_file(uploaded_ckpt)
    if uploaded_path:
        checkpoint_path = str(uploaded_path)
elif use_default_ckpt and DEFAULT_CKPT.exists():
    checkpoint_path = str(DEFAULT_CKPT)
elif release_ckpt_url:
    # attempt to download once to ckpt dir
    ckpt_dir = Path("./ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dst = ckpt_dir / Path(release_ckpt_url).name
    if not dst.exists():
        try:
            with requests.get(release_ckpt_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            checkpoint_path = str(dst)
            st.sidebar.success("Downloaded checkpoint from URL.")
        except Exception as e:
            st.sidebar.warning(f"Failed to download checkpoint: {e}")
    else:
        checkpoint_path = str(dst)

# Load model if torch available
model = None
device = None
if TORCH_AVAILABLE:
    model, device = load_model_cached(checkpoint_path if checkpoint_path else None, use_pretrained_backbone=use_imagenet_backbone)
else:
    # ensure model variable exists but remains None
    model = None
    device = None

# Main UI — input image selection
st.header("Input image")
col1, col2 = st.columns([1, 1])

uploaded_img = None
picked_demo = None
demo_preds = {}

with col1:
    uploaded_img_file = st.file_uploader("Upload an image (PNG/JPG) — optional", type=["png", "jpg", "jpeg"])
    if uploaded_img_file is not None:
        try:
            uploaded_img = Image.open(uploaded_img_file).convert("RGB")
            st.image(uploaded_img, caption="Uploaded image", width=256)
        except Exception:
            st.error("Failed to open the uploaded image.")

with col2:
    # If demo assets available, show picker
    demo_images = list_demo_images()
    if demo_images:
        demo_preds = load_demo_preds()
        picked_demo = st.selectbox("Or pick a demo image (committed to repo)", ["-- none --"] + demo_images)
        if picked_demo and picked_demo != "-- none --":
            demo_img_path = DEMO_DIR / picked_demo
            try:
                img = Image.open(demo_img_path).convert("RGB")
                st.image(img, caption=f"Demo: {picked_demo}", width=256)
            except Exception as e:
                st.warning(f"Could not open demo image: {e}")
    else:
        st.info("No `demo_assets/` directory found in the repo. Add demo images + demo_preds.json for demo-mode UI.")

# Choose source image: priority uploaded -> demo pick -> nothing
input_image = None
if uploaded_img is not None:
    input_image = uploaded_img
elif picked_demo and picked_demo != "-- none --":
    input_image = Image.open(DEMO_DIR / picked_demo).convert("RGB")

# Top-k slider
topk = st.slider("Top-k predictions", min_value=1, max_value=10, value=3)

# Action button
if st.button("Run inference"):
    if input_image is None:
        st.warning("Please upload an image or pick a demo image to run inference.")
    else:
        if TORCH_AVAILABLE and model is not None:
            # real inference
            try:
                results = predict_with_model(model, device, input_image, topk=topk)
                st.success("Inference done (model run).")
                for cls, p in results:
                    st.write(f"**{cls}** — {p*100:.2f}%")
            except Exception as e:
                st.error(f"Model inference failed: {e}")
        else:
            # Demo fallback path
            if picked_demo and picked_demo in demo_preds:
                st.info("Demo-mode: showing precomputed predictions for the selected demo image.")
                for obj in demo_preds[picked_demo][:topk]:
                    st.write(f"**{obj['class']}** — {obj['p']*100:.2f}%")
            else:
                st.warning("Torch not available and no precomputed predictions found for this image. "
                           "You can upload a checkpoint in the sidebar and redeploy to enable real inference.")

