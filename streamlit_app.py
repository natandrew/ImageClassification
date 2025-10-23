import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
from torchvision import models
from PIL import Image
import io
import os
import tempfile
import numpy as np

# ----------------- Config -----------------
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
DEFAULT_CHECKPOINT = "./ckpt/best.pth"
NUM_CLASSES = 10
CLASS_NAMES = torchvision.datasets.CIFAR10(root="./data", train=False, download=True).classes

# ---------- Helpers / model factory ----------

def make_model(num_classes=10, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    # adapt for CIFAR-10 (32x32)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource
def load_model(checkpoint_path: str | None, use_pretrained_backbone: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(num_classes=NUM_CLASSES, pretrained=use_pretrained_backbone)
    model = model.to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            # checkpoint may be a dict with 'model_state' or the bare state_dict
            state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            st.info(f"Loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            st.warning(f"Failed to load checkpoint: {e}")
    else:
        if checkpoint_path:
            st.warning(f"Checkpoint not found at {checkpoint_path}. Using randomly initialized weights (or ImageNet backbone if requested).")
        else:
            st.info("No checkpoint path provided — using model initialized from torchvision (possibly ImageNet weights if selected).")

    return model


# ---------- Preprocessing & inference ----------

preprocess = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def predict(model, image: Image.Image, topk: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1).cpu().squeeze(0).numpy()
    topk_idx = probs.argsort()[::-1][:topk]
    return [(CLASS_NAMES[i], float(probs[i])) for i in topk_idx]


# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="ImageClassification — Streamlit demo", layout="centered")
st.title("ImageClassification — Streamlit demo")

st.markdown("""
This demo app shows inference for the ResNet-18 model used in the notebook. You can:
- upload a model checkpoint (`ckpt/best.pth`) or let the app use the default path
- upload an image (any size) or draw from the CIFAR-10 test set
""")

# Sidebar: checkpoint options
st.sidebar.header("Model / checkpoint")
use_imagenet_backbone = st.sidebar.checkbox("Use ImageNet backbone weights (pretrained)", value=False)
use_default_ckpt = st.sidebar.checkbox(f"Load default checkpoint at {DEFAULT_CHECKPOINT}", value=True)
uploaded_ckpt = st.sidebar.file_uploader("Or upload a checkpoint (.pth)", type=["pt", "pth", "bin"]) 

checkpoint_path = None
if uploaded_ckpt is not None:
    # write to a temp file so torch.load can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tfile.write(uploaded_ckpt.getbuffer())
    tfile.flush()
    checkpoint_path = tfile.name
elif use_default_ckpt and os.path.exists(DEFAULT_CHECKPOINT):
    checkpoint_path = DEFAULT_CHECKPOINT

# load model (cached)
model = load_model(checkpoint_path, use_pretrained_backbone=use_imagenet_backbone)

# Input image selection
st.header("Input image")
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_img = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"]) 
    if uploaded_img is not None:
        try:
            image = Image.open(uploaded_img).convert("RGB")
        except Exception:
            st.error("Failed to open image. Try a different file.")
            image = None
    else:
        image = None

with col2:
    if st.button("Pick random CIFAR-10 test image"):
        ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
        idx = np.random.randint(len(ds))
        img, label = ds[idx]
        image = img
        st.write(f"Random label (ground-truth): {CLASS_NAMES[label]}")

if image is None:
    st.info("Upload an image or pick a random CIFAR-10 example to get predictions.")
else:
    st.image(image, caption="Input image", width=256)

    # run prediction
    topk = st.slider("Top-k", min_value=1, max_value=10, value=3)
    if st.button("Run inference"):
        with st.spinner("Predicting..."):
            results = predict(model, image, topk=topk)
        st.success("Done")
        for cls, p in results:
            st.write(f"**{cls}** — {p*100:.2f}%")
