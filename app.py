import streamlit as st
import torch
import timm
import faiss
import numpy as np
import os
import glob
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Setup
st.set_page_config(page_title="DINOv2 Image Similarity", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    model.eval().to(device)
    return model

@st.cache_data
import tarfile
import requests

@st.cache_data
def prepare_dataset():
    os.makedirs("dataset", exist_ok=True)
    flowers_dir = "dataset/flowers"
    tar_path = "102flowers.tgz"

    # Download if not already
    if not os.path.exists(flowers_dir):
        st.info("Downloading flower dataset...")
        url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract
        with tarfile.open(tar_path) as tar:
            tar.extractall("dataset")
        os.rename("dataset/jpg", flowers_dir)

    image_paths = sorted(glob.glob(f"{flowers_dir}/*.jpg"))
    vectors = []
    valid_paths = []

    for path in image_paths:
        try:
            vec = extract_features(path)
            if vec is not None and vec.shape[0] > 0:
                vectors.append(vec)
                valid_paths.append(path)
        except Exception as e:
            print(f"[ERROR] Skipped {path} - {e}")

    if not vectors:
        raise ValueError("No valid image features extracted. Dataset might be corrupted.")

    image_vectors = np.vstack(vectors).astype("float32")
    return valid_paths, image_vectors



@st.cache_resource
def build_index(image_vectors):
    index = faiss.IndexFlatL2(image_vectors.shape[1])
    index.add(image_vectors)
    return index

def extract_features(image_input):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        return features[:, 0].squeeze().cpu().numpy()

# Load model and dataset
model = load_model()
image_paths, image_vectors = prepare_dataset()
index = build_index(image_vectors)

# Streamlit UI
st.title("🔍 DINOv2 Image Similarity Search")
st.write("Upload an image to find similar images from the Oxford Flowers dataset.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    query_image = Image.open(uploaded_file)
    st.image(query_image, caption="Uploaded Image", use_column_width=True)

    # Feature extraction and similarity search
    query_vec = extract_features(query_image).reshape(1, -1)
    D, I = index.search(query_vec, k=5)

    st.subheader("🔎 Top-5 Most Similar Images")
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(image_paths):
            continue
        img = Image.open(image_paths[idx])
        ax[i].imshow(img)
        ax[i].set_title(f"Rank {i+1}")
        ax[i].axis("off")
    st.pyplot(fig)
