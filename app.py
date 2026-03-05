import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from groq import Groq


GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(
        torch.load("poster_attractiveness_model.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Grad CAM
def generate_gradcam(model, image_tensor):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    output.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    fh.remove()
    bh.remove()

    return cam


def extract_features(image):

    img_np = np.array(image.resize((224,224)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray) / 255
    contrast = np.std(gray) / 128
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
    color_intensity = np.mean(img_np) / 255

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (224 * 224)

    color_variance = np.var(img_np) / (255**2)

    thresh = cv2.adaptiveThreshold(gray,255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,11,2)
    text_ratio = np.sum(thresh > 0) / (224 * 224)

    return {
        "Brightness": round(brightness * 10, 2),
        "Contrast": round(contrast * 10, 2),
        "Sharpness": round(min(sharpness, 1) * 10, 2),
        "Color Intensity": round(color_intensity * 10, 2),
        "Edge Density": round(edge_density * 10, 2),
        "Color Variance": round(color_variance * 10, 2),
        "Text Region Ratio": round(text_ratio * 10, 2),
    }


def predict(image):

    image_resized = image.resize((224, 224))
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        raw_score = torch.sigmoid(output).item()

    score = round(raw_score * 10, 2)
    heatmap = generate_gradcam(model, image_tensor)

    return score, heatmap, image_resized


# groq

def groq_chatbot(user_question, score, features):

    prompt = f"""
    You are an expert poster design analyst.

    The poster received an attractiveness score of {score}/10.

    Extracted features:
    {features}

    User question:
    {user_question}

    Give a professional explanation and improvement suggestions.
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a professional visual design expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return completion.choices[0].message.content


st.title("🎬 AI Poster Analyzer ")

uploaded = st.file_uploader("Upload Poster", type=["jpg", "png", "jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    score, heatmap, image_resized = predict(image)
    features = extract_features(image)

    st.subheader("Poster &  Heatmap")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_resized, caption="Input Poster ", use_container_width=True)

    with col2:
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )
        st.image(heatmap_color, caption="Grad-CAM Heatmap ", use_container_width=True)

    st.markdown("### 🔥 Heatmap Color Importance")
    st.markdown("""
    - 🔴 **Red** → Most important areas influencing score  
    - 🟡 **Yellow** → Moderately important  
    - 🔵 **Blue** → Less important  
    """)

    st.markdown("---")
    st.subheader(f"⭐ Predicted Attractiveness Score: {score}/10")

    st.markdown("### 📊 Poster Feature Analysis")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(features.keys(), features.values())
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Poster Quality Metrics (7 Key Features)")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🤖 AI Poster Assistant")

    user_input = st.text_input("Ask about this poster")

    if user_input:
        with st.spinner("Analyzing with AI..."):
            response = groq_chatbot(user_input, score, features)
        st.write(response)
