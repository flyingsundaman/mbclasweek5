import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(page_title="Multi-Model AI Demo", layout="wide")

# -----------------------------
# Sidebar Navigasi
# -----------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Model", ["Beranda", "Kucing vs Anjing", "Food-101", "Analisis Emosi"]
)


# -----------------------------
# Fungsi Preprocessing Gambar
# -----------------------------
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


# -----------------------------
# Ekstrak Feature Map
# -----------------------------
def get_feature_maps(model, input_tensor, layer_names):
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    return activation_model.predict(input_tensor)


# -----------------------------
# Load Model (dengan cache)
# -----------------------------
@st.cache_resource
def load_catdog_model():
    try:
        model_path = hf_hub_download(
            repo_id="kyomotodie/lasweek5", filename="catdog_au.h5"
        )
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model Kucing vs Anjing: {e}")
        return None


@st.cache_resource
def load_food_model():
    try:
        model_path = hf_hub_download(
            repo_id="kyomotodie/lasweek5", filename="model_food_101.h5"
        )
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model Food-101: {e}")
        return None


@st.cache_resource
def load_nlp_model():
    try:
        model_name = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Gagal memuat model NLP: {e}")
        return None, None


# -----------------------------
# Halaman: Beranda
# -----------------------------
if page == "Beranda":
    st.title("Multi-Model AI Demo")
    st.markdown("""
    Aplikasi ini menampilkan tiga model deep learning:

    - **Kucing vs Anjing**: Klasifikasi gambar binatang menggunakan CNN
    - **Food-101**: Klasifikasi 101 jenis makanan dunia
    - **Analisis Emosi**: Klasifikasi emosi teks Bahasa Indonesia

    Gunakan menu di sebelah kiri untuk mencoba setiap model.
    """)

    st.image(
        "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
        caption="Demo Multi-Model Deep Learning",
        use_column_width=True,
    )

# -----------------------------
# Halaman: Kucing vs Anjing
# -----------------------------
elif page == "Kucing vs Anjing":
    st.title("Klasifikasi Gambar: Kucing vs Anjing")
    model = load_catdog_model()
    if model is None:
        st.stop()

    st.write("Input shape model:", model.input_shape)

    uploaded = st.file_uploader(
        "Upload gambar kucing atau anjing", type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        target_size = tuple(model.input_shape[1:3])
        input_tensor = preprocess_image(image, target_size)

        with st.spinner("Menganalisis..."):
            pred = model.predict(input_tensor, verbose=0)[0][0]
        label = "Kucing" if pred > 0.5 else "Anjing"
        confidence = float(pred if pred > 0.5 else 1 - pred)

        st.subheader("Hasil Prediksi")
        st.write(f"**Kelas:** {label}")
        st.metric("Keyakinan", f"{confidence:.4f}")

        st.subheader("Visualisasi Feature Map")
        conv_layers = [layer.name for layer in model.layers if "conv" in layer.name][:4]
        if conv_layers:
            activations = get_feature_maps(model, input_tensor, conv_layers)
            for layer_name, act in zip(conv_layers, activations):
                st.markdown(f"**Layer:** {layer_name}")
                fig, axes = plt.subplots(1, min(4, act.shape[-1]), figsize=(10, 3))
                for j in range(min(4, act.shape[-1])):
                    axes[j].imshow(act[0, :, :, j], cmap="viridis")
                    axes[j].axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# -----------------------------
# Halaman: Food-101
# -----------------------------
elif page == "Food-101":
    st.title("Klasifikasi Makanan: Food-101")
    model = load_food_model()
    if model is None:
        st.stop()

    st.write("Input shape model:", model.input_shape)

    uploaded = st.file_uploader("Upload gambar makanan", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Makanan yang diunggah", use_column_width=True)

        target_size = tuple(model.input_shape[1:3])
        input_tensor = preprocess_image(image, target_size)

        with st.spinner("Menganalisis makanan..."):
            pred = model.predict(input_tensor, verbose=0)[0]
            top_indices = np.argsort(pred)[::-1][:3]

        class_names = [
            "apple pie",
            "baby back ribs",
            "baklava",
            "beef carpaccio",
            "beef tartare",
            "beet salad",
            "beignets",
            "bibimbap",
            "bread pudding",
            "breakfast burrito",
            "bruschetta",
            "caesar salad",
            "cannoli",
            "caprese salad",
            "carrot cake",
            "ceviche",
            "cheesecake",
            "cheese plate",
            "chicken curry",
            "chicken quesadilla",
            "chicken wings",
            "chocolate cake",
            "chocolate mousse",
            "churros",
            "clam chowder",
            "club sandwich",
            "crab cakes",
            "creme brulee",
            "croque madame",
            "cup cakes",
            "deviled eggs",
            "donuts",
            "dumplings",
            "edamame",
            "eggs benedict",
            "escargots",
            "falafel",
            "filet mignon",
            "fish and chips",
            "foie gras",
            "french fries",
            "french onion soup",
            "french toast",
            "fried calamari",
            "fried rice",
            "frozen yogurt",
            "garlic bread",
            "gnocchi",
            "greek salad",
            "grilled cheese sandwich",
            "grilled salmon",
            "guacamole",
            "gyoza",
            "hamburger",
            "hot and sour soup",
            "hot dog",
            "huevos rancheros",
            "hummus",
            "ice cream",
            "lasagna",
            "lobster bisque",
            "lobster roll sandwich",
            "macaroni and cheese",
            "macarons",
            "miso soup",
            "mussels",
            "nachos",
            "omelette",
            "onion rings",
            "oysters",
            "pad thai",
            "paella",
            "pancakes",
            "panna cotta",
            "peking duck",
            "pepper pizza",
            "pho",
            "pizza",
            "pork chop",
            "poutine",
            "prime rib",
            "pulled pork sandwich",
            "ramen",
            "ravioli",
            "red velvet cake",
            "risotto",
            "samosa",
            "sashimi",
            "scallops",
            "seaweed salad",
            "shrimp and grits",
            "spaghetti bolognese",
            "spaghetti carbonara",
            "spring rolls",
            "steak",
            "strawberry shortcake",
            "sushi",
            "tacos",
            "octopus",
            "omelette",
            "onion rings",
        ]

        st.subheader("Top 3 Prediksi")
        for i, idx in enumerate(top_indices):
            label = class_names[idx].title()
            confidence = pred[idx]
            st.write(f"{i + 1}. **{label}** — {confidence:.4f}")

        st.subheader("Visualisasi Feature Map")
        conv_layers = [layer.name for layer in model.layers if "conv" in layer.name][:3]
        if conv_layers:
            activations = get_feature_maps(model, input_tensor, conv_layers)
            for layer_name, act in zip(conv_layers, activations):
                st.markdown(f"**Layer:** {layer_name}")
                fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                for j in range(3):
                    if j < act.shape[-1]:
                        axes[j].imshow(act[0, :, :, j], cmap="plasma")
                        axes[j].axis("off")
                    else:
                        axes[j].axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# -----------------------------
# Halaman: Analisis Emosi
# -----------------------------
elif page == "Analisis Emosi":
    st.title("Analisis Emosi - Bahasa Indonesia")
    st.markdown("""
    Model: [StevenLimcorn/indonesian-roberta-base-emotion-classifier](https://huggingface.co/StevenLimcorn/indonesian-roberta-base-emotion-classifier)  
    Dataset: IndoNLU EmoT | F1-score: 72.05% | Akurasi: 71.81%
    """)

    model, tokenizer = load_nlp_model()
    if model is None:
        st.stop()

    user_input = st.text_area(
        "Masukkan teks dalam Bahasa Indonesia:",
        height=150,
        placeholder="Contoh: 'Aku merasa sedih hari ini...' atau 'Saya bangga denganmu!'",
    )

    if st.button("Analisis Emosi"):
        if not user_input.strip():
            st.warning("Mohon masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis emosi..."):
                try:
                    inputs = tokenizer(
                        user_input,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512,
                    )
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    predicted_class_id = logits.argmax(-1).item()

                    label_mapping = {
                        0: "SADNESS",
                        1: "ANGER",
                        2: "SUPPORT",
                        3: "HOPE",
                        4: "DISAPPOINTMENT",
                    }

                    description_mapping = {
                        0: "Komentar bernuansa kesedihan atau duka.",
                        1: "Komentar bernuansa kemarahan atau kekesalan.",
                        2: "Komentar yang menunjukkan dukungan atau pembelaan.",
                        3: "Komentar yang mengekspresikan harapan positif untuk masa depan.",
                        4: "Komentar bernuansa kekecewaan terhadap situasi atau pihak tertentu.",
                    }

                    predicted_label = label_mapping.get(predicted_class_id, "Unknown")
                    predicted_desc = description_mapping.get(
                        predicted_class_id, "Tidak dikenali"
                    )

                    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

                    st.subheader("Hasil Prediksi")
                    st.markdown(f"**Label Numerik:** `{predicted_class_id}`")
                    st.markdown(f"**Label Teks:** `{predicted_label}`")
                    st.markdown(f"**Keterangan:** {predicted_desc}")

                    st.subheader("Distribusi Probabilitas")
                    fig, ax = plt.subplots(figsize=(7, 6))
                    labels = [label_mapping[i] for i in range(5)]
                    colors = ["#ff9999", "#ffcc99", "#99cc99", "#66b3ff", "#ffcc66"]
                    ax.pie(
                        probs,
                        labels=labels,
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=colors,
                        textprops={"fontsize": 10},
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                    with st.expander("Detail Probabilitas"):
                        for i in range(5):
                            st.write(f"{label_mapping[i]}: {probs[i]:.4f}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
