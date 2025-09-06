import streamlit as st
import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(
    page_title="Multi-Model AI Demo", layout="wide", initial_sidebar_state="expanded"
)

# -----------------------------
# CSS Kustom
# -----------------------------
st.markdown(
    """
<style>
    .main { background-color: #f9f9f9; }
    .section-title {
        font-weight: 600;
        color: #1a1a1a;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #f1f3f5;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .feedback-box {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.95em;
    }
    .correct { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .wrong { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9em;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar Navigasi
# -----------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Model", ["Beranda", "Kucing vs Anjing", "Food-101", "Analisis Emosi"]
)

# -----------------------------
# Inisialisasi Session State
# -----------------------------
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()


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
# Load Model
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
        st.error(f"Gagal muat model Kucing vs Anjing: {e}")
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
        st.error(f"Gagal muat model Food-101: {e}")
        return None


@st.cache_resource
def load_nlp_model():
    try:
        model_name = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Gagal muat model NLP: {e}")
        return None, None


# -----------------------------
# URL Meme
# -----------------------------
MEME_NLP = {
    "SADNESS": "https://media.tenor.com/f1fVZ5yuLaIAAAAM/sad.gif",
    "ANGER": "https://media1.tenor.com/m/KuuC_7Oy1FEAAAAd/prabowo-marah-angry.gif",
    "SUPPORT": "https://media1.tenor.com/m/ba-O8SSOaswAAAAC/cute-bear-silvia-emoji.gif",
    "HOPE": "https://media1.tenor.com/m/quKCeO2K3fIAAAAC/homer-simpson-prier.gif",
    "DISAPPOINTMENT": "https://media1.tenor.com/m/ciNDyf6AgH0AAAAd/disappointed-disappointed-fan.gif",
}

MEME_CNN = {
    "correct": "https://media1.tenor.com/m/7Ypq9_9najcAAAAd/thumbs-up-double-thumbs-up.gif",  # thumbs up
    "wrong": "https://media1.tenor.com/m/y--slZ8dhqYAAAAC/donald-trump-wrong.gif",  # shrug
}

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
        use_container_width=True,
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
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        target_size = tuple(model.input_shape[1:3])
        input_tensor = preprocess_image(image, target_size)

        with st.spinner("Menganalisis..."):
            pred = model.predict(input_tensor, verbose=0)[0][0]

        predicted_class = "Kucing" if pred > 0.5 else "Anjing"
        confidence = float(pred if pred > 0.5 else 1 - pred)

        st.markdown(
            '<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        col1.write(f"**Kelas:** {predicted_class}")
        col2.metric("Keyakinan", f"{confidence:.4f}")

        # Feedback user
        st.markdown(
            '<div class="section-title">Apakah prediksi ini benar?</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        feedback = None
        with cols[0]:
            if st.button("✅ Benar", key="dogcat_correct"):
                feedback = "correct"
                st.success("Terima kasih!")
                st.balloons()
        with cols[1]:
            if st.button("❌ Salah", key="dogcat_wrong"):
                feedback = "wrong"
                st.warning("Terima kasih atas masukan!")

        # Tampilkan meme jika ada feedback
        if feedback:
            st.image(MEME_CNN[feedback], width=200, caption="Reaksi Anda")

        # Simpan feedback
        if feedback:
            st.session_state.feedback_log.append(
                {
                    "task": "catdog",
                    "predicted": predicted_class,
                    "feedback": feedback,
                    "timestamp": str(datetime.now()),
                }
            )

        # Feature Map
        st.markdown(
            '<div class="section-title">Visualisasi Feature Map</div>',
            unsafe_allow_html=True,
        )
        conv_layers = [layer.name for layer in model.layers if "conv" in layer.name][:4]
        if conv_layers:
            activations = get_feature_maps(model, input_tensor, conv_layers)
            for layer_name, act in zip(conv_layers, activations):
                st.markdown(f"**{layer_name}**")
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
        st.image(image, caption="Makanan yang diunggah", use_container_width=True)

        target_size = tuple(model.input_shape[1:3])
        input_tensor = preprocess_image(image, target_size)

        with st.spinner("Menganalisis makanan..."):
            pred = model.predict(input_tensor, verbose=0)[0]
            top_idx = np.argsort(pred)[::-1][0]

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
        ]

        predicted_food = class_names[top_idx].title()
        confidence = float(pred[top_idx])

        st.markdown(
            '<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True
        )
        st.write(f"**Makanan:** {predicted_food}")
        st.metric("Keyakinan", f"{confidence:.4f}")

        # Feedback
        st.markdown(
            '<div class="section-title">Apakah prediksi ini benar?</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        feedback = None
        with cols[0]:
            if st.button("✅ Benar", key="food_correct"):
                feedback = "correct"
                st.success("Terima kasih!")
                st.balloons()
        with cols[1]:
            if st.button("❌ Salah", key="food_wrong"):
                feedback = "wrong"
                st.warning("Terima kasih atas masukan!")

        if feedback:
            st.image(MEME_CNN[feedback], width=200, caption="Reaksi Anda")

        if feedback:
            st.session_state.feedback_log.append(
                {
                    "task": "food",
                    "predicted": predicted_food,
                    "feedback": feedback,
                    "timestamp": str(datetime.now()),
                }
            )

# -----------------------------
# Halaman: Analisis Emosi
# -----------------------------
elif page == "Analisis Emosi":
    st.title("Analisis Emosi - Bahasa Indonesia")

    st.markdown(
        """
    <div style="background-color: #e7f3ff; padding: 14px; border-radius: 8px; border-left: 5px solid #007BFF;">
    <b>Indo RoBERTa Emotion Classifier</b><br>
    Dataset: IndoNLU EmoT • F1: 72.05% • Akurasi: 71.81%<br>
    <a href="https://huggingface.co/StevenLimcorn/indonesian-roberta-base-emotion-classifier" target="_blank">Lihat di Hugging Face</a>
    </div>
    """,
        unsafe_allow_html=True,
    )

    model, tokenizer = load_nlp_model()
    if model is None:
        st.stop()

    # Label mapping (di luar try)
    label_mapping = {
        0: "SADNESS",
        1: "ANGER",
        2: "SUPPORT",
        3: "HOPE",
        4: "DISAPPOINTMENT",
    }
    desc_mapping = {
        0: "Komentar bernuansa kesedihan atau duka.",
        1: "Komentar bernuansa kemarahan atau kekesalan.",
        2: "Komentar yang menunjukkan dukungan atau pembelaan.",
        3: "Komentar yang mengekspresikan harapan positif untuk masa depan.",
        4: "Komentar bernuansa kekecewaan terhadap situasi atau pihak tertentu.",
    }

    # Contoh teks
    st.markdown("##### Contoh Teks:")
    examples = [
        "Aku merasa sangat sedih hari ini.",
        "Kamu hebat! Terus semangat!",
        "Aku marah sekali dengan keputusan itu.",
        "Semoga besok lebih baik.",
        "Aku kecewa dengan janji yang tidak ditepati.",
    ]
    for ex in examples:
        if st.button(f"💬 {ex}", key=f"ex_{ex}"):
            st.session_state.user_input = ex

    user_input = st.text_area(
        "Masukkan teks dalam Bahasa Indonesia:",
        value=st.session_state.get("user_input", ""),
        height=150,
        key="input_area",
    )

    if st.button("🔍 Analisis Emosi"):
        if not user_input.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis..."):
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
                    pred_id = logits.argmax(-1).item()
                    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

                    label = label_mapping.get(pred_id, "Unknown")
                    desc = desc_mapping.get(pred_id, "Tidak dikenali")

                    st.session_state.last_result = {
                        "text": user_input,
                        "label": label,
                        "desc": desc,
                        "probs": probs,
                    }
                except Exception as e:
                    st.error(f"Error: {e}")

    if "last_result" in st.session_state:
        res = st.session_state.last_result

        st.markdown(
            '<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True
        )
        st.markdown(f"**Label Teks:** `{res['label']}`")
        st.markdown(f"**Keterangan:** {res['desc']}")

        # Meme sesuai label emosi
        if res["label"] in MEME_NLP:
            st.image(
                MEME_NLP[res["label"]], width=250, caption=f"Reaksi: {res['label']}"
            )

        # Pie Chart
        st.markdown(
            '<div class="section-title">Distribusi Probabilitas</div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(7, 7))
        labels = [label_mapping[i] for i in range(5)]
        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
        ax.pie(
            res["probs"], labels=labels, autopct="%1.1f%%", startangle=90, colors=colors
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close()

# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
if st.session_state.feedback_log:
    st.sidebar.markdown("### 📊 Statistik Feedback")
    df = pd.DataFrame(st.session_state.feedback_log)
    correct = len(df[df.feedback == "correct"])
    total = len(df)
    st.sidebar.metric("Benar", f"{correct}/{total}")
st.sidebar.markdown(
    """
<div class='footer'>
Dibangun dengan ❤️ menggunakan Streamlit
</div>
""",
    unsafe_allow_html=True,
)

