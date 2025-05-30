import streamlit as st
import streamlit.components.v1 as stc # Not used in the provided snippet, but keeping it
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model (cached if possible by Streamlit)
@st.cache_resource # Use st.cache_resource for models/large objects
def load_model():
    with open('gradient_boosting_regressor_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

Gradient_Boosting_Regressor_Model = load_model()

st.set_page_config(
    page_title="Medical Cost Prediction", page_icon="💊", layout="centered"
)

with st.sidebar:
    st.markdown("### Menu")
    # This selectbox label is explicitly set, so it's unlikely to be the cause of the 'empty label' warning.
    # The warning might come from some other internal Streamlit component if not this one.
    page = st.selectbox(
        label="Navigasi",
        options=["Home", "Machine Learning App", "Dashboard"],
        index=0,
        label_visibility="collapsed"
    )

# --- Utility Functions (Global Scope) ---
def calculate_bmi(height_cm, weight_kg):
    """
    Calculates BMI given height in centimeters and weight in kilograms.
    BMI = weight (kg) / (height (m))^2
    """
    if height_cm <= 0 or weight_kg <= 0:
        return 0 # Handle invalid input gracefully
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def preprocess_input(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    """Konversi input user ➜ DataFrame yang kompatibel dengan model."""
    cols = [
        "age",
        "bmi",
        "children",
        "sex_female",
        "sex_male",
        "smoker_no",
        "smoker_yes",
        "region_northeast",
        "region_northwest",
        "region_southeast",
        "region_southwest",
    ]

    data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        # one-hot encoding — sex (matching 'Pria'/'Wanita' from selectbox)
        "sex_female": 1 if sex == "Wanita" else 0,
        "sex_male": 1 if sex == "Pria" else 0,
        # one-hot encoding — smoker (matching 'Ya'/'Tidak' from selectbox)
        "smoker_no": 1 if smoker == "Tidak" else 0,
        "smoker_yes": 1 if smoker == "Ya" else 0,
        # one-hot encoding — region
        "region_northeast": 1 if region == "northeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    return pd.DataFrame([data])[cols]

# -----------------------------------------------------------------------------
# 🏠 PAGE — Home
# -----------------------------------------------------------------------------
if page == "Home":
    st.title("💊 Medical Cost Predictor App")
    st.markdown(
        "Aplikasi Machine Learning ini di buat untuk memprediksi biaya medis tahunan pasien berdasarkan model Regresi yang telah dilatih sebelumnya dengan sumber dataset Medical Cost Personal Datasets Kaggle."
    )
    st.markdown(
        "Data Source : Link https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download"
    )

    # ---------- Team section ----------
    st.subheader("👨‍⚕️ Delta Seekers Team")
    members = [
        {
            "name": "Ahmad Azhar Naufal Farizky",
            "photo": "profile.svg", # Ensure profile.svg exists or provide a placeholder
            "li": "https://linkedin.com/in/ahmad-azhar-naufal-farizky-3b3b3b2b", # Placeholder LinkedIn
        },
        {
            "name": "Kristina Sarah Yuliana",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/kristina-sarah-yuliana-3b3b3b2c", # Placeholder LinkedIn
        },
        {
            "name": "Latif Dwi Mardani",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/latif-dwi-mardani-3b3b3b2d", # Placeholder LinkedIn
        },
        {
            "name": "Jalu Prayoga",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/jalu-prayoga-3b3b3b2e", # Placeholder LinkedIn
        },
        {
            "name": "Ayasha Naila Ismunandar",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/ayasha-naila-ismunandar-3b3b3b2f", # Placeholder LinkedIn
        },
    ]

    # Tampilkan 5 anggota dalam 1 baris horizontal
    cols = st.columns(len(members))
    for col, member in zip(cols, members):
        with col:
            st.image(member["photo"], width=100)
            st.markdown(
                f"**{member['name']}** \n"
                f"[LinkedIn]({member['li']})"
            )

# -----------------------------------------------------------------------------
# 🤖 PAGE — Machine Learning App
# -----------------------------------------------------------------------------
elif page == "Machine Learning App":
    st.title("💊 Medical Cost Predictor App")
    st.markdown(
        "Masukkan informasi pasien untuk memprediksi **biaya medis tahunan** menggunakan model regresi yang telah dilatih sebelumnya (Medical Cost Personal Dataset)."
    )

    # Membuat Struktur Form
    left, right = st.columns((2, 2))
    age = st.slider("Usia", 18, 100, 30)
    sex = left.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
    smoker = right.selectbox('Apakah Merokok', ('Ya', 'Tidak'))
    height = left.number_input('Tinggi Badan (cm)', min_value=100.0, max_value=250.0, value=170.0)
    weight = right.number_input('Berat Badan (kg)', min_value=30.0, max_value=200.0, value=70.0)
    children = left.selectbox("Jumlah Anak", list(range(0, 6)), index=0)
    region = right.selectbox('Lokasi Tinggal', ("southeast", "southwest", "northeast", "northwest"))

    # Calculate BMI here, before the predict button logic
    bmi = calculate_bmi(height, weight)

    # Single "Predict Medical Cost" button
    if st.button("Predict Medical Cost"):
        try:
            # model is already loaded and assigned to Gradient_Boosting_Regressor_Model
            model = Gradient_Boosting_Regressor_Model
        except Exception as e: # Catch a broader exception for initial debugging if model loading fails
            st.error(f"⚠️ **Error loading model**: {e}. Pastikan file model Anda benar.")
            st.stop()

        # Preprocess input with the correct variable names
        input_df = preprocess_input(age, sex, bmi, children, smoker, region)

        with st.spinner("Menghitung prediksi ..."):
            prediction = model.predict(input_df)[0]

        st.subheader("💵 Estimasi Biaya Medis Tahunan")
        st.metric("Charges (USD)", f"${prediction:,.2f}")

        with st.expander("Detail input"):
            st.dataframe(input_df, use_container_width=True)

# -----------------------------------------------------------------------------
# 📊 PAGE — Dashboard
# -----------------------------------------------------------------------------
elif page == "Dashboard":
    st.title("📊 Medical Cost Dashboard")
    st.markdown("Analisis data dan visualisasi statistik pasien.")

    try:
        df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        st.error("⚠️ **insurance.csv** tidak ditemukan. Letakkan file dataset di folder yang sama dengan *app.py*.")
        st.stop()

    st.subheader("Ringkasan Statistik")
    st.dataframe(df.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi BMI")
        # Ensure 'bmi' column exists and is numeric if using df['bmi'] directly
        if 'bmi' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df["bmi"], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Kolom 'bmi' tidak ditemukan dalam dataset.")

    with col2:
        st.subheader("Perbandingan Jumlah Perokok")
        if 'smoker' in df.columns:
            smoker_counts = df["smoker"].value_counts()
            fig, ax = plt.subplots()
            smoker_counts.plot(kind='bar', ax=ax)
            ax.set_ylabel('Jumlah')
            ax.set_title('Jumlah Perokok vs Non-Perokok')
            st.pyplot(fig)
        else:
            st.warning("Kolom 'smoker' tidak ditemukan dalam dataset.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Biaya Medis berdasarkan Usia")
        if 'age' in df.columns and 'charges' in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(x="age", y="charges", data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Kolom 'age' atau 'charges' tidak ditemukan dalam dataset.")

    with col4:
        st.subheader("Jumlah Anak per Region")
        if 'region' in df.columns and 'children' in df.columns:
            children_region = df.groupby("region")["children"].sum()
            fig, ax = plt.subplots()
            children_region.plot(kind='bar', ax=ax)
            ax.set_ylabel('Jumlah Anak')
            ax.set_title('Total Anak per Region')
            st.pyplot(fig)
        else:
            st.warning("Kolom 'region' atau 'children' tidak ditemukan dalam dataset.")

    st.subheader("📊 Korelasi antar Fitur Numerik")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("💰 Sebaran Biaya Medis per Region")
    fig, ax = plt.subplots()
    sns.boxplot(x="region", y="charges", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Distribusi Usia Pasien")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=10, kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

    st.subheader("🗺️ Proporsi Pasien per Region")
    region_counts = df["region"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(region_counts, labels=region_counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
