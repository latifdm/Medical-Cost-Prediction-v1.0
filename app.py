import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Medical Cost Prediction", page_icon="💊", layout="centered"
)

with st.sidebar:
    st.markdown("### Menu")
    page = st.selectbox("", ["Home", "Machine Learning App", "Dashboard"], index=0)


# -----------------------------------------------------------------------------
# 🏠  PAGE — Home
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
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/member1",
        },
        {
            "name": "Kristina Sarah Yuliana",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/member2",
        },
        {
            "name": "Latif Dwi Mardani",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/member3",
        },
        {
            "name": "Jalu Prayoga",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/member4",
        },
        {
            "name": "Ayasha Naila Ismunandar",
            "photo": "profile.svg",
            "li": "https://linkedin.com/in/member5",
        },
    ]

    # Tampilkan 5 anggota dalam 1 baris horizontal
    cols = st.columns(len(members))
    for col, member in zip(cols, members):
        with col:
            st.image(member["photo"], width=100)
            st.markdown(
                f"**{member['name']}**  \n"
                f"[LinkedIn]({member['li']})"
            )

# -----------------------------------------------------------------------------
# 🤖  PAGE — Machine Learning App
# -----------------------------------------------------------------------------
elif page == "Machine Learning App":
    st.title("💊 Medical Cost Predictor App")
    st.markdown(
        "Masukkan informasi pasien untuk memprediksi **biaya medis tahunan** menggunakan model regresi yang telah dilatih sebelumnya (Medical Cost Personal Dataset)."
    )
    
    #Membuat Struktur Form
    left, right = st.columns((2,2))
    age = st.slider("Usia", 18, 100, 30)
    sex = left.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
    smoker = right.selectbox('Apakah Merokok', ('Ya', 'Tidak'))
    height = left.number_input('Tinggi Badan')
    weight = right.number_input('Berat Badan')
    children = left.selectbox("Jumlah Anak", list(range(0, 6)), index=0)
    region = right.selectbox('Lokasi Tinggal', ("southeast", "southwest", "northeast", "northwest"))
    predict_btn = st.button("Predict Medical Cost", type="primary")

    # ---------------------------------------------------------------------------
    # 🧮 Fungsi Konversi BMI
    # ---------------------------------------------------------------------------
    def calculate_bmi(height, weight):
        weight = height / 100
        return weight / (height ** 2)

# -----------------------------------------------------------------------------
# 📊  PAGE — Dashboard
# -----------------------------------------------------------------------------
elif page == "Dashboard":
    st.title("📊 Medical Cost Dashboard")
    st.markdown("Analisis data dan visualisasi statistik pasien.")

    df = pd.read_csv("insurance.csv")

    st.subheader("Ringkasan Statistik")
    st.dataframe(df.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi BMI")
        st.bar_chart(df["bmi"])

    with col2:
        st.subheader("Perbandingan Jumlah Perokok")
        st.bar_chart(df["smoker"].value_counts())

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Biaya Medis berdasarkan Usia")
        st.scatter_chart(df[["age", "charges"]])

    with col4:
        st.subheader("Jumlah Anak per Region")
        children_region = df.groupby("region")["children"].sum()
        st.bar_chart(children_region)
    
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

# ---------------------------------------------------------------------------
# 🔄 Utilitas Pre-processing Input
# ---------------------------------------------------------------------------
def preprocess_input(age, sex, height, weight, children, smoker, region):
    """Konversi input user ➜ DataFrame yang kompatibel dengan model."""

    bmi = calculate_bmi(height, weight)

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
        # one-hot encoding — sex
        "sex_female": 1 if sex == "female" else 0,
        "sex_male": 1 if sex == "male" else 0,
        # one-hot encoding — smoker
        "smoker_no": 1 if smoker == "no" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        # one-hot encoding — region
        "region_northeast": 1 if region == "northeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    return pd.DataFrame([data])[cols]


# ---------------------------------------------------------------------------
# 📦 Load Model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str = "gradient_boosting_regressor_model.pkl"):
    """Muat model regressi tersimpan dalam file pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "⚠️ **model.pkl** tidak ditemukan. Pastikan file model sudah berada di folder yang sama dengan *app.py*."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 🧮 Prediksi biaya
# ---------------------------------------------------------------------------
if st.button("Predict Medical Cost", type="primary"):
        try:
            model = load_model()
        except FileNotFoundError:
            st.error("⚠️  **model.pkl** tidak ditemukan. Letakkan file model di folder yang sama dengan *app.py*.")
            st.stop()

        input_df = preprocess_input(age, sex, bmi, children, smoker, region)

        with st.spinner("Menghitung prediksi ..."):
            prediction = model.predict(input_df)[0]

        st.subheader("💵 Estimasi Biaya Medis Tahunan")
        st.metric("Charges (USD)", f"${prediction:,.2f}")

        with st.expander("Detail input"):
            st.dataframe(input_df, use_container_width=True)
