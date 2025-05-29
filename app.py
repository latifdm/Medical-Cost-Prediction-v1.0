import streamlit as st
import streamlit.components.v1 as stc
import pickle

with open('Logistic_Regression_Model.pkl', 'rb') as file:
    Logistic_Regression_Model = pickle.load(file)

st.set_page_config(
    page_title="Medical Cost Prediction", page_icon="💊", layout="centered"
)

with st.sidebar:
    st.markdown("### Menu")
    page = st.selectbox("", ["Home", "Machine Learning App"], index=0)

# -----------------------------------------------------------------------------
# 🏠  PAGE — Home
# -----------------------------------------------------------------------------
if page == "Home":
    st.title("💊 Medical Cost Predictor App")
    st.markdown(
        "Masukkan informasi pasien untuk memprediksi **biaya medis tahunan** menggunakan model regresi yang telah dilatih sebelumnya (Medical Cost Personal Dataset)."
    )

    # ---------- Team section ----------
    st.subheader("Delta Seekers Team")
    {
            "name": "Member 1",
            "photo": "assets/member1.jpg",  # ganti path sesuai aset
            "ig": "https://instagram.com/member1",
            "li": "https://linkedin.com/in/member1",
        },
        {
            "name": "Member 2",
            "photo": "assets/member2.jpg",
            "ig": "https://instagram.com/member2",
            "li": "https://linkedin.com/in/member2",
        },
        {
            "name": "Member 3",
            "photo": "assets/member3.jpg",
            "ig": "https://instagram.com/member3",
            "li": "https://linkedin.com/in/member3",
        },
        {
            "name": "Member 4",
            "photo": "assets/member4.jpg",
            "ig": "https://instagram.com/member4",
            "li": "https://linkedin.com/in/member4",
        },
        {
            "name": "Member 5",
            "photo": "assets/member5.jpg",
            "ig": "https://instagram.com/member5",
            "li": "https://linkedin.com/in/member5",
        },
    ]

    for m in members:
        col_photo, col_info = st.columns([1, 3])
        with col_photo:
            st.image(m["photo"], width=90, caption=" ")
        with col_info:
            st.markdown(
                f"**{m['name']}**  \n[Instagram]({m['ig']}) | [LinkedIn]({m['li']})"
            )
        st.divider()

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
def load_model(path: str = "Logistic_Regression_Model.pkl"):
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

