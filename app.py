import streamlit as st
import streamlit.components.v1 as stc
import pickle

with open('Logistic_Regression_Model.pkl', 'rb') as file:
    Logistic_Regression_Model = pickle.load(file)

st.set_page_config(
    page_title="Medical Cost Prediction", page_icon="💊", layout="centered"
)

st.title("💊 Medical Cost Predictor")

st.markdown(
    "Masukkan informasi pasien untuk memprediksi **biaya medis tahunan** menggunakan model regresi yang telah dilatih sebelumnya (Medical Cost Personal Dataset)."
)

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Medical Cost Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">Copyright 2025. Delta Seekers Team</h4> 
                """

desc_temp = """ ### Loan Prediction App 
                This app is used by Credit team for deciding Loan Application
                
                #### Data Source
                Kaggle: Link <Masukkan Link>
                """

def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

def run_ml_app():
    design = """<div style="padding:15px;">
                    <h3 style="color:#fff">Medical Cost Prediction</h3>
                </div
             """
    st.markdown(design, unsafe_allow_html=True)
    
    #Membuat Struktur Form
    left, right = st.columns((2,2))
    usia = st.slider("Usia", 18, 64, 30)
    sex = left.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
    smoker = right.selectbox('Apakah Merokok', ('Ya', 'Tidak'))
    height = left.number_input('Tinggi Badan')
    weight = right.number_input('Berat Badan')
    children = left.selectbox("Jumlah Anak (dependents)", list(range(0, 6)), index=0)
    region = right.selectbox('Lokasi Tinggal', ("southeast", "southwest", "northeast", "northwest"))
    button = st.button("Predict")

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
if predict_btn:
    input_df = preprocess_input(age, sex, height, weight, children, smoker, region)

    with st.spinner("Menghitung prediksi ..."):
        prediction = model.predict(input_df)[0]

    st.subheader("💵 Estimasi Biaya Medis Tahunan")
    st.metric(label="Charges (USD)", value=f"${prediction:,.2f}")

    # Detail input
    with st.expander("Lihat detail input"):
        st.dataframe(input_df, use_container_width=True)

if __name__ == "__main__":
    main()
