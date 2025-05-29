import streamlit as st
import streamlit.components.v1 as stc
import pickle

with open('Logistic_Regression_Model.pkl', 'rb') as file:
    Logistic_Regression_Model = pickle.load(file)

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
    tinggi = left.number_input('Tinggi Badan')
    berat = right.number_input('Berat Badan')
    children = left.selectbox("Jumlah Anak (dependents)", list(range(0, 6)), index=0)
    region = right.selectbox('Lokasi Tinggal', ("Southeast", "Southwest", "Northeast", "Northwest"))
    button = st.button("Predict")

    #If button is clilcked
    if button: 
        result = predict(usia, sex, tinggi, berat, smoker, children, region)
        
        if result == 'Eligible':
            st.success(f'You Are {result} for the loan')
        else:
            st.error(f'You are {result} for the loan')

def predict(gender, married, dependent, education, self_employed, applicant_income, coApplicant_income
                         ,loan_amount, loan_amount_term, credit_history, property_area):
    #Preprocessing User input
    gen = 0 if gender == 'Male' else 1
    mar = 0 if married == 'Yes' else 1
    edu = 0 if education == 'Graduate' else 1
    sem = 0 if self_employed == 'Yes' else 1
    pro = 0 if property_area == 'Semiurban' else 1 if property_area == 'Urban' else 2
    

    #Making prediction
    prediction = Logistic_Regression_Model.predict(
        [[gen, mar, dependent, edu, sem, applicant_income, coApplicant_income,
        loan_amount, loan_amount_term, credit_history, pro]])
    
    result = 'Not Eligible' if prediction == 0 else 'Eligible'
    return result

if __name__ == "__main__":
    main()
