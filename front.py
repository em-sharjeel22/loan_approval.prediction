# import numpy as np
# import pickle as pk 
# import streamlit as st
# import pandas as pd
# import os
# import plotly.express as px

# model_path = "final_loan_model.sav"
# if not os.path.exists(model_path):
#     print("Model file not found!")
# else:
#     loaded_model = pk.load(open(model_path, 'rb'))
# # Helper function to prepare input data for prediction
# def prepare_input_data(input_data):
#     df = pd.DataFrame([{
#         "Income":     input_data[0],
#         "Ownership":  input_data[1],
#         "Loan Amount": input_data[2],
#         "Loan Intent ":  input_data[3],
#         "Interest Rate": input_data[4],
#         "Credit.Hist":   input_data[5],
#         "Cr.Score": input_data[6],
#         "Previous Loans":input_data[7]
#     }])
#     return df
# def loan_approval_app():
#     st.title("Loan Approval Prediction")
#     st.write("Enter the details of the loan applicant:")

#     income = st.number_input("Income", min_value=0)
#     ownership = st.selectbox("Ownership", ["RENT", "OWN", "MORTGAGE"])
#     loan_amount = st.number_input("Loan Amount", min_value=0)
#     loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME", "DEBTCONSOLIDATION"])
#     interest_rate = st.number_input("Interest Rate", min_value=0.0, step=0.01)
#     credit_hist = st.number_input("Credit History Length (years)", min_value=0)
#     credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
#     previous_loans = st.selectbox("Previous Loans", ["Yes", "No"])

#     if st.button("Predict"):
#         input_data = [income, ownership, loan_amount, loan_intent, interest_rate, credit_hist, credit_score, previous_loans]
#         new_person = prepare_input_data(input_data)
#         prediction = loaded_model.predict(new_person)
#         if prediction[0] == 1:
#             st.success("Loan APPROVED ✓")
#         else:
#             st.error("Loan REJECTED ✗")

#     st.success(f"Loan Approval Prediction App is ready! Please enter the details and click Predict.", icon="✅")

# if __name__ == "__main__":  
#     loan_approval_app()            


import numpy as np
import pickle as pk
import streamlit as st
import pandas as pd
import os
import plotly.express as px

# model_path = "final_loan_model.sav"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "final_loan_model.sav")
# FIX: Use global so the variable is accessible inside the function
loaded_model = ""

if not os.path.exists(model_path):
    st.error("Model file not found! Please make sure 'final_loan_model.sav' is in the same directory.")
else:
    loaded_model = pk.load(open(model_path, 'rb'))


def prepare_input_data(input_data):
    """Helper function to prepare input data for prediction."""
    df = pd.DataFrame([{
        "Income":         input_data[0],
        "Ownership":      input_data[1],
        "Loan Amount":    input_data[2],
        "Loan Intent":    input_data[3],   # FIX: removed trailing space in key
        "Interest Rate":  input_data[4],
        "Credit.Hist":    input_data[5],
        "Cr.Score":       input_data[6],
        "Previous Loans": input_data[7]
    }])
    return df


def loan_approval_app():
    st.title("Loan Approval Prediction")
    st.write("Enter the details of the loan applicant:")

    # FIX: Use global loaded_model inside the function
    global loaded_model

    income        = st.number_input("Income", min_value=0)
    ownership     = st.selectbox("Ownership", ["RENT", "OWN", "MORTGAGE"])
    loan_amount   = st.number_input("Loan Amount", min_value=0)
    loan_intent   = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME", "DEBTCONSOLIDATION"])
    interest_rate = st.number_input("Interest Rate", min_value=0.0, step=0.01)
    credit_hist   = st.number_input("Credit History Length (years)", min_value=0)
    credit_score  = st.number_input("Credit Score", min_value=300, max_value=850)
    previous_loans = st.selectbox("Previous Loans", ["Yes", "No"])

    if st.button("Predict"):
        if loaded_model is None:
            st.error("Model is not loaded. Please check that the model file exists.")
        else:
            input_data = [income, ownership, loan_amount, loan_intent,
                          interest_rate, credit_hist, credit_score, previous_loans]
            new_person = prepare_input_data(input_data)
            prediction = loaded_model.predict(new_person)

            if prediction[0] == 1:
                st.success("Loan APPROVED ✓")
            else:
                st.error("Loan REJECTED ✗")

    # FIX: Changed st.success → st.info for the welcome banner (success green is misleading here)
    st.info("Loan Approval Prediction App is ready! Please enter the details and click Predict.")


if __name__ == "__main__":
    loan_approval_app()
