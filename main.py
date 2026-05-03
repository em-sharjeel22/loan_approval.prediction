import numpy as np
import pandas as pd
import pickle

import os

model_path = "final_loan_model.sav"

if not os.path.exists(model_path):
    print("Model file not found!")
else:
    loaded_model = pickle.load(open('"C:/Users/M.LAPTOP/Downloads/loan_dataset/final_loan_model.sav', 'rb'))



# loaded_model = pickle.load(open("final_loan_model.sav", 'rb'))

new_person = pd.DataFrame([{
    "Income":     int(input("Enter Income: ")),
    "Ownership":  input("Enter Ownership (RENT/OWN/MORTGAGE): ").upper() ,
    "Loan Amount": int(input("Enter Loan Amount: ")),
    "Loan Intent":  input("Enter Loan Intent (PERSONAL/EDUCATION/MEDICAL/VENTURE/HOME/DEBTCONSOLIDATION): ").upper() ,
    "Interest Rate": float(input("Enter Interest Rate: ")),
    "Credit.Hist":   float(input("Enter Credit History Length: ")),
    "Cr.Score": int(input("Enter Credit Score: ")),
    "Previous Loans":input("Enter Previous Loans (Yes/No): ").title()
}])


try:
        expected_cols = loaded_model.named_steps['preprocessor'].feature_names_in_
except AttributeError:
        expected_cols = loaded_model.feature_names_in_
 
    # Strip whitespace from both sides for comparison, then map to expected names
        expected_stripped = {col.strip(): col for col in expected_cols}
new_person.columns = [expected_stripped.get(c.strip(), c) for c in new_person.columns]

prediction = loaded_model.predict(new_person)
if prediction[0] == 1:
    print("\nResult: Loan APPROVED ✓")
else:
    print("\nResult: Loan REJECTED ✗")

        

