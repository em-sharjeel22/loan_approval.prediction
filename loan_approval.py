# MUHAMMAD SHARJEEL (CF-112)
# ============================================================
#  LOAN APPROVAL PREDICTION: 
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── LOAD DATA ────────────────────────────────────────────────
df = pd.read_csv('loan_data.csv')

#  DATA PREPROCESSING AND CLEANING:
print("Shape:", df.shape)
print("Duplicates:", df.duplicated().sum())
print("Nulls:\n", df.isnull().sum())
print(df.describe())
print("Columns:", df.columns.tolist())

# DROP & RENAME COLUMNS:
df.drop(columns=['person_age', 'person_gender','person_education', 'person_emp_exp','loan_percent_income'],inplace=True)
df.rename(columns={
    'person_income':'Income',
    'person_home_ownership':'Ownership',
    'loan_amnt':'Loan Amount',
    'loan_intent':'Loan Intent',
    'loan_int_rate':'Interest Rate',
    'cb_person_cred_hist_length':'Credit.Hist',
    'credit_score':'Cr.Score',
    'previous_loan_defaults_on_file':'Previous Loans',
    'loan_status':'Approval'},
        inplace=True)


print(df.head(4))

print(df.describe())

print("Ownership values :", df['Ownership'].unique())

print("Approval values  :", df['Approval'].unique())

print("Previous Loans   :", df['Previous Loans'].unique())

print("Cr.Score sample  :", df['Cr.Score'].unique()[:10])

print("Credit.Hist sample:", df['Credit.Hist'].unique()[:10])

#CORRELATION HEATMAP :
plt.figure(figsize=(8, 6))
plt.title('Heatmap for Co-relation')
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.tight_layout()
plt.show()

#  OUTLIER REMOVAL & TRAIN-TEST SPLIT:

plt.figure(figsize=(8,6))
plt.title('Heatmap for co-relation')
sns.heatmap(df.corr(numeric_only=True),annot=True)

sns.boxplot(df['Income'])
print("Rows with Income > 400k:", len(df[df['Income'] > 400000]))
df = df[df['Income'] < 400000]
sns.boxplot(df['Income'])
print("Shape after outlier removal:", df.shape)

sns.boxplot(df['Interest Rate'])
df = df[df['Interest Rate'] < 19]
sns.boxplot(df['Interest Rate'])

# Features / Target
x = df.drop(columns=['Approval']) 
y = df['Approval']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#  ENCODING & SCALING  (single clean ColumnTransformer):

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer

categorical_cols  = ['Ownership', 'Loan Intent']   
numerical_cols    = ['Income', 'Loan Amount', 'Interest Rate', 'Credit.Hist', 'Cr.Score']
binary_col        = ['Previous Loans']

column_transform = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    (StandardScaler(),                       numerical_cols),
    (OrdinalEncoder(),                       binary_col),
    remainder='drop'
)

x_train_trans = column_transform.fit_transform(x_train)
x_test_trans  = column_transform.transform(x_test)

#  MODELING:
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.ensemble      import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from xgboost               import XGBClassifier
from sklearn.metrics       import (accuracy_score, precision_score,
                                   recall_score, f1_score,
                                   confusion_matrix, ConfusionMatrixDisplay)

lr  = LogisticRegression(max_iter=1000)
dt  = DecisionTreeClassifier()
knn = KNeighborsClassifier()
rfc = RandomForestClassifier()
bgc = BaggingClassifier()
etc = ExtraTreesClassifier()
xgb = XGBClassifier(eval_metric='logloss', verbosity=0)   # FIXED: suppress XGB warnings

classifiers = {
    'LR' : lr,
    'DT' : dt,
    'KNN': knn,
    'RFC': rfc,
    'XGB': xgb
}

# ── TRAINING FUNCTION ─────────────────────────────────────────
def train_classifier(name, classifier, x_train, x_test, y_train, y_test):
    # Training:
    classifier.fit(x_train, y_train)

    # Prediction:
    y_pred = classifier.predict(x_test)

    # Evaluation Metrics:
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title(f'Confusion Matrix for {name}')   # FIXED: now uses the 'name' parameter
    plt.xlabel("PREDICTED LABELS")
    plt.ylabel("TRUE LABELS")
    plt.tight_layout()
    plt.show()

    return accuracy, precision, recall, f1

accuracy_scores  = []
precision_scores = []
recall_scores    = []
f1_scores        = []

for name, clf in classifiers.items():
    print(f'\nTraining {name}...')
    acc, prec, rec, f1 = train_classifier(
        name, clf, x_train_trans, x_test_trans, y_train, y_test)  # FIXED: pass name

    print(f'Name      : {name}')
    print(f'Accuracy  : {acc:.4f}')
    print(f'Precision : {prec:.4f}')
    print(f'Recall    : {rec:.4f}')
    print(f'F1 Score  : {f1:.4f}')
    print('----------------------------')

    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)

print("Best Model Selection based on Accuracy:")

results         = {name: acc for name, acc in zip(classifiers.keys(), accuracy_scores)}
best_model_name = max(results, key=results.get)
print("\nBest Model:", best_model_name)

print("FINAL PIPELINE")
from sklearn.pipeline import Pipeline

final_model = Pipeline(steps=[
    ("preprocessing", column_transform),
    ("model",  classifiers[best_model_name])
])

final_model.fit(x_train, y_train)

print("REAL-LIFE PREDICTION ")

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

prediction = final_model.predict(new_person)
if prediction[0] == 1:
    print("\nResult: Loan APPROVED ✓")
else:
    print("\nResult: Loan REJECTED ✗")
    
    
    
    
import pickle
file_name = 'final_loan_model.sav'    
pickle.dump(final_model, open(file_name, 'wb')) 

loaded_model = pickle.load(open('final_loan_model.sav', 'rb'))


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

prediction = loaded_model.predict(new_person)
if prediction[0] == 1:
    print("\nResult: Loan APPROVED ✓")
else:
    print("\nResult: Loan REJECTED ✗")
