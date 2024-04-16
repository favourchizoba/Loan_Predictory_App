import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('Loan_Data.csv')

st.markdown("<h1 style = 'color: #FF204E; text-align: center; font-size: 60px; font-family: Georgia'>LOAN PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #A0153E; text-align: center; font-family: italic'>BUILT BY TECH TITANS </h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)

# #add image
st.image('pngwing.com (3).png')

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)


st.markdown("<p>Banks and financial institutions receive numerous loan applications from customers seeking financial assistance for various purposes such as purchasing a home, starting a business, or funding education. However, approving loans without proper assessment of creditworthiness can lead to financial losses due to defaults. The objective is to develop a predictive model that evaluates the credit risk associated with each loan applicant and predicts whether they are qualified to receive a loan or not. This model will analyze various features or attributes of the customer and their financial history to make an informed decision</p>", unsafe_allow_html = True)

st.sidebar.image('png7.png',caption = 'Welcome User')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)



# primaryColor="#FF4B4B"  
# backgroundColor="#99E6FF"
# secondaryBackgroundColor="#CCFCFF"
# textColor="#331133"
# font="sans serif"


st.sidebar.subheader('User Input Variables')
sel_columns = ['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Dependents',
                'Property_Area', 'Credit_History', 'Loan_Amount_Term' , 'Loan_Status']

app_income = st.sidebar.number_input('Applicant Income', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())
loan_amt = st.sidebar.number_input('Loan Amount', data['LoanAmount'].min(), data['LoanAmount'].max())
coapp_income = st.sidebar.number_input('CoApplicant Income', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
dep = st.sidebar.selectbox('Dependents', data['Dependents'].unique())
prop_area = st.sidebar.selectbox('Property Area', data['Property_Area'].unique())
cred_hist = st.sidebar.number_input('Credit History', data['Credit_History'].min(), data['Credit_History'].max())
loan_amt_term = st.sidebar.number_input('Loan Amount Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())

#users input
input_var = pd.DataFrame()
input_var['ApplicantIncome'] = [app_income]
input_var['LoanAmount'] = [loan_amt]
input_var['CoapplicantIncome'] = [coapp_income]
input_var['Dependents'] = [dep]
input_var['Property_Area'] = [prop_area]
input_var['Credit_History'] = [cred_hist]
input_var['Loan_Amount_Term'] = [loan_amt_term]

# in a situation where the loanamount was scaled you should save it to a new variable 
LoanAmount = int(input_var['LoanAmount'].values[0])

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)

# import the transformers
app_trans = joblib.load('ApplicantIncome_scaler.pkl')
co_app_trans = joblib.load('CoapplicantIncome_scaler.pkl')
prop_trans = joblib.load('Property_Area_encoder.pkl')
dep_trans =joblib.load('Dependents_encoder.pkl')

# transform the users input with the imported scalers
input_var['ApplicantIncome'] = app_trans.transform(input_var[['ApplicantIncome']])
input_var['CoapplicantIncome'] = co_app_trans.transform(input_var[['CoapplicantIncome']])
input_var['Property_Area'] = prop_trans.transform(input_var[['Property_Area']])
input_var['Dependents'] = dep_trans.transform(input_var[['Dependents']])

# st.header('Transformed Input Variable')
# st.dataframe(input_var, use_container_width = True)


model = joblib.load('LoanModel.pkl')
predict = model.predict(input_var)

if st.button('Check Your Loan Approval Status'):
    if predict[0] == 0:
        st.error(f"Unfortunately...Your Loan of {LoanAmount} dollar has been declined")
        st.image('pngwing.com (7).png', width = 300)
    else:
        st.success(f"Congratulations... Your loan of {LoanAmount} dollar has been approved. Pls come to the office to process your loan")
        st.image('pngwing.com (6).png', width = 300)
        st.balloons()
        