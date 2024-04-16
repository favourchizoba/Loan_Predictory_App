import pandas as pd
import streamlit as st 
import joblib
import warnings
warnings.filterwarnings('ignore')

ds = pd.read_csv('Loan_Data.csv')

# st.title('START UP PROFIT PREDICTOR APP')
#st.subheader('Built By Salmon Crusher')
st.markdown("<h1 style = 'color: #0C359E; text-align: center; font-size: 60px; font-family: Helvetica'>LOAN PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: Helvetica '>Built By 3knight</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (3).png')

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown('Banks and financial institutions receive numerous loan applications from customers seeking financial assistance for various purposes such as purchasing a home, starting a business, or funding education. However, approving loans without proper assessment of creditworthiness can lead to financial losses due to defaults.The objective is to develop a predictive model that evaluates the credit risk associated with each loan applicant and predicts whether they are qualified to receive a loan or not. This model will analyze various features or attributes of the customer and their financial history to make an informed decision.')


st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.header('Project Data')
st.dataframe(ds ,use_container_width= True)

st.sidebar.image('pngwing.com (4).png')
#line break
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

st.sidebar.subheader('User input Variable')

sel_cols = ['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Dependents', 'Property_Area',
            'Credit_History', 'Loan_Amount_Term', 'Loan_Status']

applicant_inc = st.sidebar.number_input('Applicant Income',ds['ApplicantIncome'].min(),ds['ApplicantIncome'].max())
loan = st.sidebar.number_input('LoanAmount',ds['LoanAmount'].min(),ds['LoanAmount'].max())
coapp = st.sidebar.number_input('Coapplicant Income',ds['CoapplicantIncome'].min(),ds['CoapplicantIncome'].max())
dep= st.sidebar.selectbox('Dependents', ds['Dependents'].unique())
Credit= st.sidebar.number_input('Credit_History',ds['Credit_History'].min(),ds['Credit_History'].max())
loan_term = st.sidebar.number_input('Loan_Amount_Term',ds['Loan_Amount_Term'].min(),ds['Loan_Amount_Term'].max())
property = st.sidebar.selectbox('Property_Area',ds['Property_Area'].unique())

user_input = pd.DataFrame()
user_input['ApplicantIncome'] = [applicant_inc]
user_input['LoanAmount'] = [loan]
user_input['CoapplicantIncome'] = [coapp]
user_input['Dependents'] = [dep]
user_input['Credit_History'] = [Credit]
user_input['Loan_Amount_Term'] = [loan_term]
user_input['Property_Area'] = [property]


#LoanAmount = int(user_input['LoanAmount'].values[0])

# import transformers 
applicant_inc = joblib.load('ApplicantIncome_scaler.pkl')
coapp = joblib.load('CoapplicantIncome_scaler.pkl')
property = joblib.load('Property_Area_encoder.pkl')
dep = joblib.load('Dependents_encoder.pkl')


st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.header('Input Variable')
st.dataframe(user_input,use_container_width= True)



st.header('Transformed Input Variable')
st.dataframe(user_input, use_container_width = True)


 # transform users input according to training scale and encoding
user_input['ApplicantIncome'] = applicant_inc.transform(user_input[['ApplicantIncome']])
user_input['CoapplicantIncome'] = coapp.transform(user_input[['CoapplicantIncome']]) 
user_input['Property_Area'] = property.transform(user_input[['Property_Area']])
user_input['Dependents'] = dep.transform(user_input[['Dependents']])


model = joblib.load('LoanModelll.pkl')
predict = model.predict(user_input)


    
#to have a button for the user

if st.button('Check Your Loan Approval'):
    if predict[0] == 0:
        st.error(f"Unfortunately...Your Loan of {LoanAmount} dollar will not be approved")
       # st.image('pngwing.com (7).png', width = 300)
    else:
        st.success(f"Congratulations... Your loan of {LoanAmount} dollar will be approved. Pls come to the office to process your loan approval")
      #  st.image('pngwing.com (6).png', width = 300)
        



        import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

st.markdown("<h1 style = 'color: #FF204E; text-align: center; font-family: 'Georgia'>LOAN APPROVAL PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #A0153E; text-align: center; font-family: 'cursive: font-style: italic; '>Built By TECH TITANS </h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)    

st.image('pngwing.com (3).png', use_column_width=True)

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown('Banks and financial institutions receive numerous loan applications from customers seeking financial assistance for various purposes such as purchasing a home, starting a business, or funding education. However, approving loans without proper assessment of creditworthiness can lead to financial losses due to defaults.The objective is to develop a predictive model that evaluates the credit risk associated with each loan applicant and predicts whether they are qualified to receive a loan or not. This model will analyze various features or attributes of the customer and their financial history to make an informed decision.')
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


primaryColor="#FF4B4B"  
backgroundColor="#99E6FF"
secondaryBackgroundColor="#CCFCFF"
textColor="#331133"
font="sans serif"

data= pd.read_csv('Loan_Data.csv')
st.dataframe(data.drop('Loan_ID', axis = 1))

st.sidebar.image('pngwing.com (4).png', caption = 'Welcome User')
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)

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

user_input = pd.DataFrame()
user_input['ApplicantIncome'] = [app_income]
user_input['LoanAmount'] = [loan_amt]
user_input['CoapplicantIncome'] = [coapp_income]
user_input['Dependents'] = [dep]
user_input['Property_Area'] = [prop_area]
user_input['Credit_History'] = [cred_hist]
user_input['Loan_Amount_Term'] = [loan_amt_term]

# in a situation where the loanamount was scaled you should save it to a new variable 
LoanAmount = int(user_input['LoanAmount'].values[0])

# st.markdown("<br>", unsafe_allow_html= True)
# st.divider()
# st.subheader('Users Inputs')
# st.dataframe(user_input, use_container_width = True)

# import the transformers
app_trans = joblib.load('ApplicantIncome_scaler.pkl')
co_app_trans = joblib.load('CoapplicantIncome_scaler.pkl')
prop_trans = joblib.load('Property_Area_encoder.pkl')
dep_trans =joblib.load('Dependents_encoder.pkl')



st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(user_input, use_container_width = True)




st.header('Transformed Input Variable')
st.dataframe(user_input,use_container_width = True)



# transform the users input with the imported scalers
user_input['ApplicantIncome'] = app_trans.transform(user_input[['ApplicantIncome']])
user_input['CoapplicantIncome'] = co_app_trans.transform(user_input[['CoapplicantIncome']])
user_input['Dependents'] = dep_trans.transform(user_input[['Dependents']])
user_input['Property_Area'] = prop_trans.transform(user_input[['Property_Area']])

# st.header('Transformed Input Variable')
# st.dataframe(user_input, use_container_width = True)

#st.dataframe(user_input)
model = joblib.load('LoanModell.pkl')
predict = model.predict(user_input)

if st.button('Check Your Loan Approval'):
    if predict[0] == 0:
        st.error(f"Unfortunately...Your Loan of {LoanAmount} dollar will not be approved")
        st.image('pngwing.com (7).png', width = 300)
        
    else:
        st.success(f"Congratulations... Your loan of {LoanAmount} dollar will be approved. Pls come to the office to process your loan approval")
        st.image('pngwing.com (6).png', width = 300)
        st.balloons()