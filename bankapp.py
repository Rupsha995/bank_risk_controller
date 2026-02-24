import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from joblib import load

xgb = load(r"C:\Users\rupsh\MDTM28\final_project\xgb_model.pkl")
bank_data = pd.read_csv(r"C:\Users\rupsh\MDTM28\final_project\bank_data.csv")

#setting up sidebar
st.set_page_config(page_title= "BANK_RISK_CONTROLLER", layout= "wide", page_icon="😒")
st.header("BANK_RISK_CONTROLLER APP")
st.sidebar.header("Give Specifications :")

def predict(data):
    data = pd.DataFrame(data, index=[0])
    pred = xgb.predict(data)
    return pred
    
External_source_3 = st.sidebar.number_input(
	  "External source points_3:",
	  min_value =  0.000001,
	  max_value =  55000.0,
	  placeholder = "enter number ",
	  format = "%.6f"
)
External_source_2 = st.sidebar.number_input(
	  "External source points_2:",
	  min_value =  0.0000001,
	  max_value =  1.0,
	  placeholder = "enter number ",
	  format = "%.6f"
)


Days_Birth = st.sidebar.number_input(
	"Days birth(neg):",
	min_value = -30000,
	max_value = -7000,
	placeholder = "enter number of days"
)
Days_Employed = st.sidebar.number_input(
	  "Days employed(neg):",
	  min_value = -18000,
	  max_value =  400000,
	  placeholder = "enter number of days"
)
Days_lastphone_change = st.sidebar.number_input(
	    "Days last phone change(neg):",
	     min_value = -5000,
	     max_value = 0,
	     placeholder = "enter number of days"
)
education_type  = st.sidebar.selectbox(
	    "Select education type:",
	     options=bank_data['NAME_EDUCATION_TYPE'].unique(),
)
Days_id_publish = st.sidebar.number_input(
	    "Days id publish(neg):",
	     min_value = -8000,
	     max_value = 0,
	     placeholder = "enter number of days"
)
Gender  = st.sidebar.selectbox(
	    "Gender:",
	     options=bank_data['CODE_GENDER'].unique(),
)
Annual_amount = st.sidebar.number_input(
	    "Amount annuity:",
	     min_value = 2.0,
	     max_value = 6.0,
	     placeholder = "enter number",
	     format = "%.6f"
)
Days_registration = st.sidebar.number_input(
	    "Days registration(neg):",
	     min_value = -25000,
	     max_value = 0,
	     placeholder = "enter number of days"
)
Income_Type  = st.sidebar.selectbox(
	    "Income type:",
	     options=bank_data['NAME_INCOME_TYPE'].unique(),
)
region_rating  = st.sidebar.selectbox(
	    "Region rating:",
	     options=bank_data['REGION_RATING_CLIENT_W_CITY'].unique(),
)

Total_area = st.sidebar.number_input(
	    "Total Area Mode:",
	     min_value = 0.0001,
	     max_value = 1.0,
	     placeholder = "enter number",
	     format = "%.4f"
)

input_dict = {
	"EXT_SOURCE_3":External_source_3,
	"EXT_SOURCE_2" :External_source_2,
    "DAYS_BIRTH":Days_Birth,
	"DAYS_EMPLOYED":Days_Employed,	
    "DAYS_LAST_PHONE_CHANGE":Days_lastphone_change, 	
    "NAME_EDUCATION_TYPE":education_type,
    "DAYS_ID_PUBLISH":Days_id_publish,
    "CODE_GENDER":Gender,
	"AMT_ANNUITY_x":Annual_amount,
    "DAYS_REGISTRATION":Days_registration,	
    "NAME_INCOME_TYPE":Income_Type,
    "REGION_RATING_CLIENT_W_CITY":region_rating,
	"TOTALAREA_MODE":Total_area
}
st.header('result:')

if st.sidebar.button("Predict", type='primary'):
    res = predict(input_dict)
    st.success(res)




