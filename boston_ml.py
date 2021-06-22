import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.write("""
# Boston Housing Price Prediction App
This app predicts the **Boston Housing** price!
""")

st.sidebar.header('User Input Parameters')

def user_input_features(boston_df ):
  
  values_df = boston_df.describe()

  i=0
  RM = st.sidebar.slider('Average number of rooms per dwelling', values_df['RM']['min'], values_df['RM']['max'], float(values_df['RM']['mean']))
  i += 1
  LSTAT = st.sidebar.slider('% lower status of the population', values_df['LSTAT']['min'], values_df['LSTAT']['max'], float(values_df['LSTAT']['mean']))
  i += 1
  DIS = st.sidebar.slider('weighted distances to five Boston employment centres', values_df['DIS']['min'], values_df['DIS']['max'], float(values_df['DIS']['mean']))
  i += 1
  CRIM = st.sidebar.slider('per capita crime rate', values_df['CRIM']['min'], values_df['CRIM']['max'], float(values_df['CRIM']['mean']))
  i += 1
  NOX = st.sidebar.slider('nitric oxides concentration', values_df['NOX']['min'], values_df['NOX']['max'], float(values_df['NOX']['mean']))
  i += 1
  PTRATIO = st.sidebar.slider('pupil-teacher ratio by town', values_df['PTRATIO']['min'], values_df['PTRATIO']['max'], float(values_df['PTRATIO']['mean']))
  i += 1
  AGE = st.sidebar.slider('proportion of owner-occupied units built prior to 1940',values_df['AGE']['min'], values_df['AGE']['max'], float(values_df['AGE']['mean']))
  i += 1
  B = st.sidebar.slider('Bk is the proportion of blacks by town', values_df['B']['min'], values_df['B']['max'], float(values_df['B']['mean']))
  i += 1
  TAX = st.sidebar.slider('full-value property-tax rate per $10,000', values_df['TAX']['min'], values_df['TAX']['max'], float(values_df['TAX']['mean']))
  i += 1
  INDUS =st.sidebar.slider('proportion of non-retail business acres per town', values_df['INDUS']['min'],values_df['INDUS']['max'], float(values_df['INDUS']['mean']))
  i += 1
  RAD =st.sidebar.slider('index of accessibility to radial highways', values_df['RAD']['min'], values_df['RAD']['max'], float(values_df['RAD']['mean']))
  i += 1
  CHAS =st.sidebar.slider('Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)', values_df['CHAS']['min'], values_df['CHAS']['max'], float(values_df['CHAS']['mean']))
  i += 1
  ZN = st.sidebar.slider('proportion of residential land zoned for lots over 25,000 sq.ft.', values_df['ZN']['min'], values_df['ZN']['max'], float(values_df['ZN']['mean']))



  data = {'RM':RM,
        'LSTAT':LSTAT,
        'DIS': DIS,
        'CRIM': CRIM,
        'NOX': NOX,
        'PTRATIO': PTRATIO,
        'AGE': AGE,
        'B':B,
        'TAX':TAX,
        'INDUS': INDUS,
        'RAD': RAD,
        'CHAS': CHAS,
        'ZN': ZN}
  features = pd.DataFrame(data, index=[0])
  return features

boston  =datasets.load_boston()
X = boston.data
y = boston.target
boston_df = pd.DataFrame(X,columns = boston.feature_names)
boston_df['target'] = y

df = user_input_features(boston_df)

st.subheader('User Input parameters')
st.write(df)


rf = RandomForestRegressor()
sca = StandardScaler()
X_scaled  =sca.fit_transform(X)
rf.fit(X_scaled,y)

df_scaled = sca.transform(df)
prediction = rf.predict(df_scaled)



st.subheader('Price Prediction')
st.write(f"{round(float(prediction*10000))} USD")
# st.write(prediction)

