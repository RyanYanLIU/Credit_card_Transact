import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import datetime

#1 is fraud and 0 is not -- Feature engineering
Trainset = pd.read_csv('FraudTrain.csv', index_col = 0)
Trainset['trans_date_trans_time'] = Trainset['trans_date_trans_time'].astype('datetime64[ns]')

Trainset['dob'] = Trainset['dob'].astype('datetime64[ns]')

Trainset.sort_index(inplace = True)
print(Trainset.info())

Low_lat, High_lat = Trainset['lat'].mean() - 5.0, Trainset['lat'].mean() + 5.0
Trainset['New_lat'] = Trainset['merch_lat'].apply(lambda x: 0 if (x > Low_lat) & (x < High_lat) else 1)

Low_long, High_long = Trainset['long'].mean() - 5.0, Trainset['long'].mean() + 5.0
Trainset['New_long'] = Trainset['merch_long'].apply(lambda x: 0 if (x > Low_long) & (x < High_long) else 1)

Trainset['Fraudelent on Lat&Long'] = Trainset['New_lat'] * Trainset['New_long'] #First New feature

#Hour Encoding
Trainset['Fraudelent on Hour Encoding'] = Trainset['trans_date_trans_time'].apply(lambda x: 
                                            0 if (x.hour <= 21) and (x.hour >= 5) else 1 )
#log_amt & age
Trainset['log_amt'] = np.log(Trainset['amt'])
Trainset['now'] = datetime.date.today().year
Trainset['dob_year'] = Trainset['dob'].apply(lambda x: x.year)
Trainset['age'] = Trainset['now'] - Trainset['dob_year']

Trainset['trans_date_trans_time'] = Trainset['trans_date_trans_time'].apply(lambda x: 
                                    str(x.year) + '-' + str(x.month) + '-' + str(x.day))
Trainset['trans_date_trans_time'] = pd.to_datetime(Trainset['trans_date_trans_time'])
#Trainset.set_index(['trans_date_trans_time'], inplace = True)
#Use the number of amount to testify whether it is fraud
Trainset['is_fraud_amt'] = Trainset['amt'].apply(lambda x: 0 if x <= 3000 else 
                            (1 if x <= 6000 else (2 if x <= 10000 else 3)))
Trainset['frequency'] = ''
##Feature Extraction
#trainset = Trainset[['category', 'gender', 'city_pop', 'age', 'Fraudelent on Hour Encoding', 'amt',
#                    'Fraudelent on Lat&Long' ,'job', 'log_amt', 'is_fraud_amt', 'is_fraud']]
#trainset.to_csv('trainset.csv')

#Frequency of Transactions
df_group = Trainset.copy()
df_group['Frequency'] = 1
#df_cc = df_group.groupby(['cc_num']).agg({'Frequency': 'sum'})
df_group['trans_date_trans_time'] = df_group['trans_date_trans_time'].apply(lambda x: 
                                    str(x.year) + '-' + str(x.month) + '-' + str(x.day))
df_group['trans_date_trans_time'] = pd.to_datetime(df_group['trans_date_trans_time'])

df_cc = df_group.groupby(['trans_date_trans_time', 'cc_num']).agg({'Frequency': 'sum'})
