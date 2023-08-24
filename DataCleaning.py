# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 01:49:03 2021

@author: KUMAR YASHASWI
"""

import os, zipfile
import pandas as pd

dir_name = 'C:/Users/KUMAR YASHASWI/Documents/item_000027057'
extension = ".zip"

os.chdir(dir_name) # change directory from working dir to dir with files

for item in os.listdir(dir_name): # loop through items in dir
    if item.endswith(extension): # check for ".zip" extension
        file_name = os.path.abspath(item) # get full path of files
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object
        zip_ref.extractall(dir_name) # extract file to dir
        zip_ref.close() # close file
        #os.remove(file_name) # delete zipped file
        
        
for item in os.listdir(dir_name): # loop through items in dir
    if item.endswith(extension): # check for ".zip" extension
        file_name = os.path.abspath(item)
        os.remove(file_name) # delete zipped file
        
        

for count, filename in enumerate(os.listdir("item_000027057")):
        print(filename)
        x=filename.split('_')
        x[4]=x[4].replace(".csv", "")
        dst = x[0]+'_'+x[1]+'_'+x[2]+'_'+x[3]+'_'+x[4]
        src ='item_i000056738/'+ filename
        dst ='item_i000056738/'+ dst
        
        # rename() function will
        # rename all the files
        os.rename(src, dst)
data=[]
fold= 'item_000027057/'
SPPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/SP.csv')
SPPrice['quote_date'] = pd.to_datetime(SPPrice['Date'])
SPPrice['Date'] = pd.to_datetime(SPPrice['Date'])
TresRate=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/RiskRate.csv')
TresRate['Date'] = pd.to_datetime(TresRate['Date'])
TresRate['risk_perc']=TresRate['risk_perc']/100
SPPrice = pd.merge(SPPrice,TresRate[['Date', 'risk_perc']],on='Date')
SPPrice['SPClose'] = SPPrice['Close']
SPPrice = SPPrice[['quote_date', 'Close','risk_perc']]
for count, filename in enumerate(os.listdir("item_000027057")):
        print(filename)
        optionPrice=pd.read_csv(fold+filename)
        optionPrice['quote_date'] = pd.to_datetime(optionPrice['quote_date'])
        optionPrice=optionPrice[['quote_date', 'expiration', 'strike',
       'option_type', 'close', 'trade_volume', 'bid_1545', 'ask_1545',
       'underlying_bid_1545', 'underlying_ask_1545',
       'implied_underlying_price_1545', 'active_underlying_price_1545',
       'implied_volatility_1545', 'bid_eod',
       'ask_eod']]
        optionPrice = pd.merge(optionPrice,SPPrice[['quote_date', 'Close','risk_perc']],on=['quote_date'])
        mask = (optionPrice['option_type']=='C')
        mask =  ((optionPrice['strike']==3500) | (optionPrice['strike']==4000) | (optionPrice['strike']==4500) | (optionPrice['strike']==2000) | (optionPrice['strike']==2500) | (optionPrice['strike']==3000)) & (optionPrice['expiration']=='2020-12-18')
        optionPrice = optionPrice.loc[mask]
        data.append(optionPrice)
 
 
len(optionPrice['quote_date'].unique())       
 
dfs = [df.set_index('quote_date') for df in data]
dfs1=pd.concat(dfs) 
mask = (dfs1['strike']==2500)  & (dfs1['option_type']=='C')
optionPrice = dfs1.loc[mask]
optionPrice = optionPrice.reset_index()
optionPrice=optionPrice.loc[optionPrice.groupby(['quote_date'])['trade_volume'].idxmax()]


     
count=dfs1[['expiration','strike','option_type','quote_date']].groupby(['expiration', 'strike','option_type']).count()
 

count=count.to_frame()

count1=dfs1[['expiration','strike','option_type','trade_volume']].groupby(['expiration', 'strike','option_type']).sum()

cnt=count.join(count1)






data=[]
fold= 'item_000027057/'
SPPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/SP.csv')
SPPrice['quote_date'] = pd.to_datetime(SPPrice['Date'])
SPPrice['Date'] = pd.to_datetime(SPPrice['Date'])
TresRate=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/RiskRate.csv')
TresRate['Date'] = pd.to_datetime(TresRate['Date'])
TresRate['risk_perc']=TresRate['risk_perc']/100
SPPrice = pd.merge(SPPrice,TresRate[['Date', 'risk_perc']],on='Date')
SPPrice['SPClose'] = SPPrice['Close']
SPPrice = SPPrice[['quote_date', 'Close','risk_perc']]
for count, filename in enumerate(os.listdir("item_000027057")):
        print(filename)
        optionPrice=pd.read_csv(fold+filename)
        optionPrice['quote_date'] = pd.to_datetime(optionPrice['quote_date'])
        optionPrice=optionPrice[['quote_date', 'expiration', 'strike',
       'option_type', 'close', 'trade_volume', 'bid_1545', 'ask_1545',
       'underlying_bid_1545', 'underlying_ask_1545',
       'implied_underlying_price_1545', 'active_underlying_price_1545',
       'implied_volatility_1545', 'bid_eod',
       'ask_eod']]
        optionPrice = pd.merge(optionPrice,SPPrice[['quote_date', 'Close','risk_perc']],on=['quote_date'])
        #mask = (optionPrice['option_type']=='P')
        #mask =  ((optionPrice['strike']==3500) | (optionPrice['strike']==4000) | (optionPrice['strike']==4500) | (optionPrice['strike']==2000) | (optionPrice['strike']==2500) | (optionPrice['strike']==3000)) & (optionPrice['expiration']=='2020-12-18')
        #optionPrice = optionPrice.loc[mask]
        data.append(optionPrice)
        
dfs = [df.set_index('quote_date') for df in data]
dfs1=pd.concat(dfs) 
dfs1=dfs1.reset_index()
dfs1['expiration'] = pd.to_datetime(dfs1['expiration'])
dfs1['month'] = dfs1['expiration'].dt.month
optionPrice['quote_date'] = pd.to_datetime(optionPrice['quote_date'])
dfs1['TimeToMaturity']=dfs1['expiration'] - dfs1['quote_date']
dfs1['TimeToMaturity']=dfs1['TimeToMaturity'].dt.days
#optionPrice['TimeToMaturity'] = optionPrice['TimeToMaturity'].dt.days.astype('int16')

mask =  ((dfs1['month']==2) | (dfs1['month']==4) | (dfs1['month']==6) | (dfs1['month']==8) | (dfs1['month']==12))  & (dfs1['option_type']=='C')
#mask =  (dfs1['option_type']=='C')
dfs1 = dfs1.loc[mask]
dfs1=dfs1.reset_index()
dfs1=dfs1.sort_values(by=['quote_date'])

dfs1=dfs1.loc[dfs1.groupby(['quote_date','strike'])['TimeToMaturity'].idxmin()]
dfs1=dfs1.loc[dfs1.groupby(['quote_date'])['trade_volume'].idxmax()]
#len(optionPrice['quote_date'].unique())
count=dfs1.groupby(['expiration', 'strike','option_type']).size()
optionPrice=dfs1
optionPrice=optionPrice.reset_index()
plt.plot(optionPrice['implied_volatility_1545'])
plt.plot(optionPrice['implied_underlying_price_1545'])
count=count.to_frame()

count1=dfs1[['quote_date','expiration','strike','option_type','trade_volume','TimeToMaturity']].groupby(['quote_date','expiration', 'strike','option_type']).sum()
count1=
cnt=count.join(count1)

mask = (dfs1['strike']==2500)  & (dfs1['option_type']=='C')
optionPrice = dfs1.loc[mask]
optionPrice = optionPrice.reset_index()
