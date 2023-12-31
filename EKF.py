# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 02:29:37 2021

@author: KUMAR YASHASWI
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy import dot
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
import scipy
from scipy.stats import norm
import math
import pandas as pd
import datetime
from scipy.linalg import sqrtm
import matplotlib.pylab as plt






def extended_KF_predict(X,S,ks,T,uK,P,Q,R):
     Xk=X[0]
     XK_k = f(Xk,uK,P,Q)[0]
     P_k = f(Xk,uK,P,Q)[1]
     #print(S)
     #print(T)
     #print(XK_k)
     y1_k = BS(XK_k,S,ks,T)
     #print(y1_k)
     #print(' ')
     H = Jacobian(XK_k,S,ks,T)
     #print(H)
     #print(y1_k)
     H=np.array(H)
     #print(' ')
     H= np.expand_dims(H, 0)
     F_K = dot(H, dot(P_k, H.T)) + R
     return (y1_k,P_k,F_K,XK_k,H)

def Jacobian(XK_k,S,k,T):
     X=[r/100 for r in XK_k]
     X[0]=np.max([X[0],0.0000001])
     dk_plus = (np.log(S/k) + (X[1] + (X[0]/2))*T)/np.sqrt(X[0]*T)
     dk_minus = dk_plus - np.sqrt(X[0]*T)
     Vega= (0.5*S*np.sqrt(T)*scipy.stats.norm(0, 1).pdf(dk_plus))/(np.sqrt(X[0]))
     #print(S)
     #print(math.exp(-XK_k[1]*T))
     rho =  k*T*(math.exp(-X[1]*T))*scipy.stats.norm(0, 1).pdf(dk_minus)
     return (Vega,rho)

  
def f(Xn,uN,Sigma_n,Q):
     omega= 0
     alpha = 0
     beta = 1
     #print('wwww')
     #print(Xn)
     XNZero = omega + alpha*(uN * uN) + beta*Xn[0]
     A=[]
     A.append([beta*beta,0])
     A.append([0,1])
     Sigma_N =dot(A,Sigma_n) + Q
     XNOne=Xn[1]
     XN=(XNZero,XNOne)
     return (XN,Sigma_N)
    
def extended_KF_update(y_K,y1_k, P_k, F_k, XK_k,H):
     #print("H matrix   ",H)
     print("Real Price  ",y_K)
     print("Estimated Value  ",y1_k)
     b_K = y_K - y1_k
     #F_k=np.array((F_k))
     #F_k= np.expand_dims(F_k, 0)
     #print("State Cov", P_k)
     K_K = dot(P_k, H.T)*(1/F_k)
     #print('Difference   ',b_K)
     #print("State  ",XK_k)
     #print("Kalman Gain  ",K_K)  
     #print("Kalman matrix   ",dot(K_K,b_K).T)
     XK_K = XK_k + dot(K_K,b_K).T
     #print(dot(K_K,b_K).T)
     P_final = P_k - dot(K_K , dot(H,P_k))
     #print("New State  ",XK_K)
     #print(dot(K_K,b_K).T)
     #print(P_final)
     #print(' ')
     return (XK_K,P_final)
     
def BS(XK_k,S,k,T):
     X=[r/100 for r in XK_k]
     X[0]=np.max([X[0],0.0000001])
     dk_plus = (np.log(S/k) + (X[1] + (X[0]/2))*T)/np.sqrt(X[0]*T)
     dk_minus = dk_plus - np.sqrt(X[0]*T)
     rhs = math.exp(-X[1]*T)
     c = S*norm.cdf(dk_plus) - k*rhs*norm.cdf(dk_minus)
     return c








def dataset():
    market_price=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSEI.csv')
    market_price['Date'] = pd.to_datetime(market_price['Date'])
    #optionPrice['Date'] = pd.to_datetime(optionPrice['Date'])
    #optionPrice['Expiry'] = pd.to_datetime(optionPrice['Expiry'])
    #optionPrice = pd.merge(optionPrice,market_price[['Date', 'Close']],on='Date')
    optionPrice['expiration'] = pd.to_datetime(optionPrice['expiration'])
    optionPrice['TimeToMaturity']=optionPrice['expiration'] - optionPrice['quote_date']
    optionPrice['TimeToMaturity']=optionPrice['TimeToMaturity'].dt.days
    #optionPrice['TimeToMaturity'] = optionPrice['TimeToMaturity'].dt.days.astype('int16')
    
    optionPrice=optionPrice.reset_index()
    
    optionPrice=optionPrice.sort_values(by=['quote_date'])
    optionPrice['StockPR']=optionPrice['Close']
    optionPrice['PriceChange'] = optionPrice['StockPR'].pct_change()
    optionPrice=optionPrice.bfill(axis ='rows')
    #optionPrice=optionPrice.loc[240:]
    optionPrice['Set']='Train'
    optionPrice['Set'] = np.where(optionPrice['index']< 320, 'Train', 'Test')
    
    del optionPrice['index']
    optionPrice['OptionPR']=(optionPrice['bid_1545']+optionPrice['ask_1545'])/2
    optionPrice['StrikePrice']=optionPrice['strike']
    
    
    EKF_prices=[]
    EKF_prices_test=[]
    real_prices=[]
    real_prices_test=[]
    vol_EKF_train =[]
    risk_EKF_train = []
    vol_EKF_test =[]
    risk_EKF_test = []
    stock_price=[optionPrice.loc[0,'StockPR']]
    xbar_0 = [0.19/(252 ** 0.5),0.0256]
    P_0 = []
    P_0.append([0.001,0])
    P_0.append([0,0.1])
    x_temp= xbar_0
    P_temp = P_0
    Q = []
    Q.append([0.001,0])
    Q.append([0,0.5])
    R= [930]
    
    P_w=Q
    P_v=R
    x_temp=np.array([x_temp])
    
    for e in range(1,optionPrice.shape[0]-30):
        if(optionPrice.loc[e,'Set']=='Test'):
            price_bfr=optionPrice.loc[e-1,'StockPR']
            price= GBM(np.array(stock_price[-60:]),price_bfr)
            price_chg= (price-price_bfr)/price_bfr
            PredResult = extended_KF_predict(x_temp,price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],price_chg,P_temp, Q,R)   
            pred_value= PredResult[0]
            pred_state = PredResult[3]
            EKF_prices_test.append(pred_value)
            real_prices_test.append(optionPrice.loc[e,'OptionPR'])
            vol_EKF_test.append(pred_state[0])
            risk_EKF_test.append(pred_state[1])
        PredResult=extended_KF_predict(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],P_temp, Q,R)
        
        UpdateResult = extended_KF_update(optionPrice.loc[e,'OptionPR'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        x_temp= UpdateResult[0]
        P_temp = UpdateResult[1]
        state=x_temp.tolist()[0]
        stock_price.append(np.log(optionPrice.loc[e,'StockPR']/optionPrice.loc[e-1,'StockPR']))
        if(optionPrice.loc[e,'Set']=='Train'):
            EKF_prices.append(BS(x_temp.tolist()[0],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
            real_prices.append(optionPrice.loc[e,'OptionPR'])
            vol_EKF_train.append(state[0])
            risk_EKF_train.append(state[1])
 
a=optionPrice['quote_date'].tolist()
a=a[320:optionPrice.shape[0]-30]


plt.figure(figsize = (15, 4))   
#plt.plot(UKF_prices_test)
#plt.plot(real_prices_test)
#plt.plot(a,real_prices_test,label="RL Portfolio-Return")
plt.plot(a,EKF_prices_test,linestyle='-', marker='o',label="EKF Predicted Price")
plt.plot(a,real_prices_test,linestyle='-', label="Real Option Price")
plt.plot(label="UKF")
plt.legend(loc="upper left")
plt.xlabel('Quote Date')
plt.ylabel('Option Prices')
plt.title("Strike Price=2500")
plt.grid()
plt.show()        
 
RMSE(real_prices_test,EKF_prices_test,2500)-0.035158277152056536
    
plt.plot(EKF_prices_test)
plt.plot(real_prices_test)

plt.plot(risk_pf_test)

plt.plot([a*(252 ** 0.5) for a in vol_pf_train])
plt.plot(optionPrice['implied_volatility_1545'])