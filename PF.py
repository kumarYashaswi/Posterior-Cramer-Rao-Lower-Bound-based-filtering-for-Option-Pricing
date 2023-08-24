# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 02:29:49 2021

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


def normal_dist1(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def normal_dist(x , mean ,var):
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def multivariate_pdf(vector, mean, cov):
    quadratic_form = np.dot(np.dot(vector-mean,np.linalg.inv(cov)),np.transpose(vector-mean))
    return np.exp(-.5 * quadratic_form)/ (2*np.pi * np.linalg.det(cov))


def degneracy(w_K):
    weight_sqr=0
    for i in range(0,N):
       weight_sqr= weight_sqr + (w_K[i]*w_K[i])
    return weight_sqr

def Particle_Filter(x_part,uN,Q,w_k,N,y_k,S,k,T,e,R):
    xK_part=[]
    w_K=[]
    res_list=[]
    stateMean=[]
    state_cov= np.array([[0, 0], [0, 0]], dtype=np.float)
    weight_sum=0
    for i in range(0,N):
       state = f(x_part[i],uN,Q,Q)[0]
       xK_part.append(state)
       #print(i," BS ",BS(state,S,k,T))
       #print(i," BS ",state[0]*T)
       price_diff= y_k - BS(state,S,k,T)
       
       w_K.append(w_k[i]*normal_dist(price_diff,0,R))
       stateMean.append([w_K[i] * r for r in xK_part[i]])
       weight_sum= weight_sum + w_K[i]
    #print(i," BS ",weight_sum)
    state_mean = np.array(stateMean).sum(axis=0).tolist()
    
    w_K = [x / weight_sum for x in w_K]
    state_mean = [x / weight_sum for x in state_mean]
    
    for i in range(0,N):
        error=[x1 - x2 for (x1, x2) in zip(xK_part[i], state_mean)]
        error_matrix=np.expand_dims(np.array(error),axis=1)
        state_cov= np.add(state_cov,w_K[i]*np.dot(error_matrix,error_matrix.T))
    
    w_sqr= degneracy(w_K)
    Bessel=(1-w_sqr)
    state_cov=(1/(Bessel))*state_cov 
    n_e= N/2
    Neff = 1/w_sqr
    #print(n_e)
    #print(" PArticle ",w_K)
    xK_resample,w_K = resample_real(xK_part,w_K,N,Neff,n_e)
    for i in range(0,N):
        res_list.append([w_K[i] * r for r in xK_resample[i]])
    x_K = np.array(res_list).sum(axis=0).tolist()
    
            
    #print("State  ",x_K)
    #print(" PArticle ",xK_resample)
    #print(" Matrix ",state_cov)
    #print("Weight  ",w_K[5])
    return (x_K,xK_resample,state_cov,w_K)

def Particle_Filter_Pred(x_part,uN,Q,w_k,N,y_k,S,k,T,e,R):
    xK_part=[]
    w_K=[]
    res_list=[]
    stateMean=[]
    state_cov= np.array([[0, 0], [0, 0]], dtype=np.float)
    weight_sum=0
    for i in range(0,N):
       state = f(x_part[i],uN,Q,Q)[0]
       xK_part.append(state)
       #print(i," BS ",BS(state,S,k,T))
       #print(i," BS ",state[0]*T)
       #price_diff= y_k - BS(state,S,k,T)
       
       w_K.append(w_k[i])
       stateMean.append([w_K[i] * r for r in xK_part[i]])
       weight_sum= weight_sum + w_K[i]
    #print(i," BS ",weight_sum)
    state_mean = np.array(stateMean).sum(axis=0).tolist()
    
    w_K = [x / weight_sum for x in w_K]
    state_mean = [x / weight_sum for x in state_mean]
    
    
            
    #print("State  ",x_K)
    #print(" PArticle ",xK_resample)
    #print(" Matrix ",state_cov)
    #print("Weight  ",w_K[5])
    return (state_mean)

def resample_real(xK,wK,N,Neff,n_e):
    if Neff < n_e:
        (xK,wK) = resample_particle(xK,wK,N)
        return (xK,wK)
    else:
        return (xK,wK)

def resample_particle(x_part,w_K,N):
    #print(w_K)
    #print(" ")
    w_K.insert(0,0)
    w_K=np.cumsum(w_K)
    #print(w_K)
    s = np.random.uniform(0,1,N)
    #print(s)
    ind=[]
    new_particle =[] 
    for i in range(0,N):
        particle_no=s[i]
        for j in range(1,len(w_K)):
            if particle_no<=w_K[j]:
              ind.append(j-1)
              break
    for k in range(0,N):
        new_particle.append(x_part[ind[k]])
    new_weight=[1/N] * N
    return (new_particle,new_weight)


def Jacobian(XK_k,S,k,T):
     X=[r/100 for r in XK_k]
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
    
(XK_k,S,k,T)=(xbar_0,stock_price[0],2500,715)
     
def BS(XK_k,S,k,T):
     X=[r/100 for r in XK_k]
     X[0]=np.max([X[0],0.0000001])
     dk_plus = (np.log(S/k) + (X[1] + (X[0]/2))*T)/np.sqrt(X[0]*T)
     dk_minus = dk_plus - np.sqrt(X[0]*T)
     rhs = math.exp(-X[1]*T)
     c = S*norm.cdf(dk_plus) - k*rhs*norm.cdf(dk_minus)
     return c

xbar_0 = [0.19/(252 ** 0.5),0.02556]
BS(xbar_0,stock_price[0],2500,715)

def dataset():
    #optionPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSE.csv')
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
    
    PF_prices=[]
    PF_prices_test=[]
    real_prices=[]
    real_prices_test=[]
    vol_pf_train =[]
    risk_pf_train = []
    vol_pf_test =[]
    risk_pf_test = []
    stock_price=[optionPrice.loc[0,'StockPR']]
    xbar_0 = [0.19/(252 ** 0.5),0.02556]
    P_0 = []
    P_0.append([0.001,0])
    P_0.append([0,0.001])
    x_temp= xbar_0
    P_temp = P_0
    Q = []
    Q.append([0.001,0])
    Q.append([0,0.1])
    R= [6400]
    L=5
    (xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
    P_w=Q
    P_v=R
    x_temp=np.array([x_temp])
    N=1000
    x_particle= np.random.multivariate_normal(xbar_0, P_0, N)
    w_particle= [1/N] * N
    
    for e in range(1,optionPrice.shape[0]-30):
        #R=std(x_particle,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'])
        #print(x_temp)- y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v
        print(e)
        if(optionPrice.loc[e,'Set']=='Test'):
            price_bfr=optionPrice.loc[e-1,'StockPR']
            price= GBM(np.array(stock_price[-60:]),price_bfr)
            print(price)
            price_chg= (price-price_bfr)/price_bfr
            
            PredResult = Particle_Filter_Pred(x_particle,price_chg,Q,w_particle,N,optionPrice.loc[e,'OptionPR'],price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e,R[0])   
            PF_prices_test.append(BS(PredResult,price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
            real_prices_test.append(optionPrice.loc[e,'OptionPR'])
            vol_pf_test.append(PredResult[0])
            risk_pf_test.append(PredResult[1])
            
        (state,x_particle,particle_cov,w_particle)=Particle_Filter(x_particle,optionPrice.loc[e,'PriceChange'],Q,w_particle,N,optionPrice.loc[e,'OptionPR'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e,R[0])
        
        #print(max(w_particle))
        #print(min(w_particle))
        #print(sum(w_particle))
        #print(particle_cov)
        #print(" ")
        stock_price.append(np.log(optionPrice.loc[e,'StockPR']/optionPrice.loc[e-1,'StockPR']))
        if(optionPrice.loc[e,'Set']=='Train'):
            PF_prices.append(BS(state,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
            real_prices.append(optionPrice.loc[e,'OptionPR'])
            vol_pf_train.append(state[0])
            risk_pf_train.append(state[1])
            if not state[0]:
                print(e)
                print(state[0])
                print(state[1])

a=optionPrice['quote_date'].tolist()
a=a[320:optionPrice.shape[0]-30]


plt.figure(figsize = (15, 4))   
#plt.plot(UKF_prices_test)
#plt.plot(real_prices_test)
#plt.plot(a,real_prices_test,label="RL Portfolio-Return")
plt.plot(a,PF_prices_test,linestyle='-', marker='o',label="PF Predicted Price")
plt.plot(a,real_prices_test,linestyle='-', label="Real Option Price")
plt.legend(loc="upper left")
plt.xlabel('Quote Date')
plt.ylabel('Option Prices')
plt.title("Strike Price=2500")
plt.grid()
plt.show() 

def RMSE(real,predict,strike):
    N=len(real)
    error=[(a-b)*(a-b) for (a,b) in zip(real,predict)]
    error=sum(error)
    error=error/N
    error=error/(strike*strike)
    return np.sqrt(error)
    
RMSE(real_prices_test,PF_prices_test,2500)-0.03270111230226948
