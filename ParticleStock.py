# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:11:02 2020

@author: Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:50:36 2020

@author: Kumar
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:50:36 2020

@author: Kumar
"""
import math
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

from __future__ import print_function
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException

'''
def cramer_pf_PF(x_part,uN,Q,w_k,N,y_k,S,k,T,e):
    xK_part=[]
    w_K=[]
    weight_sum=0
    res_list = []
    for i in range(0,N):
       state = f(x_part[i],uN,Q,Q)[0]
       xK_part.append(state)
       price_diff= y_k - BS(state,S,k,T)
       if(not price_diff):
           price_diff = 0.02
       w_K.append(w_k[i]*normal_dist(price_diff,0,1))
       weight_sum= weight_sum + w_K[i]
    w_K = [x / weight_sum for x in w_K]
    xK_resample,w_K = resample(xK_part,w_K,N)
    for i in range(0,N):
        res_list.append([w_K[i] * r for r in xK_resample[i]])
    x_K = np.array(xK_resample).sum(axis=0).tolist()
    x_K = [x / N for x in x_K]
    x_particle= np.random.multivariate_normal(f(x_K,uN,Q,Q)[0], Q, N)
    new_weights= [1/N] * N

    return (x_particle,new_weights)
(x_part,uN,P_x,N,y_K,S,k,T,Q,R,data)=(state,uN,state_cov,N,y_K,S_front,k_front,T_front,Q,R,particle_data)
'''

def cramer_pf_PF(x_part,uN,P_x,N,y_K,S,k,T,Q,R,data):
    xk_resample,temp_weight = resample(data[0],data[1],N)
    weight_sum=0
    res_list = []
    w_K=[]
    x_k = np.array(xk_resample).sum(axis=0).tolist()
    x_k = [x / N for x in x_k]
    var=(np.array(Q)/N).tolist()
    x_Kk= np.random.multivariate_normal(f(x_k,uN,var,Q)[0],f(x_k,uN,var,Q)[1], N) 
    w_k= [1/N] * N
    
    for i in range(0,N):
       state = x_Kk[i]
       price_diff = y_K - BS(state,S,k,T)
       w_K.append(w_k[i]*normal_dist(price_diff,0,R[0]))
       weight_sum= weight_sum + w_K[i]
    w_K = [x / weight_sum for x in w_K]
    xK_resample,w_K = resample(x_Kk,w_K,N)
    
    wKk = cramer_forward_particles(xK_resample,xk_resample,Q,N) 
    xKk=[[a.tolist(),b] for a, b in zip(xK_resample,xk_resample)]
    xKk_resample,wKk = resample(xKk,wKk,N)
    
    x_particle_front = x_Kk
    
    x_particle_back =[a[1] for a in xKk_resample]
    
    return (x_particle_front,x_particle_back)

(x_part,uN,P_x,N,y_K,S,k,T,Q,R)=(state,uN,state_cov,N,y_K,S_front,k_front,T_front,Q,R)
def cramer_pf(x_part,uN,P_x,N,y_K,S,k,T,Q,R):
    #print(x_part)
    #x_part=x_part[0]
    xk_resample = np.random.multivariate_normal(x_part, P_x, N)
    weight_sum=0
    res_list = []
    w_K=[]
    x_k = np.array(xk_resample).sum(axis=0).tolist()
    x_k = [x / N for x in x_k]
    var=(np.array(Q)/N).tolist()
    x_Kk= np.random.multivariate_normal(f(x_k,uN,var,Q)[0],f(x_k,uN,var,Q)[1], N) 
    w_k= [1/N] * N
    
    for i in range(0,N):
       state = x_Kk[i]
       price_diff = y_K - BS(state,S,k,T)
       #print(normal_dist(price_diff,0,1))
       #print(BS(state,S,k,T))
       #print(' ')
       w_K.append(w_k[i]*normal_dist(price_diff,0,R[0]))
       weight_sum= weight_sum + w_K[i]
    w_K = [x / weight_sum for x in w_K]
    xK_resample,w_K = resample(x_Kk,w_K,N)
    wKk = cramer_forward_particles(xK_resample,xk_resample,Q,N)
    xKk=[[a.tolist(),b.tolist()] for a, b in zip(xK_resample,xk_resample)]
    xKk_resample,wKk = resample(xKk,wKk,N)
    
    x_particle_front = x_Kk
    
    x_particle_back =[a[1] for a in xKk_resample]
    
    return (x_particle_front,x_particle_back)

state[0]
xKk[0][1].tolist()
(x , mean , sd)= (price_diff,0,1)
(np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
def cramer_forward_particles(x_front_particles,x_particles,Q,N):
    tau = []
    tau_sum = 0
    weight_sum =0 
    for i in range(0,N):
        temp_particle=x_front_particles[i]
        for k in range(0,N):
            diff = temp_particle - x_particles[k]
            res = multivariate_pdf(diff,[0,0],Q)
            tau_sum = tau_sum + res
        pdf = multivariate_pdf(temp_particle-x_particles[i],[0,0],Q)
        pdf_denom=pdf/(N*tau_sum)
        tau.append(pdf_denom)
        weight_sum = weight_sum + pdf_denom
    wKk = [x / weight_sum for x in tau]
    
    return wKk
    
D_t22=np.array([[     1,      0.],
       [     0., 260000.]])        
            
def Fisher_Matrix(J_t,D_t11,D_t12,D_t22):
    temp1= inv(np.add(J_t,D_t11))  
    temp2= np.dot(np.dot(D_t12.T,temp1),D_t12)
    return np.subtract(D_t22,temp2)

def PCRLB(J_t,D_t11,D_t12,D_t22):
    temp1= np.dot(inv(D_t22),D_t12.T)  
    temp2= np.dot(np.dot(D_t12,inv(D_t22)),D_t12.T)
    temp2 = inv(np.subtract(temp2,np.add(J_t,D_t11)))
    temp3= np.dot(D_t12,inv(D_t22))  
    res= np.dot(temp1,np.dot(temp2,temp3))
    return np.subtract(inv(D_t22),res)
    
w_K=w_particle
def resample(x_part,w_K,N):
    w_K.insert(0,0)
    w_K=np.cumsum(w_K)
    s = np.random.uniform(0,1,N)
    ind=[]
    new_particle =[] 
    for i in range(0,N):
        particle_no=s[i]
        for j in range(1,len(w_K)):
            if particle_no<=w_K[j]:
              ind.append(j-1)
              break
    #print(ind)
    for k in range(0,N):
        new_particle.append(x_part[ind[k]])
    new_weight=[1/N] * N
    return (new_particle,new_weight)
       
def GBM(S,price):
    S_0 = price
    sigma= S.std()
    mu = S.mean() + (sigma**2/2)
    return S_0*np.exp(t*(mu-(sigma**2/2))+sigma*np.random.normal())
    
          
def first_derivative_f(x_part_back):
    A=[]
    beta = 0.06
    A.append([beta,0])
    A.append([0,1])
    return np.array(A)
   
def first_derivative_BS(x_part_front,S,k,T):
     dk_plus = (np.log(S/k) + (x_part_front[1] + (x_part_front[0]/2))*T)/np.sqrt(x_part_front[0]*T)
     dk_minus = dk_plus - np.sqrt(x_part_front[0]*T)
     Vega= (0.5*S*np.sqrt(T)*scipy.stats.norm(0, 1).pdf(dk_plus))/(np.sqrt(x_part_front[0]))
     rho =  k*T*(math.exp(-x_part_front[1]*T))*scipy.stats.norm(0, 1).pdf(dk_minus) 
     return np.expand_dims(np.array([Vega,rho]),0)
  
    first_der.shape
(x_part_back,Q) = (x_particles_back[i],Q)
def Dt11(x_part_back,Q):
    first_der = first_derivative_f(x_part_back)
    res=np.dot(np.dot(first_der.T,inv(np.array(Q))),first_der)
    return res
         
def Dt12(x_part_back,Q):
    first_der = first_derivative_f(x_part_back)
    res=-np.dot(first_der.T,inv(np.array(Q)))
    return res

first_der.shape
(x_part_front,R,S,k,T)=(x_particles_front[i],R,S_front,k_front,T_front) 

(x_part_front,R,S,k,T)=(x_particles_front[i],R,S_front,k_front,T_front)
def Dt22(x_part_front,R,S,k,T):
    first_der = first_derivative_BS(x_part_front,S,k,T)
    res= np.dot(first_der.T,first_der)*(1/R[0])
    return res

x_particles_back = np.random.multivariate_normal(xbar_0, Q, N)
x_particles_front = np.random.multivariate_normal(xbar_0, Q, N)
M=1
S_front= 100
k_front = 100
T_front = 50

(a,b)=cramer_pf(x_particle[2],optionPrice.loc[1,'PriceChange'],P_0,w_particle,N,optionPrice.loc[1,'OptionPR'],optionPrice.loc[1,'StockPR'],optionPrice.loc[1,'StrikePrice'],optionPrice.loc[1,'TimeToMaturity'],Q)

(x_particles_back,x_particles_front,M,N,Q,R,S_front,k_front,T_front) = (b,a,1,N,P_0,R,optionPrice.loc[1,'StockPR'],optionPrice.loc[1,'StrikePrice'],optionPrice.loc[1,'TimeToMaturity'])
i = 5

J_t = inv(np.array(Q))

(J_t,state,state_cov,M,N,Q,R,y_K,S_front,k_front,T_front,uN,isParticle,particle_data)=(J_UKF,state_uns,Cov_uns,M,N,Q,R,optionPrice.loc[e+1,'OptionPR'],optionPrice.loc[e+1,'StockPR'],optionPrice.loc[e+1,'StrikePrice'],optionPrice.loc[e+1,'OptionPR'],optionPrice.loc[e+1,'PriceChange'],0,[])
(J_t,state,state_cov,M,N,Q,R,y_K,S_front,k_front,T_front,uN,isParticle,particle_data)=(J_PF,state_particle,Cov_part,M,N,Q,R,optionPrice.loc[e+1,'OptionPR'],optionPrice.loc[e+1,'StockPR'],optionPrice.loc[e+1,'StrikePrice'],optionPrice.loc[e+1,'OptionPR'],optionPrice.loc[e+1,'PriceChange'],1,[x_particle,w_particle])
def cramer_elements(J_t,state,state_cov,M,N,Q,R,y_K,S_front,k_front,T_front,uN,isParticle,particle_data):
    D_t11 = np.array([[0, 0], [0, 0]], dtype=np.float)
    D_t12 = np.array([[0, 0], [0, 0]], dtype=np.float)
    D_t22 = np.array([[0, 0], [0, 0]], dtype=np.float)
    if(isParticle==0):
        (x_particles_front,x_particles_back) = cramer_pf(state,uN,state_cov,N,y_K,S_front,k_front,T_front,Q,R)
    elif(isParticle==1):
        (x_particles_front,x_particles_back) = cramer_pf_PF(state,uN,state_cov,N,y_K,S_front,k_front,T_front,Q,R,particle_data)
    for j in range (0,M):
        for i in range(0,N):
            D_t11 = np.add(D_t11,Dt11(x_particles_back[i],Q))
            D_t12 = np.add(D_t12,Dt12(x_particles_back[i],Q))
            D_t22 = np.add(D_t22,Dt22(x_particles_front[i],R,S_front,k_front,T_front))
    D_t11 = D_t11/(M*N)
    D_t12 = D_t12/(M*N)
    D_t22 = D_t22/(M*N)
    D_t22 = np.add(inv(np.array(Q)),D_t22)
    J_T=Fisher_Matrix(J_t,D_t11,D_t12,D_t22)
    J_T_inv=PCRLB(J_t,D_t11,D_t12,D_t22)
    return (J_T,J_T_inv)
   (J_inv,P_x,state,P_w,P_v,N,data)=([J_EKF_inv,J_UKF_inv, J_PF_inv],[P_temp,Cov_uns,Cov_part],[x_temp[0].tolist(),state_uns,state_particle],P_w,P_v,N,[x_particle,w_particle])
i=2
def best_estimator(J_inv,P_x,state,P_w,P_v,N,data):
    optimal_bayesian = 0
    max_sum = -math.inf
    for i in range(0,len(J_inv)):
        temp1 = J_inv[i].diagonal().tolist()
        temp2 = np.array(P_x[i]).diagonal().tolist()
        phi= [a/b for a,b in zip(temp1,temp2)]
        phi_sum = sum(phi)
        print(temp2)
        if(phi_sum>max_sum):
            optimal_bayesian = i
            max_sum = phi_sum
    #print(state[optimal_bayesian])
    (state_unscented,cov_unscented)= UKF_matrix(state[optimal_bayesian],P_x[optimal_bayesian],P_w,P_v)
    if(optimal_bayesian!=2):
        state_particle = np.random.multivariate_normal(state[optimal_bayesian],P_x[optimal_bayesian], N)
        weight_particle = [1/N] * N
    elif(optimal_bayesian==2):
        state_particle = data[0]
        weight_particle = data[1]
    return (state[optimal_bayesian],P_x[optimal_bayesian],state_unscented,cov_unscented,x_particle,w_particle,optimal_bayesian)
    
  state[0]       
def dataset():
    optionPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSE.csv')
    market_price=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSEI.csv')
    market_price['Date'] = pd.to_datetime(market_price['Date'])
    optionPrice['Date'] = pd.to_datetime(optionPrice['Date'])
    optionPrice['Expiry'] = pd.to_datetime(optionPrice['Expiry'])
    optionPrice = pd.merge(optionPrice,market_price[['Date', 'Close']],on='Date')
    optionPrice['TimeToMaturity']=optionPrice['Expiry'] - optionPrice['Date']
    optionPrice['TimeToMaturity']=optionPrice['TimeToMaturity'].dt.days
    #optionPrice['TimeToMaturity'] = optionPrice['TimeToMaturity'].dt.days.astype('int16')
    optionPrice=optionPrice.loc[optionPrice['StrikePrice'] == 10500]
    optionPrice=optionPrice.reset_index()
    del optionPrice['index']
    optionPrice=optionPrice.sort_values(by=['Date'])
    optionPrice['StockPR']=optionPrice['Close']
    optionPrice=optionPrice.bfill(axis ='rows')
    optionPrice['PriceChange'] = optionPrice['StockPR'].pct_change()
    optionPrice=optionPrice.loc[170:]
    optionPrice=optionPrice.reset_index()
    del optionPrice['index']
    optionPrice.loc['Set']='Train'
    xbar_0 = [0.012,0.008]
    xbar_0 = [0.01,0.01]
    AAF_prices=[]
    AAF_prices_test=[]
    real_prices=[]
    real_prices_test=[]
    vol_AAF_train =[]
    risk_AAF_train = []
    vol_AAF_test =[]
    risk_AAF_test = []
    stock_price=[optionPrice.loc[1,'StockPR']]
    filter_used_AAF_vol_train=[]
    filter_used_AAF_risk_train=[]
    filter_used_AAF=[]
    filter_used_AAF_vol_test=[]
    filter_used_AAF_risk_test=[]
    
    P_0 = []
    P_0.append([0.00001,0])
    P_0.append([0,0.0001])
    Q_Part = []
    Q_Part.append([0.0001,0])
    Q_Part.append([0,0.001])
    x_temp= xbar_0
    P_temp = P_0
    Q = []
    Q.append([0.001,0])
    Q.append([0,0.001])
    R= [2500]
    stock_price=[optionPrice.loc[1,'StockPR']]
    L=5
    (xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
    P_w=Q
    P_v=R
    x_temp=np.array([x_temp])
    N=150
    N1=200
    M=1
    x_particle= np.random.multivariate_normal(xbar_0, P_0, N1)
    w_particle= [1/N1] * N1
    J=inv(np.array(P_0))
    (J_0,J_0_inv)=cramer_elements(J,x_temp[0],P_temp,1,N,Q,R,optionPrice.loc[1,'OptionPR'],optionPrice.loc[1,'StockPR'],optionPrice.loc[1,'StrikePrice'],optionPrice.loc[1,'TimeToMaturity'],optionPrice.loc[1,'PriceChange'],0,x_particle)
    (J_EKF,J_UKF, J_PF)= (J_0,J_0,J_0)
    (J_EKF_inv,J_UKF_inv, J_PF_inv)= (J_0_inv,J_0_inv,J_0_inv)
    e=1
    for e in range(1,optionPrice.shape[0]-1):
        
        if(optionPrice.loc[e,'Set']=='Test'):
            price_bfr=optionPrice.loc[e-1,'StockPR']
            price= GBM(np.array(stock_price[-60:]),price_bfr)
            price_chg= (price-price_bfr)/price_bfr
            if(filter_used_AAF[-1]==0):
                PredResult=extended_KF_predict(x_temp,price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],price_chg,P_temp, Q,R)        
            elif(filter_used_AAF[-1]==1): 
                PredResult = Unscented_KF_predict(xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],price_chg,price,optionPrice.loc[e,'StrikePrice'])
            elif(filter_used_AAF[-1]==2): 
                PredResult = Particle_Filter_Pred(x_particle,price_chg,Q,w_particle,N,price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],R)   
                
            vol_AAF_test.append(PredResult[0][0])
            risk_AAF_test.append(PredResult[0][1])
            AAF_prices_test.append(BS(PredResult[0],price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
            real_price_test.append(optionPrice.loc[e,'OptionPR'])
            
        PredResult = extended_KF_predict(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],P_temp, Q,R)
        UpdateResult_EKF = extended_KF_update(optionPrice.loc[e,'OptionPR'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        (x_temp,P_temp)= UpdateResult_EKF
        (xuns_temp,Pa_O,state_uns,Cov_uns,P_w,P_v)=Unscented_KF_update(optionPrice.loc[e,'OptionPR'],xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],P_w,P_v)
        (state_particle,x_particle,Cov_part,w_particle)=Particle_Filter(x_particle,optionPrice.loc[e,'PriceChange'],Q_Part,w_particle,N1,optionPrice.loc[e,'OptionPR'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e,R[0])
        
        PCRLB_EKF = cramer_elements(J_EKF,x_temp[0],P_temp,M,N,Q,R,optionPrice.loc[e+1,'OptionPR'],optionPrice.loc[e+1,'StockPR'],optionPrice.loc[e+1,'StrikePrice'],optionPrice.loc[e+1,'TimeToMaturity'],optionPrice.loc[e+1,'PriceChange'],0,[])
        PCRLB_UKF = cramer_elements(J_UKF,state_uns,Cov_uns,M,N,Q,R,optionPrice.loc[e+1,'OptionPR'],optionPrice.loc[e+1,'StockPR'],optionPrice.loc[e+1,'StrikePrice'],optionPrice.loc[e+1,'TimeToMaturity'],optionPrice.loc[e+1,'PriceChange'],0,[])
        PCRLB_PF = cramer_elements(J_PF,state_particle,Cov_part,M,N,Q,R,optionPrice.loc[e+1,'OptionPR'],optionPrice.loc[e+1,'StockPR'],optionPrice.loc[e+1,'StrikePrice'],optionPrice.loc[e+1,'TimeToMaturity'],optionPrice.loc[e+1,'PriceChange'],0,[x_particle,w_particle])
        #print(x_temp[0])
        (x_temp,P_temp,xuns_temp,Pa_O,x_particle,w_particle,index) = best_estimator([J_EKF_inv,J_UKF_inv, J_PF_inv],[P_temp,Cov_uns,P_temp],[x_temp[0].tolist(),state_uns,state_particle],P_w,P_v,N,[x_particle,w_particle])
        
        (J_EKF,J_UKF, J_PF)= (PCRLB_EKF[0],PCRLB_UKF[0],PCRLB_PF[0])
        (J_EKF_inv,J_UKF_inv, J_PF_inv)= (PCRLB_EKF[1],PCRLB_UKF[1],PCRLB_PF[1])
        print('Real Price',optionPrice.loc[e,'OptionPR'])
        print('Predicted Price',BS(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
        print('filter',index)
        state_temp=x_temp1
        if(optionPrice.loc[e,'Set']=='Train'):
            vol_AAF_train.append(x_temp[0])
            risk_AAF_train.append(x_temp[1])
            AAF_prices.append(BS(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
            filter_used_AAF.append(index)
            #filter_used_AAF_vol_train.append(index)
            #filter_used_AAF_risk_train.append(index)
            real_price.append(optionPrice.loc[e,'OptionPR'])
            stock_price.append(np.log(optionPrice.loc[e,'StockPR']/optionPrice.loc[e-1,'StockPR']))
            x_temp=np.array([x_temp])
        if(optionPrice.loc[e,'Set']=='Test'):
            #vol_AAF_test.append(x_temp[0])
            #risk_AAF_test.append(x_temp[1])
            #ABF_prices.append(BS(x_temp1,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
            #filter_used_AAF_vol_test.append(index)
            #filter_used_AAF_risk_test.append(index)
            #real_price.append(optionPrice.loc[e,'OptionPR'])
            stock_price.append(np.log(optionPrice.loc[e,'StockPR']/optionPrice.loc[e-1,'StockPR']))
            x_temp=np.array([x_temp])
            filter_used_AAF.append(index)
        
        
if(optionPrice.loc[e,'Set']=='Test'):
            price_bfr=optionPrice.loc[e-1,'StockPR']
            price= GBM(np.array(stock_price[-60:]),price_bfr)
            price_chg= (price-price_bfr)/price_bfr
            if(filter_used[-1]==0):
                PredResult=extended_KF_predict(final_state[-1],price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],price_chg,final_state_var[-1], Q,R)
            elif(filter_used[-1]==1): 
                PredResult = Unscented_KF_predict(xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],price_chg,price,optionPrice.loc[e,'StrikePrice'])
            elif(filter_used[-1]==2): 
                PredResult = Particle_Filter(x_particle,price_chg,Q,w_particle,N,price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e)   
                
            test_state.append(PredResult[0])
            option_price_test.append(BS(PredResult[0],price,optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))        


def test_dataset():
    optionPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/OptionTestingData.csv')
    optionPrice['Date'] = pd.to_datetime(optionPrice['Date'])
    optionPrice['TimeToMaturity']= datetime.datetime(2020, 1, 5) - optionPrice['Date']
    optionPrice['TimeToMaturity'] = optionPrice['TimeToMaturity'].dt.days.astype('int16')
    optionPrice['StrikePrice']=90
    optionPrice['PriceChange'] = optionPrice['StockPR'].pct_change()
    xbar_0 = final_state[-1]
    initial_filter=filter_used[-1]
    P_0 = []
    final_state =[]
    filter_used=[]
    P_0.append([1,0])
    P_0.append([0,1])
    x_temp= xbar_0
    P_temp = P_0
    Q = []
    Q.append([1,0])
    Q.append([0,1])
    R= [1]
    L=5
    (xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
    P_w=Q
    P_v=R
    x_temp=np.array([x_temp])
    N=100
    x_particle= np.random.multivariate_normal(xbar_0, P_0, N)
    w_particle= [1/N] * N
    (J_0,J_0_inv)=cramer_elements(x_temp,P_temp,M,N,Q,R,optionPrice.loc[1,'OptionPR'],optionPrice.loc[1,'StockPR'],optionPrice.loc[1,'StrikePrice'],optionPrice.loc[1,'OptionPR'],optionPrice.loc[1,'PriceChange'])
    (J_EKF,J_UKF, J_PF)= (J_0,J_0,J_0)
    (J_EKF_inv,J_UKF_inv, J_PF_inv)= (J_0_inv,J_0_inv,J_0_inv)
 
    for e in range(1,optionPrice.shape[0]-1):
        PredResult = extended_KF_predict(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],P_temp, Q,R)
        XK_k, P_k, y1_k, P_yy, P_xy = Unscented_KF_predict(xa_uns,P_a,L,T,uK,S,ks)
        XK_k, P_k, y1_k, P_yy, P_xy = ParticleTest(xa_uns,P_a,L,T,uK,S,ks)
        
        

def StateEstimation(P_X, P_Y, M_0, P_0, N, y, u1, u2, theta, T):
    
    x1_ = [0] * T
    x2_ = [0] * T
    x=np.zeros((N,2))
    #Initialization
    InitialParticle = np.random.multivariate_normal(M_0, P_0, N)  #Sampling fromInitial State Density
    x[:,0] = InitialParticle[:,0]  #Particles for Initial State 1
    x[:,1] = InitialParticle[:,1]  #Particles for Initial State 2
    x1_[0] = M_0[0]
    x2_[0] = M_0[1]
    x1= np.zeros((N))
    x2= np.zeros((N))
    #Start Time Loop
    for t in range(1,T):
        #Prediction
        Snoise = np.random.multivariate_normal([0,0], P_X, N)
        for k in range(0,N):
            x1[k] = f1(x[k,0], x[k,1], u1[t], theta) + Snoise[k,0]   #Prediction for State 1
            x2[k] = f2(x[k,0], x[k,1], u1[t], u2[t], theta) + Snoise[k,1]  #Prediction for State 2
        
        #Update
        PredError =  np.asarray([y[t]]* N) - g(x1)  # Prediction error
        w= pe(PredError, P_Y) # Weight calculation
        w=w/np.sum(w)    # Normalization
        
        #State Estimate
        x1_[t]= np.dot(w , x1)   #Estimation for State 1
        x2_[t]= np.dot(w , x2)   #Estimation for State 2
        
        #Resampling
        x= np.array([x1,x2]).transpose() 
        ind= resampling(w)   #systematic resampling
        x=x[ind,:]
        
    return  x1_,x2_ 
k=2
t=1
np.arange(M=100)
def resampling(q):
    qc = q.cumsum() 
    M = len(q)
    u = np.arange(M) 
    u = u + np.random.uniform(0,1)
    u = u/M
    i= [0]*M
    k=0
    for j in range(0,M):
        while(qc[k]<u[j]):
            k=k+1
        i[j]=k
    return i


N=10
T=300
M_0 = [5,0]  
P_0 = np.diag([0.1, 0.05]) 
P_X = np.diag([0.001, 0.001])
P_Y = 1
x1 = [0] * T
x2 = [0] * T
x1[0]=5
x2[0]=0


theta=[0.31,0.18, 0.55, 0.03]
u1=list(np.random.uniform(0.15,0.25,T))
u2= list(np.random.uniform(5,35,T))

#State Transition and Measurement Eqution
def f1(x1, x2, u1, theta):
    return x1+(0.1*(((theta[0]*x2)/(theta[1]+x2))-u1-theta[3])*x1)
  
def f2(x1, x2, u1, u2, theta):
    return x2+(0.1*((((-theta[0]*x2)/(theta[1]+x2))*(x1/theta[2]))+ u1*(u2-x2)))

def g(x1):
    return x1

#weight function
def pe(PredError, P_Y):
    w=[]
    for i in range(0,len(PredError)):
        w.append(math.exp(-(math.sqrt(abs(PredError[i]))/(2*P_Y)))/math.sqrt(2*3.14*P_Y))
    return np.array(w)

y=np.zeros(T)
for t in range(1,T):
    Snoise = np.random.multivariate_normal([0,0], P_X, 1)
    x1[t]= f1(x1[t-1], x2[t-1], u1[t-1],theta)+ Snoise[:,0]
    x2[t]=f2(x1[t-1],x2[t-1],u1[t-1],u2[t-1],theta)+ Snoise[:,1]
    MNoise=np.random.normal(0, math.sqrt(P_Y), 1)
    y[t]=g(x1[t])+ MNoise
    
xTrue= np.array([x1,x2]).transpose()

x1_,x2_ = StateEstimation(P_X, P_Y, M_0, P_0, N, y, u1, u2, theta, T)

rmse=[]
for i in [1,10,100,500,1000,5000,10000,100000]:
    N=i
    x1_,x2_ = StateEstimation(P_X, P_Y, M_0, P_0, N, y, u1, u2, theta, T)
    rms = sqrt(mean_squared_error(x1_, xTrue[:,0]))
    rmse.append(rms)
    
plt.plot(rmse[1:])
plt.title('RMSE PF vs True State')
plt.xlabel('N_SIMULATONS_POINTS:[10,100,500,1000,5000,10000,100000]')
plt.ylabel('RMSE')


plt.plot(y)
plt.plot(x1_)
plt.title('Pf performance')
plt.legend(['True Value','PF Estimate'])
plt.xlabel('timestep')
plt.ylabel('Concentration'); 

plt.plot(x1_)
plt.plot(xTrue[:,0])
plt.title('Pf performance')
plt.legend(['PF Estimate', 'True State'])
plt.xlabel('timestep')
plt.ylabel('Concentration'); 

plt.plot(x2_)
plt.plot(xTrue[:,1])
plt.title('Pf performance on State 2')
plt.legend(['PF Estimate', 'True State'])
plt.xlabel('timestep')
plt.ylabel('Concentration'); 


y=data['TSLA'].pct_change().dropna()

def f1(x1):
    return x1
  

def g(x1):
    return x1

def pe(PredError, P_Y):
    w=[]
    for i in range(0,len(PredError)):
        w.append(math.exp(-(math.sqrt(abs(PredError[i]))/(2*P_Y)))/math.sqrt(2*3.14*P_Y))
    return np.array(w)
rmse=[]
for i in [1,10,100,500,1000,5000,10000,100000]:
    N=i
    x1_ = StateEstimation(P_X, P_Y, M_0, P_0, N, y, T)
    rms = sqrt(mean_squared_error(x1_[1:], state_means[1:]))
    rmse.append(rms)
    
plt.plot([10,100,500,1000,5000,10000,100000],rmse[1:])
plt.title('RMSE PF vs Kalman')
plt.xlabel('N_SIMULATONS_POINTS')
plt.ylabel('RMSE');

plt.plot(x1_)
plt.plot(state_means[1:])
plt.title('Pf performance')
plt.legend(['PF Estimate', 'Kalman State'])
plt.xlabel('timestep')
plt.ylabel('Returns');

plt.plot(y)
plt.plot(state_means)
plt.title('Kalman filter estimate of average')
plt.legend(['returns','Kalman Estimate'])
plt.xlabel('timestep')
plt.ylabel('returns');

plt.plot(y)
plt.plot(x1_)
plt.title('Paricle filter estimate')
plt.legend(['returns','PF Estimate'])
plt.xlabel('timestep')
plt.ylabel('returns');          


rms = sqrt(mean_squared_error(x1_, state_means))