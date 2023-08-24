# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:10:04 2021

@author: KUMAR YASHASWI
"""

EKF_prices=[]
real_prices=[]
xbar_0 = [0.012,0.008]
P_0 = []
P_0.append([0.00000001,0])
P_0.append([0,0.00000001])
x_temp= xbar_0
P_temp = P_0
Q = []
Q.append([0.00000001,0])
Q.append([0,0.00000001])
R= [400]
L=5
(xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
P_w=Q
P_v=R
x_temp=np.array([x_temp])
N=500
x_particle= np.random.multivariate_normal(xbar_0, P_0, N)
w_particle= [1/N] * N

e=1
for e in range(1,optionPrice.shape[0]):
        R=std(x_particle,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'])
        #print(x_temp)- y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v
        print(R)
        (state,x_particle,particle_cov,w_particle)=Particle_Filter(x_particle,optionPrice.loc[e,'PriceChange'],Q,w_particle,N,optionPrice.loc[e,'OptionPR'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e,R)

(state,x_particle,particle_cov,w_particle)=(x_K,xK_resample,state_cov,w_K)  
e=e+1     
(x_part,uN,Q,w_k,N,y_k,S,k,T,e,R)=(x_particle,optionPrice.loc[e,'PriceChange'],Q,w_particle,N,optionPrice.loc[e,'OptionPR'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e,R)
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
       print(i," BS ",price_diff)
       w_K.append(w_k[i]*normal_dist(price_diff,0,1600)*100)
       stateMean.append([w_K[i] * r for r in xK_part[i]])
       weight_sum= weight_sum + w_K[i]
    #print(i," BS ",weight_sum)
    state_mean = np.array(stateMean).sum(axis=0).tolist()
    
    w_K = [x / weight_sum for x in w_K]
    state_mean = [x / weight_sum for x in state_mean]
    sum(w_K)/500
    max(w_K)
    min(w_K)
    w_K.sort()
    np.array(w_K).cumsum()
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
    xK_resample,w_K = resample(xK_part,w_K,N,Neff,n_e)
    for i in range(0,N):
        res_list.append([w_K[i] * r for r in xK_resample[i]])
    x_K = np.array(res_list).sum(axis=0).tolist()
    
            
    #print("State  ",x_K)
    #print(" PArticle ",xK_resample)
    #print(" Matrix ",state_cov)
    #print("Weight  ",w_K[5])
    return (x_K,xK_resample,state_cov,w_K)


optionPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSE.csv')
market_price=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSEI.csv')
market_price['Date'] = pd.to_datetime(market_price['Date'])
optionPrice['Date'] = pd.to_datetime(optionPrice['Date'])
optionPrice['Expiry'] = pd.to_datetime(optionPrice['Expiry'])
optionPrice = pd.merge(optionPrice,market_price[['Date', 'Close']],on='Date')
optionPrice['TimeToMaturity']=optionPrice['Expiry'] - optionPrice['Date']
optionPrice['TimeToMaturity']=optionPrice['TimeToMaturity'].dt.days
#optionPrice['TimeToMaturity'] = optionPrice['TimeToMaturity'].dt.days.astype('int16')
optionPrice=optionPrice.loc[optionPrice['StrikePrice'] == 7000]
optionPrice=optionPrice.reset_index()
del optionPrice['index']
optionPrice=optionPrice.sort_values(by=['Date'])
optionPrice['StockPR']=optionPrice['Close']
optionPrice=optionPrice.bfill(axis ='rows')
optionPrice['PriceChange'] = optionPrice['StockPR'].pct_change()

plt.figure(figsize = (12, 4))
plt.plot(np.array(optionPrice['OptionPR']),label="OHLC-Rl_Portfolio")
#plt.plot(np.array(mark_rewards).cumsum(),label="Markowitz Portfolio Return")

#plt.plot(np.array(optionPrice['StockPR']),label="Random Portfolio Return")
plt.legend(loc="upper left")
plt.xlabel('time period in days')
plt.ylabel('Test set returns')
plt.show()