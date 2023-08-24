# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 00:53:00 2021

@author: KUMAR YASHASWI
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:41:55 2021

@author: KUMAR YASHASWI
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
     Vega= (-0.5*S*np.sqrt(T)*scipy.stats.norm(0, 1).pdf(-dk_plus))/(np.sqrt(X[0]))
     #print(S)
     #print(math.exp(-XK_k[1]*T))
     rho =  -k*T*(math.exp(-X[1]*T))*scipy.stats.norm(0, 1).pdf(-dk_minus)
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
     c = -S*norm.cdf(-dk_plus) + k*rhs*norm.cdf(-dk_minus)
     return c



def Unscented_KF_Transform(xa_uns,P_a,L):
    W_m=[]
    W_c=[]
    X_uns=[]
    #L=5
    alpha=0.00001
    kappa= 3-L
    lamda = (alpha*alpha)*(L + kappa) - L
    beta=2
    #print(xa_uns)
    #print(P_a)
    #print('check')
    #print(' ')
    #array_2d = np.array([[1, 4], [9, 16]], dtype=np.float)
    #array_2d = np.array([[1, 4]], dtype=np.float)
    #array_2d=array_2d.tolist()[0]
    #X_uns.append(array_2d)
    X_uns.append(xa_uns.tolist())
    W_m.append(lamda/(lamda+L))
    W_c.append((lamda/(lamda+L)) + 1 - (alpha*alpha) + beta)
    #print(L + lamda)
    #print((L + lamda)*P_a)
    #np.sqrt((L + lamda)*P_a[:,3-1])
    #np.add(xa_uns,np.sqrt((L + lamda)*P_a[:,1]))
    for i in range(1,L+1):
       #print(np.sqrt((L + lamda)*P_a[:,i-1]))
       X_uns.append(np.add(xa_uns,np.linalg.cholesky((L + lamda)*P_a)[:,i-1]).tolist())
       W_m.append(1/(2*(lamda+L)))
       W_c.append(1/(2*(lamda+L)))
    for j in range(1,L+1):
       X_uns.append(np.subtract(xa_uns,np.linalg.cholesky((L + lamda)*P_a)[:,j-1]).tolist())
       W_m.append(1/(2*(lamda+L)))
       W_c.append(1/(2*(lamda+L)))
       
    X_uns_matrix = np.array(X_uns).T 
    #print(X_uns_matrix)
    X_x = X_uns_matrix[0:2,:]
    X_w = X_uns_matrix[2:4,:]
    X_v = X_uns_matrix[4:5,:]
    #print(X_uns) 
    #print(X_uns_matrix)
    #print(X_x)
    #print(X_w)
    #print(X_v) 
    #print(' ')
    return (X_uns,X_uns_matrix,X_x,X_w,X_v,W_m,W_c)
    
def Unscented_KF_predict(xa_uns,P_a,L,T,uK,S,ks):
     # Sigma Points of x_k
     X_uns,X_uns_matrix,X_x,X_w,X_v,W_m,W_c = Unscented_KF_Transform(xa_uns,P_a,L)
     ChiK_k=[]
     y1_k=[]
     weighted_state=np.array([0, 0], dtype=np.float)
     weighted_measure=np.array([0], dtype=np.float)
     cov = np.array([[0, 0], [0, 0]], dtype=np.float)
     cov_y = np.array([0], dtype=np.float)
     cov_yx=np.expand_dims(np.array([0, 0], dtype=np.float),axis=1)
     # Mean of x
     #i=3
     X_x=np.array(X_x)
     X_v=np.array(X_v)
     X_w=np.array(X_w)
     for i in range(0,(2*L)+1):
        #print("state1",X_x[:,i].tolist())
        a= X_x[:,i].tolist()
        b=X_w[:,i].tolist()
        #print("state2",X_w[:,i].tolist())
        mean=np.array(state(a,b,uK))
        ChiK_k.append(mean)
        #print(weighted_state.shape)
        #print(mean.shape)
        weighted_state= np.add(weighted_state,(W_m[i]*mean))
        #print(' ')
        #print(W_m[i]*mean)
        #print(' ')
        
     # Covariance of y
     for i in range(0,(2*L)+1):
         #print(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).shape)
         #print(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).T.shape)
         P_xx= np.dot(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1),np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).T)
         #print(P_xx)
         #print(cov.shape)
         cov= np.add(cov,(W_c[i]*P_xx))
         #print(' ')
     # Mean of y
     for i in range(0,(2*L)+1):
         #np.array([1])
         #print('State ',ChiK_k[i].tolist())
         y_mean = np.add(np.array(BS(ChiK_k[i].tolist(),S,ks,T)),X_v[:,i])
         #if(X_v[:,i]<0):
             #print('state',X_v[:,i])
         y1_k.append(y_mean)
         #print(weighted_measure.shape)
         weighted_measure = np.add(weighted_measure,(W_m[i]*(y_mean)))
         #print((y_mean).shape)
     #print(' ')
     # Covariance of y
     for i in range(0,(2*L)+1):
         P_yy = np.dot(np.expand_dims(np.subtract(y1_k[i], weighted_measure),axis=1),(np.expand_dims(np.subtract(y1_k[i], weighted_measure),axis=1)).T)[0]
         #print(P_yy)
         P_xy = np.dot(np.expand_dims(np.subtract(ChiK_k[i], weighted_state), axis=1),np.expand_dims((np.subtract(y1_k[i], weighted_measure)),axis=1).T)
         #print(P_xy.shape)
         #print(cov_yx.shape)
         #print(' ')
         cov_y= np.add(cov_y ,(W_c[i]*P_yy))
         cov_yx= np.add(cov_yx ,(W_c[i]*P_xy))
     #print('y',weighted_measure[0])
#np.expand_dims(np.subtract(ChiK_k[i], weighted_state), axis=1).shape   
     return (weighted_state,cov,weighted_measure[0],cov_y,cov_yx) 

def state(X,v,uN):
     omega= 0
     alpha = 0
     beta = 1
     XN=[]
     XN.append(omega + alpha*(uN * uN) + beta*X[0] + v[0])
     XN.append(X[1] + v[1])
     return XN


def Unscented_KF_update(y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v):
     XK_k, P_k, y1_k, P_yy, P_xy = Unscented_KF_predict(xa_uns,P_a,L,T,uK,S,ks)
     b_K = y_K - y1_k
     #print('Real Price',y_k)
     #print('Pred Price',y1_k)
     #(1/P_yy)
     K_K = P_xy*(1/P_yy[0])
     #print(P_xy)
     #print(y1_k)
     #print(K_K.shape)
     #print(XK_k.shape)
     #K_K=np.expand_dims(K_K,axis=1)
     XK_K = np.add(XK_k,(K_K*b_K).T)
     #print(XK_K)
     #print(dot(K_K , np.expand_dims(dot(P_yy,K_K.T),axis=1).T).shape)
     P_final = np.subtract(P_k,dot(K_K , np.expand_dims(dot(P_yy,K_K.T),axis=1).T))
     #print(P_final)
     #print(P_final)
     #print(' ')
     UnX,UnP = UKF_matrix(XK_K.tolist()[0],P_final, P_w,P_v)
     #print(UnX)
     #print(UnP)
     #print(' ')
     #UnX = [XK_K,0,0,0]
     #UnP = [P_final, Pw,Pv]
     return (UnX,UnP,XK_K.tolist()[0],P_final,P_w,P_v)




    



def UKF_matrix(x,P_O, P_w,P_v):
    x_t=x.copy()
    x_t.extend([0,0,0])
    Pa_O=np.pad(P_O, ((0,3),(0,3)), mode='constant', constant_values=0)
    Pa_O[2:4,2:4]=  P_w
    Pa_O[4:,4:]=  P_v
    return (np.array(x_t),Pa_O)


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
    N=len(w_K)
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
     Vega= (-0.5*S*np.sqrt(T)*scipy.stats.norm(0, 1).pdf(-dk_plus))/(np.sqrt(X[0]))
     #print(S)
     #print(math.exp(-XK_k[1]*T))
     rho =  -k*T*(math.exp(-X[1]*T))*scipy.stats.norm(0, 1).pdf(-dk_minus)
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
    t=1
    return S_0*np.exp(t*(mu-(sigma**2/2))+(sigma*np.random.normal()))
    
          
def first_derivative_f(x_part_back):
    A=[]
    beta = 1
    A.append([beta,0])
    A.append([0,1])
    return np.array(A)
   
def first_derivative_BS(x_part_front,S,k,T):
     x_part_front= [r/100 for r in x_part_front]
     x_part_front[0]=np.max([x_part_front[0],0.0000001])
     dk_plus = (np.log(S/k) + (x_part_front[1] + (x_part_front[0]/2))*T)/np.sqrt(x_part_front[0]*T)
     dk_minus = dk_plus - np.sqrt(x_part_front[0]*T)
     Vega= (-0.5*S*np.sqrt(T)*scipy.stats.norm(0, 1).pdf(-dk_plus))/(np.sqrt(x_part_front[0]))
     rho =  -k*T*(math.exp(-x_part_front[1]*T))*scipy.stats.norm(0, 1).pdf(-dk_minus) 
     return np.expand_dims(np.array([Vega,rho]),0)
 
def Dt11(x_part_back,Q):
    first_der = first_derivative_f(x_part_back)
    res=np.dot(np.dot(first_der.T,inv(np.array(Q))),first_der)
    return res
         
def Dt12(x_part_back,Q):
    first_der = first_derivative_f(x_part_back)
    res=-np.dot(first_der.T,inv(np.array(Q)))
    return res

def Dt22(x_part_front,R,S,k,T):
    first_der = first_derivative_BS(x_part_front,S,k,T)
    res= np.dot(first_der.T,first_der)*(1/R[0])
    return res

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

def best_estimator(J_inv,P_x,state,P_w,P_v,N,data):
    optimal_bayesian = 0
    max_sum = -math.inf
    for i in range(0,len(J_inv)):
        temp1 = J_inv[i].diagonal().tolist()
        temp2 = np.array(P_x[i]).diagonal().tolist()
        phi= [a/b for a,b in zip(temp1,temp2)]
        phi_sum = sum(phi)
        #print(temp2)
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
    