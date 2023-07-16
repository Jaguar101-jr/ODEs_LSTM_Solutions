import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import time
import joblib
import keras
from matplotlib.pyplot import figure
import seaborn as sns
#from docx import Document
#from docx.shared import Inches
sns.set()
#from docx.shared import Mm

def groundtruth_generate_obs(t_0, t_n, N=1000, nsim=1000, A=1, beta1=10, beta2=4.5, alpha=6, mu=1, theta=0.5,
                             a1=1.2, a2=0.3, a3=2.4, b1=1.4, b2=0.2, b3=0.1,
                             gamma1=1, gamma2=1.5,
                             S_0=0.3, V_0=0.4, I_0=0.3,
                             tau1=3, tau2=5,
                             sagma1=0.03, sagma2=0.03, sagma3=0.05):
    S = []
    V = []
    I = []
    S.append(S_0)
    V.append(V_0)
    I.append(I_0)

    N = 1000
    h1 = (t_n - t_0) / N
    t = np.arange(0, t_n, h1)
    dw1 = np.sqrt(h1) * np.random.randn(nsim)
    dw2 = np.sqrt(h1) * np.random.randn(nsim)
    dw3 = np.sqrt(h1) * np.random.randn(nsim)

    M2 = int(tau2 / h1)
    M1 = int(tau1 / h1)
    ss1 = S[0];
    vv1 = V[0];
    ii1 = I[0]
    c1 = (gamma1 * (1 - math.e ** (-mu * tau1)) + mu) * (alpha + mu)
    c2 = (gamma2 * (1 - math.e ** (-mu * tau2)) + mu) * (alpha + mu)
    k1 = c1 / (alpha + mu)
    k2 = beta2 * alpha * A / (b1 * A * alpha + c2)
    Diff = k1 - k2
    T = (mu / (mu + alpha)) + (alpha * mu / c2)

    # the first way
    R = ((beta1 * A) / ((a1 * A + alpha + mu) * (gamma1 + mu + (sagma3 ** 2 / 2)))) + (
                (beta2 * alpha * A) / ((b1 * alpha * A + c2) * (gamma1 + mu + (sagma3 ** 2 / 2))))
    Rd = ((beta1 * A) / ((a1 * A + alpha + mu) * (gamma1 + mu))) + (
                (beta2 * alpha * A) / ((b1 * alpha * A + c2) * (gamma1 + mu)))
    L1 = beta1 * A / (a1 * A + alpha + mu) * (a2 + a3 * A / (alpha + mu)) + beta2 * alpha * A / (
                b1 * alpha * A + c2) * (b2 + b3 * alpha * A / c2)
    L2 = (beta2 * alpha * (c1 * (b1 * alpha * A + c2) + beta2 * A * mu * (mu + alpha))) / (
                (b1 * alpha * A + c2) ** 2 * (mu + alpha))
    # L3=beta1/((1+a1)*mu+alpha)*(beta1*mu/((1+a1)*mu+alpha)-gamma1*exp(-mu*tau1));
    infI = ((gamma1 + mu + sagma3 ** 2 / 2) * (R - 1)) / (L1 + L2)
    DD = ((beta1 * mu) / ((1 + a1) * mu + alpha)) - gamma1 * math.e ** (-mu * tau1)
    if R > 1:
        maxS = (A / (alpha + mu)) - (1 / (alpha + mu)) * (
                    (c1 / (alpha + mu)) - beta2 * alpha * A / (b1 * alpha * A + c2)) * infI
        maxV = (A * (mu + alpha)) / c2 - (mu * (mu + alpha)) / c2 * maxS - c1 / c2 * infI
    else:
        maxS = A / (alpha + mu)
        maxV = (A * (mu + alpha)) / c2 - (mu * (mu + alpha)) / c2 * maxS

    TT = infI + maxS + maxV

    # ss1, vv1, ii1
    ss1 = S_0
    vv1 = V_0
    ii1 = I_0

    Sn, Vn, In = [ss1], [vv1], [ii1]
    h2 = h1
    p = (np.exp(h2) - 1)
    dwn1 = np.sqrt(h2) * np.random.randn(nsim)
    dwn2 = np.sqrt(h2) * np.random.randn(nsim)
    dwn3 = np.sqrt(h2) * np.random.randn(nsim)

    for j in range(N - 1):
        if M2 >= j:
            Vn1, Vn2 = Vn[0], Vn[0]
        else:
            Vn1, Vn2 = Vn[j + 1 - M2], Vn[j - M2]
        if M1 >= j:
            In1, In2 = In[0], In[0]
        else:
            In1, In2 = In[j + 1 - M1], In[j - M1]

        Sn.append((Sn[j] + Sn[j] * sagma1 * dwn1[j] + p * (
                    A + gamma1 * math.exp(-mu * tau1) * (theta * In1 + (1 - theta) * In2))) / (1 + p * (
                    alpha + mu + beta1 * In[j] / (1 + a1 * Sn[j] + a2 * In[j] + a3 * Sn[j] * In[j]))))
        Vn.append((Vn[j] + Vn[j] * sagma2 * dwn2[j] + p * (
                    gamma2 * math.exp(-mu * tau2) * (theta * Vn1 + (1 - theta) * Vn2) + alpha * Sn[j])) / (1 + p * (
                    gamma2 + mu + beta2 * In[j] / (1 + b1 * Vn[j] + b2 * In[j] + b3 * Vn[j] * In[j]))))
        In.append(In[j] * (1 + sagma3 * dwn3[j] + p * beta1 * Sn[j] / (
                    1 + a1 * Sn[j] + a2 * In[j] + a3 * Sn[j] * In[j]) + p * beta2 * Vn[j] / (
                                       1 + b1 * Vn[j] + b2 * In[j] + b3 * Vn[j] * In[j])) / (1 + p * (gamma1 + mu)))

    Sn = np.array(Sn)
    Vn = np.array(Vn)
    In = np.array(In)

    return np.array([Sn, Vn, In]).T

def groundtruth_predict(t_0=0, t_n=100, N=1000, nsim=1000, A=1, beta1=10, beta2=4.5, alpha=6, mu=1, theta=0.5, 
         a1=1.2, a2=0.3, a3=2.4, b1=1.4, b2=0.2, b3=0.1, 
         gamma1=1, gamma2=1.5, 
         S_0=0.3, V_0=0.4, I_0=0.3, 
         tau1=3,  tau2=5, 
         sagma1=0.03, sagma2=0.03, sagma3=0.05):
    
    start = time.time()
    
    S = []
    V = []
    I = []
    
    S.append(S_0)
    V.append(V_0)
    I.append(I_0)
    
    N = 1000
    h1 = (t_n - t_0) / N
    t = np.arange(0, t_n, h1)
    dw1 = np.sqrt(h1)*np.random.randn(nsim)
    dw2 = np.sqrt(h1)*np.random.randn(nsim)
    dw3 = np.sqrt(h1)*np.random.randn(nsim)
    
    
    M2 = int(tau2 / h1)
    M1 = int(tau1 / h1)
    ss1=S[0]; vv1=V[0]; ii1=I[0]
    c1 = (gamma1*(1 - math.e**(-mu*tau1)) +mu) * (alpha+mu)
    c2 = (gamma2*(1 - math.e**(-mu*tau2)) +mu) * (alpha+mu)
    k1 = c1 / (alpha+mu)
    k2 = beta2*alpha*A / (b1*A*alpha+c2)
    Diff = k1 - k2
    T = (mu/(mu+alpha)) + (alpha*mu/c2)

    # the first way
    R = (  (beta1*A) / ((a1*A+alpha+mu) * (gamma1+mu+(sagma3**2/2)))  ) + (  (beta2*alpha*A) / ((b1*alpha*A+c2) * (gamma1+mu+(sagma3**2/2)))  )
    Rd = (  (beta1*A) / ((a1*A+alpha+mu) * (gamma1+mu))  ) + (  (beta2*alpha*A) / ((b1*alpha*A+c2) * (gamma1+mu))  )
    L1 = beta1*A / (a1*A+alpha+mu) * (a2+a3*A / (alpha+mu)) + beta2*alpha*A / (b1*alpha*A+c2) * (b2+b3*alpha*A / c2)
    L2 = (  beta2*alpha*(c1*(b1*alpha*A+c2) + beta2*A*mu*(mu+alpha))  ) / (  (b1*alpha*A+c2)**2*(mu+alpha)  )
    # L3=beta1/((1+a1)*mu+alpha)*(beta1*mu/((1+a1)*mu+alpha)-gamma1*exp(-mu*tau1));
    infI = (  (gamma1+mu+sagma3**2/2) * (R-1)  ) / (L1+L2)
    DD = (  (beta1*mu) / ((1+a1)*mu+alpha)  ) - gamma1*math.e**(-mu*tau1)
    if R>1:
        maxS = (A / (alpha+mu)) - (1/(alpha+mu)) * ((c1/(alpha+mu)) - beta2*alpha*A/(b1*alpha*A+c2))*infI
        maxV = (A*(mu+alpha))/c2-(mu*(mu+alpha))/c2*maxS-c1/c2*infI
    else:
        maxS = A / (alpha+mu)
        maxV = (A*(mu+alpha))/c2-(mu*(mu+alpha))/c2*maxS     

    TT = infI + maxS + maxV
    
    #------------------------
    for j in range(N-1):
        if M2>=j:
            V1=V[0]
            V2=V[0]
        else:
            V1=V[j+1-M2]
            V2=V[j-M2]

        if M1>=j:
            I1=I[0]
            I2=I[0]
        else:
            I1=I[j+1-M1]
            I2=I[j-M1]

        S.append(S[j]+h1*(A-(alpha+mu)*S[j]-beta1*S[j]*I[j]/(1+a1*S[j]+a2*I[j]+a3*S[j]*I[j])+gamma1*np.exp(-mu*tau1)*(theta*I1+(1-theta)*I2)))
        V.append(V[j]+h1*(alpha*S[j]+gamma2*np.exp(-mu*tau2)*(theta*V1+(1-theta)*V2)-(mu+gamma2)*V[j]-beta2*V[j]*I[j]/(1+b1*V[j]+b2*I[j]+b3*V[j]*I[j])))
        I.append(I[j]+h1*(beta1*S[j]*I[j]/(1+a1*S[j]+a2*I[j]+a3*S[j]*I[j])+beta2*V[j]*I[j]/(1+b1*V[j]+b2*I[j]+b3*V[j]*I[j])-(gamma1+mu)*I[j]))
    
    end = time.time()
    computation_time = end - start
        
    return S, V, I, computation_time

def create_observations(n_observations = 10000):
    
    mu = np.random.uniform(low=0.5, high=1.0, size=n_observations)
    alpha = np.random.uniform(low=5.0, high=10.0, size=n_observations)
    b1 = np.random.uniform(low=5.5, high=10.5, size=n_observations)
    b2 = np.random.uniform(low=2.5, high=5.5, size=n_observations)
    gamma1 = np.random.uniform(low=0.5, high=1.5, size=n_observations)
    gamma2 = np.random.uniform(low=1.2, high=2.4, size=n_observations)
    tau1 = np.random.uniform(low=0.5, high=1.0, size=n_observations)
    tau2 = np.random.uniform(low=1.0, high=2.0, size=n_observations)

    observations = np.array([mu, alpha, b1, b2, gamma1, gamma2, tau1, tau2]).T
    obs_timesteps = []

    for s in tqdm(observations):
        pred = groundtruth_generate_obs(t_0=0, t_n=100, mu=s[0], alpha=s[1], b1=s[2], b2=s[3], gamma1=s[4], gamma2=s[5], tau1=s[6], tau2=s[7])
        step_index = np.arange(1, len(pred)+1).reshape(-1,1)
        
        s_t = np.tile(s,(len(pred),1))
        s_t_i = np.append(s_t, step_index, 1)
        s_t_i_obs = np.append(s_t_i, pred, 1)
        obs_timesteps.append(s_t_i_obs)

    obs_timesteps_reshaped = np.array(obs_timesteps)
    
    return obs_timesteps_reshaped

def SSTM_predict(t_0=0, t_n=100, N=1000, nsim=1000, A=1, beta1=10, beta2=4.5, alpha=6, mu=1, theta=0.5, 
         a1=1.2, a2=0.3, a3=2.4, b1=1.4, b2=0.2, b3=0.1, 
         gamma1=1, gamma2=1.5, 
         S_0=0.3, V_0=0.4, I_0=0.3, 
         tau1=3,  tau2=5, 
         sagma1=0.03, sagma2=0.03, sagma3=0.05):
    
    start = time.time()

    
    S = []
    V = []
    I = []
    S.append(S_0)
    V.append(V_0)
    I.append(I_0)
    
    N = 1000
    h1 = (t_n - t_0) / N
    t = np.arange(0, t_n, h1)
    dw1 = np.sqrt(h1)*np.random.randn(nsim)
    dw2 = np.sqrt(h1)*np.random.randn(nsim)
    dw3 = np.sqrt(h1)*np.random.randn(nsim)
    
    M2 = int(tau2 / h1)
    M1 = int(tau1 / h1)
    ss1=S[0]; vv1=V[0]; ii1=I[0]
    c1 = (gamma1*(1 - math.e**(-mu*tau1)) +mu) * (alpha+mu)
    c2 = (gamma2*(1 - math.e**(-mu*tau2)) +mu) * (alpha+mu)
    k1 = c1 / (alpha+mu)
    k2 = beta2*alpha*A / (b1*A*alpha+c2)
    Diff = k1 - k2
    T = (mu/(mu+alpha)) + (alpha*mu/c2)

    # the first way
    R = (  (beta1*A) / ((a1*A+alpha+mu) * (gamma1+mu+(sagma3**2/2)))  ) + (  (beta2*alpha*A) / ((b1*alpha*A+c2) * (gamma1+mu+(sagma3**2/2)))  )
    Rd = (  (beta1*A) / ((a1*A+alpha+mu) * (gamma1+mu))  ) + (  (beta2*alpha*A) / ((b1*alpha*A+c2) * (gamma1+mu))  )
    L1 = beta1*A / (a1*A+alpha+mu) * (a2+a3*A / (alpha+mu)) + beta2*alpha*A / (b1*alpha*A+c2) * (b2+b3*alpha*A / c2)
    L2 = (  beta2*alpha*(c1*(b1*alpha*A+c2) + beta2*A*mu*(mu+alpha))  ) / (  (b1*alpha*A+c2)**2*(mu+alpha)  )
    # L3=beta1/((1+a1)*mu+alpha)*(beta1*mu/((1+a1)*mu+alpha)-gamma1*exp(-mu*tau1));
    infI = (  (gamma1+mu+sagma3**2/2) * (R-1)  ) / (L1+L2)
    DD = (  (beta1*mu) / ((1+a1)*mu+alpha)  ) - gamma1*math.e**(-mu*tau1)
    if R>1:
        maxS = (A / (alpha+mu)) - (1/(alpha+mu)) * ((c1/(alpha+mu)) - beta2*alpha*A/(b1*alpha*A+c2))*infI
        maxV = (A*(mu+alpha))/c2-(mu*(mu+alpha))/c2*maxS-c1/c2*infI
    else:
        maxS = A / (alpha+mu)
        maxV = (A*(mu+alpha))/c2-(mu*(mu+alpha))/c2*maxS     

    TT = infI + maxS + maxV
    
    # ss1, vv1, ii1
    ss1 = S_0 
    vv1 = V_0 
    ii1 = I_0
    
    Sc, Vc, Ic = [ss1], [vv1], [ii1]

    for j in range(N-1):
        if M2 >= j:
            Vc1, Vc2 = Vc[0], Vc[0]
        else:
            Vc1, Vc2 = Vc[j+1-M2], Vc[j-M2]
        if M1 >= j:
            Ic1, Ic2 = Ic[0], Ic[0]
        else:
            Ic1, Ic2 = Ic[j+1-M1], Ic[j-M1]

        Sc.append(Sc[j] + h1 * (A - (mu + alpha) * Sc[j] - beta1 * Sc[j] * Ic[j] / (1 + a1 * Sc[j] + a2 * Ic[j] + a3 * Sc[j] * Ic[j]) + gamma1 * math.exp(-mu * tau1) * (theta * Ic1 + (1 - theta) * Ic2)) + sagma1 * Sc[j] * dw1[j] + 1/2 * sagma1**2 * Sc[j] * (dw1[j]**2 - h1))
        Vc.append(Vc[j] + h1 * (alpha * Sc[j] + gamma2 * math.exp(-mu * tau2) * (theta * Vc1 + (1 - theta) * Vc2) - (mu + gamma2) * Vc[j] - beta2 * Vc[j] * Ic[j] / (1 + b1 * Vc[j] + b2 * Ic[j] + b3 * Vc[j] * Ic[j])) + sagma2 * Vc[j] * dw2[j] + 1/2 * sagma2**2 * Vc[j] * (dw2[j]**2 - h1))
        Ic.append(Ic[j] + h1 * (beta1 * Sc[j] * Ic[j] / (1 + a1 * Sc[j] + a2 * Ic[j] + a3 * Sc[j] * Ic[j]) + beta2 * Vc[j] * Ic[j] / (1 + b1 * Vc[j] + b2 * Ic[j] + b3 * Vc[j] * Ic[j]) - (gamma1 + mu) * Ic[j]) + sagma3 * Ic[j] * dw3[j] + 1/2 * sagma3**2 * Ic[j] * (dw3[j]**2 - h1))
    
    end = time.time()
    computation_time = end - start
    
    return Sc, Vc, Ic, computation_time


def SSSTNSFD_predict(t_0=0, t_n=100, N=1000, nsim=1000, A=1, beta1=10, beta2=4.5, alpha=6, mu=1, theta=0.5, 
         a1=1.2, a2=0.3, a3=2.4, b1=1.4, b2=0.2, b3=0.1, 
         gamma1=1, gamma2=1.5, 
         S_0=0.3, V_0=0.4, I_0=0.3, 
         tau1=3,  tau2=5, 
         sagma1=0.03, sagma2=0.03, sagma3=0.05):
    
    start = time.time()
    
    S = []
    V = []
    I = []
    S.append(S_0)
    V.append(V_0)
    I.append(I_0)
    
    N = 1000
    h1 = (t_n - t_0) / N
    t = np.arange(0, t_n, h1)
    dw1 = np.sqrt(h1)*np.random.randn(nsim)
    dw2 = np.sqrt(h1)*np.random.randn(nsim)
    dw3 = np.sqrt(h1)*np.random.randn(nsim)
    
    M2 = int(tau2 / h1)
    M1 = int(tau1 / h1)
    ss1=S[0]; vv1=V[0]; ii1=I[0]
    c1 = (gamma1*(1 - math.e**(-mu*tau1)) +mu) * (alpha+mu)
    c2 = (gamma2*(1 - math.e**(-mu*tau2)) +mu) * (alpha+mu)
    k1 = c1 / (alpha+mu)
    k2 = beta2*alpha*A / (b1*A*alpha+c2)
    Diff = k1 - k2
    T = (mu/(mu+alpha)) + (alpha*mu/c2)

    # the first way
    R = (  (beta1*A) / ((a1*A+alpha+mu) * (gamma1+mu+(sagma3**2/2)))  ) + (  (beta2*alpha*A) / ((b1*alpha*A+c2) * (gamma1+mu+(sagma3**2/2)))  )
    Rd = (  (beta1*A) / ((a1*A+alpha+mu) * (gamma1+mu))  ) + (  (beta2*alpha*A) / ((b1*alpha*A+c2) * (gamma1+mu))  )
    L1 = beta1*A / (a1*A+alpha+mu) * (a2+a3*A / (alpha+mu)) + beta2*alpha*A / (b1*alpha*A+c2) * (b2+b3*alpha*A / c2)
    L2 = (  beta2*alpha*(c1*(b1*alpha*A+c2) + beta2*A*mu*(mu+alpha))  ) / (  (b1*alpha*A+c2)**2*(mu+alpha)  )
    # L3=beta1/((1+a1)*mu+alpha)*(beta1*mu/((1+a1)*mu+alpha)-gamma1*exp(-mu*tau1));
    infI = (  (gamma1+mu+sagma3**2/2) * (R-1)  ) / (L1+L2)
    DD = (  (beta1*mu) / ((1+a1)*mu+alpha)  ) - gamma1*math.e**(-mu*tau1)
    if R>1:
        maxS = (A / (alpha+mu)) - (1/(alpha+mu)) * ((c1/(alpha+mu)) - beta2*alpha*A/(b1*alpha*A+c2))*infI
        maxV = (A*(mu+alpha))/c2-(mu*(mu+alpha))/c2*maxS-c1/c2*infI
    else:
        maxS = A / (alpha+mu)
        maxV = (A*(mu+alpha))/c2-(mu*(mu+alpha))/c2*maxS     

    TT = infI + maxS + maxV
    
    # ss1, vv1, ii1
    ss1 = S_0 
    vv1 = V_0 
    ii1 = I_0
    
    Sn, Vn, In = [ss1], [vv1], [ii1]
    h2 = h1
    p = (np.exp(h2)-1)
    dwn1 = np.sqrt(h2)*np.random.randn(nsim)
    dwn2 = np.sqrt(h2)*np.random.randn(nsim)
    dwn3 = np.sqrt(h2)*np.random.randn(nsim)

    for j in range(N-1):
        if M2>=j:
            Vn1, Vn2 = Vn[0], Vn[0]
        else:
            Vn1, Vn2 = Vn[j+1-M2], Vn[j-M2]
        if M1>=j:
            In1, In2 = In[0], In[0]
        else:
            In1, In2 = In[j+1-M1], In[j-M1]

        Sn.append((Sn[j] + Sn[j] * sagma1 * dwn1[j] + p * (A + gamma1 * math.exp(-mu * tau1) * (theta * In1 + (1 - theta) * In2))) / (1 + p * (alpha + mu + beta1 * In[j] / (1 + a1 * Sn[j] + a2 * In[j] + a3 * Sn[j] * In[j]))))
        Vn.append((Vn[j] + Vn[j] * sagma2 * dwn2[j] + p * (gamma2 * math.exp(-mu * tau2) * (theta * Vn1 + (1 - theta) * Vn2) + alpha * Sn[j])) / (1 + p * (gamma2 + mu + beta2 * In[j] / (1 + b1 * Vn[j] + b2 * In[j] + b3 * Vn[j] * In[j]))))
        In.append(In[j] * (1 + sagma3 * dwn3[j] + p * beta1 * Sn[j] / (1 + a1 * Sn[j] + a2 * In[j] + a3 * Sn[j] * In[j]) + p * beta2 * Vn[j] / (1 + b1 * Vn[j] + b2 * In[j] + b3 * Vn[j] * In[j])) / (1 + p * (gamma1 + mu)))
        
    end = time.time()
    computation_time = end - start

    return Sn, Vn, In, computation_time


class lstm_model:
    def __init__(self, model_filename='lstm_model'):
        self.model = keras.models.load_model(model_filename)

    def predict(self, mu=1, alpha=6, b1=10, b2=4.5, gamma1=1, gamma2=1.5, tau1=3,  tau2=5, N=1000):
        
        sample = np.array([mu, alpha, b1, b2, gamma1, gamma2, tau1, tau2, 1])
        p_step_index = np.arange(1, N+1)
        sample_input = np.tile(sample, (N,1))
        sample_input[:,-1] = p_step_index
        sample_input = sample_input.reshape(1, N, 9)
        
        start = time.time()
        predictions = self.model.predict(sample_input)
        end = time.time()
        computation_time = end - start

        predictions = predictions.reshape(N, 3)

        lstm_s = predictions[:,0]
        lstm_v = predictions[:,1]
        lstm_i = predictions[:,2]

        return lstm_s, lstm_v, lstm_i, computation_time

    
our_model = lstm_model()

def generate_observation():
    o_mu = np.random.uniform(low=0.5, high=1.0, size=1).item()
    o_alpha = np.random.uniform(low=5.0, high=10.0, size=1).item()
    o_b1 = np.random.uniform(low=5.5, high=10.5, size=1).item()
    o_b2 = np.random.uniform(low=2.5, high=5.5, size=1).item()
    o_gamma1 = np.random.uniform(low=0.5, high=1.5, size=1).item()
    o_gamma2 = np.random.uniform(low=1.2, high=2.4, size=1).item()
    o_tau1 = np.random.uniform(low=0.5, high=1.0, size=1).item()
    o_tau2 = np.random.uniform(low=1.0, high=2.0, size=1).item()
    
    return o_mu, o_alpha ,o_b1, o_b2, o_gamma1, o_gamma2, o_tau1, o_tau2

def run_simulation(simulation_name = 'Simulation', mu=1, alpha=6, b1=10, b2=4.5, gamma1=1, gamma2=1.5, tau1=1, tau2=1.5):
    
    gt_s, gt_v, gt_i, gt_computation_time = groundtruth_predict(mu=mu, alpha=alpha, b1=b1, b2=b2, gamma1=gamma1, gamma2=gamma2, tau1=tau1,  tau2=tau2)
    SSTM_s, SSTM_v, SSTM_i, SSTM_computation_time = SSTM_predict(mu=mu, alpha=alpha, b1=b1, b2=b2, gamma1=gamma1, gamma2=gamma2, tau1=tau1,  tau2=tau2)
    SSSTNSFD_s, SSSTNSFD_v, SSSTNSFD_i, SSSTNSFD_computation_time  = SSSTNSFD_predict(mu=mu, alpha=alpha, b1=b1, b2=b2, gamma1=gamma1, gamma2=gamma2, tau1=tau1,  tau2=tau2)
    lstm_s, lstm_v, lstm_i, lstm_computation_time = our_model.predict(mu=mu, alpha=alpha, b1=b1, b2=b2, gamma1=gamma1, gamma2=gamma2, tau1=tau1,  tau2=tau2)

    font1 = {'family':'serif','color':'black','size':20}
    font2 = {'family':'serif','color':'black','size':15}

    fig, (axs1, axs2, axs3) = plt.subplots(1, 3)
    fig.set_figheight(7)
    fig.set_figwidth(25)
    time_range = np.arange(1,1001)/10

    axs1.plot(time_range, gt_s, label='Deterministic S(t)', linewidth=2)
    axs1.plot(time_range, gt_v, label='Deterministic V(t)', linewidth=2)
    axs1.plot(time_range, gt_i, label='Deterministic I(t)',linewidth=2)
    axs1.plot(time_range, SSTM_s, label='SSTM S(t)', linewidth=2)
    axs1.plot(time_range, SSTM_v, label='SSTM V(t)',linewidth=2)
    axs1.plot(time_range, SSTM_i, label='SSTM I(t)',linewidth=2)
    axs1.set_title('(a) SSTM', loc='center')
    axs1.legend(loc='upper right')

        # SSSTNSFD_s
    axs2.plot(time_range, gt_s, label='Deterministic S(t)', linewidth=2)
    axs2.plot(time_range, gt_v, label='Deterministic V(t)', linewidth=2)
    axs2.plot(time_range, gt_i, label='Deterministic I(t)',linewidth=2)
    axs2.plot(time_range, SSSTNSFD_s, label='SSSTNSFD S(t)', linewidth=2)
    axs2.plot(time_range, SSSTNSFD_v, label='SSSTNSFD V(t)',linewidth=2)
    axs2.plot(time_range, SSSTNSFD_i, label='SSSTNSFD I(t)',linewidth=2)
    axs2.set_title('(b) SSSTNSFD', loc='center')
    axs2.legend(loc='upper right')

    axs3.plot(time_range, gt_s, label='Deterministic S(t)', linewidth=2)
    axs3.plot(time_range, gt_v, label='Deterministic V(t)', linewidth=2)
    axs3.plot(time_range, gt_i, label='Deterministic I(t)',linewidth=2)
    axs3.plot(time_range, lstm_s, label='LSTM S(t)', linewidth=2)
    axs3.plot(time_range, lstm_v, label='LSTM V(t)',linewidth=2)
    axs3.plot(time_range, lstm_i, label='LSTM I(t)',linewidth=2)
    axs3.set_title('(c) LSTM', loc='center')
    axs3.legend(loc='upper right')

    plt.savefig(f'Results/{simulation_name}.png', bbox_inches='tight', dpi=400)

    with open(f"Results/{simulation_name}.txt", "w") as f:
        # Write the string to the file
        content = f'''
        Simulation #{simulation_name}
        
        This simulation was generated using the following parameters:
        
        mu = {np.round(mu, 3)}
        alpha = {np.round(alpha, 3)}
        beta_1 = {np.round(b1, 3)}
        beta_2 = {np.round(b2, 3)}
        gamma_1 = {np.round(gamma1, 3)}
        gamma_2 = {np.round(gamma2, 3)}
        tau_1 = {np.round(tau1, 3)}
        tau_2 = {np.round(tau2, 3)}
        
        
        Description : Numerical simulations of the extinction results of model (3) by the three constructed methods with h = 0.1 on [0, 100] compared with the corresponding deterministic model. (a) The simulation of S, V and I paths by SSTM method. (b) The simulation of S, V and I paths by SSSTNSFD method. (c) The simulation of S, V and I paths by the LSTM method
        '''
        f.write(content)