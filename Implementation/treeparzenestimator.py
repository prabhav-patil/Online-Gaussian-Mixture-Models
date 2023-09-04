#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from igmm import igmm
from spurious_removal import deletespurious

class gen_samples:
    def __init__(self, pi, mu, C, dim):
        self.X = list()
        self.pi = pi
        self.mu = mu
        self.C = C
        self.dim = dim
        return
    
    def generate_samples(self, n_samples):
        iter = 1
        while(iter <= n_samples):
            z_i = np.argmax(np.random.multinomial(1, self.pi))
            if(self.dim==1):
                sample = np.random.normal(self.mu[z_i][0], self.C[z_i][0], 1)
            else:
                sample = np.random.multivariate_normal(self.mu[z_i], self.C[z_i], 1)
            self.X.append(np.array(sample))
            iter += 1
        return
    
def gaussian(x, mean, std):
    return np.exp(-0.5*((x-mean)/std)**2)/(std*np.sqrt(2*np.pi))

def plot_generated_samples(data_points,pi_value,mu_value,C_value,dim):
    if dim==1:
        x_vals = np.linspace(min(data_points) - 1, max(data_points) + 1, 500)
        y_mixture = np.zeros_like(x_vals)
        for i in range(len(pi_value)):
            y_vals = pi_value[i]*gaussian(x_vals,mu_value[i][0],C_value[i][0])
            y_mixture = y_mixture + y_vals
        plt.plot(x_vals,y_mixture,label='Source Mixture')
        plt.scatter(data_points, np.zeros_like(data_points), color='red', label='Generated Samples')
        plt.xlabel('X')
        plt.ylabel('Density')
        plt.title('Generated Samples from Source Mixture')
        plt.legend()
        plt.grid()
        plt.show()
    return


# In[2]:


#source
dim = 1
num_samples = 1000
pi_value = [1/8, 7/8]
pi_value = np.array(pi_value)
mu_value = [[-30],[20]]
mu_value = np.array(mu_value)
C_value = [[[1]],[[5]]]
C_value = np.array(C_value)
sample_generator = gen_samples(pi_value, mu_value, C_value,dim)
sample_generator.generate_samples(num_samples)
X = sample_generator.X
X = np.array(X)
plot_generated_samples(X,pi_value,mu_value,C_value,dim)


# In[3]:


from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm

class online_gmm:
    def __init__(self, X, dim, sigma_ini, tau, confidence):
        self.X = X
        self.dim = dim
        self.sigma_ini = sigma_ini
        self.tau = tau
        self.confidence = confidence
        self.pi = list()
        self.mean = list()
        self.cov = list()
        
    def fit(self):
        incremental_model = igmm(self.X,self.dim,self.sigma_ini,self.tau)
        incremental_model.fit()
        
        M = len(incremental_model.pi)
        params = []
        for k in range(M):
            current_components = list()
            current_components.append(incremental_model.pi[k])
            current_components.append(incremental_model.mu[k])
            current_components.append(incremental_model.C[k])
            current_components.append(incremental_model.sp[k])
            params.append(current_components)
        remove_spurious = deletespurious(params, self.dim, self.confidence)
        remove_spurious.deleteLMspurious()
        
        n = len(remove_spurious.params)
        self.pi = list()
        self.mean = list()
        self.cov = list()
        for i in range(n):
            self.pi.append(remove_spurious.params[i][0])
            self.mean.append(remove_spurious.params[i][1])
            self.cov.append(remove_spurious.params[i][2])
        self.pi = np.array(self.pi)
        self.mean = np.array(self.mean)
        self.cov = np.array(self.cov)
        bic_value = self.calculate_bic(n)
        return bic_value        

    def calculate_bic(self, n_components):
        log_likelihood = 0.0
        n_samples = len(self.X)
        for i in range(n_components):
            component_likelihood = multivariate_normal.pdf(self.X[0], mean=self.mean[i][0], cov=self.cov[i])
            log_likelihood += self.pi[i] * component_likelihood
        k = (n_components - 1) + n_components * (self.dim + self.dim*(self.dim + 1) / 2)
        bic = -2 * log_likelihood + k * np.log(n_samples)
        return bic       


# In[4]:


from sklearn.neighbors import KernelDensity

class tpe_opt:
    def __init__(self,X,dim,max_iter,Y,confidence):
        self.X = X
        self.dim = dim
        self.max_iter = max_iter
        self.confidence = confidence
        self.Y = Y
        self.samples = list()
        self.loss_values = list()
        self.P_samples = list()
        self.Q_samples = list()
        
    def fit(self):
        num_samples = 10
        tau_values = np.random.uniform(0,0.5,num_samples)
        sigma_values = np.random.uniform(0.5,10,num_samples)
        for i in range(num_samples):
            self.samples.append([tau_values[i],sigma_values[i]])
            model = online_gmm(self.X, self.dim, sigma_values[i], tau_values[i], self.confidence)
            loss_value = model.fit()
            self.loss_values.append(model.fit())
            if(loss_value < self.Y):
                self.P_values.append([tau_values[i],sigma_values[i]])
            else:
                self.Q_values.append([tau_values[i],sigma_values[i]])
        
        for iter in range(self.max_iter):
            l_kde = KernelDensity(kernel='gaussian', bandwidth=5.0)
            g_kde = KernelDensity(kernel='gaussian', bandwidth=5.0)
            l_kde.fit(self.P_values)
            g_kde.fit(self.Q_values)
            
            n_samples = 100
            samples = l_kde.sample(n_samples)
            l_score = l_kde.score(samples)
            g_score = g_kde.score(samples)
            hps = samples[np.argmax(g_score/l_score)]
            
            model = online_gmm(self.X, self.dim, hps[1], hps[0], self.confidence)
            loss_value = model.fit()
            if(loss_value < self.Y):
                self.P_values.append(hps)
            else:
                self.Q_values.append(hps)
        return hps
            


# In[ ]:




