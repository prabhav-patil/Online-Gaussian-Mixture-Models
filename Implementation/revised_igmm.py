#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import igmm
from scipy.stats import multivariate_normal

class revised_igmm:
    def __init__(self, X, dim, sigma_ini, tau, outlierthreshold):
        self.X = X
        self.dim = dim
        self.sigma_ini = sigma_ini
        self.tau = tau
        self.t0 = outlierthreshold
        self.pi_obj = list()
        self.mu_obj = list()
        self.C_obj = list()
        self.sp_obj = list()
        self.pi_outlier = list()
        self.mu_outlier = list()
        self.C_outlier = list()
        self.sp_outlier = list()
        self.outlierdata = list()
        
    def posterior_prob(self, x, component, ctr):
        if(ctr==1):
            mvn = multivariate_normal(mean = np.array(self.mu_obj[component][0]), cov = np.array(self.C_obj[component]))
            pdf_value = mvn.pdf(x)
            return self.pi_obj[component]*pdf_value
        else:
            mvn = multivariate_normal(mean = np.array(self.mu_outlier[component][0]), cov = np.array(self.C_outlier[component]))
            pdf_value = mvn.pdf(x)
            return self.pi_outlier[component]*pdf_value
        
    def objclustermatch(self, x):
        createnew = True
        for i in range(len(self.pi_obj)):
            novelty_criterion = self.tau/(((2*math.pi)**(self.dim/2))*math.sqrt(np.linalg.det(np.array(self.C_obj[i]))))
            mvn = multivariate_normal(mean = self.mu_obj[i][0], cov = np.array(self.C_obj[i]))
            pdf_value = mvn.pdf(x)
            if(pdf_value >= novelty_criterion):
                createnew = False
                return createnew
        return createnew
    
    def createnewornot(self, x, ctr):
        createnew = True
        if(ctr==1):
            for i in range(len(self.pi_obj)):
                novelty_criterion = self.tau/(((2*math.pi)**(self.dim/2))*math.sqrt(np.linalg.det(np.array(self.C_obj[i]))))
                mvn = multivariate_normal(mean = self.mu_obj[i][0], cov = np.array(self.C_obj[i]))
                pdf_value = mvn.pdf(x)
                if(pdf_value >= novelty_criterion):
                    createnew = False
                    return createnew
        else:
            for i in range(len(self.pi_outlier)):
                novelty_criterion = self.tau/(((2*math.pi)**(self.dim/2))*math.sqrt(np.linalg.det(np.array(self.C_outlier[i]))))
                mvn = multivariate_normal(mean = self.mu_outlier[i][0], cov = np.array(self.C_outlier[i]))
                pdf_value = mvn.pdf(x)
                if(pdf_value >= novelty_criterion):
                    createnew = False
                    return createnew
        return createnew
    
    def igmm_update(self, x, ctr):
        if(ctr==1):
            for j in range(len(self.sp_obj)):
                posterior_value = self.posterior_prob(x,j,1)
                self.sp_obj[j] += posterior_value
                prev_mu = self.mu_obj[j]
                self.mu_obj[j] = self.mu_obj[j] + (posterior_value/self.sp_obj[j])*(np.array(x)-self.mu_obj[j])
                self.C_obj[j] = self.C_obj[j] - np.matmul((self.mu_obj[j]-prev_mu),(self.mu_obj[j]-prev_mu).T) + (posterior_value/self.sp_obj[j])*(np.matmul((np.array(x)-self.mu_obj[j]),(np.array(x)-self.mu_obj[j]).T)-self.C_obj[j])
            total_sum = np.sum(self.sp_obj)
            for j in range(len(self.pi_obj)):
                self.pi_obj[j] = self.sp_obj[j]/total_sum
        else:
            self.outlierdata.append(x)
            createnew = self.createnewornot(x,2)
            if(createnew == True):
                self.mu_outlier.append(np.array(x))
                self.C_outlier.append((self.sigma_ini**2)*np.eye(self.dim))
                self.sp_outlier.append(1)
            
                total_sum = np.sum(self.sp_outlier)
                for j in range(len(self.sp_outlier)-1):
                    self.pi_outlier[j] = self.sp_outlier[j]/total_sum
                self.pi_outlier.append(self.sp[len(self.sp_outlier)-1]/total_sum)
            else:
                for j in range(len(self.sp_outlier)):
                    posterior_value = self.posterior_prob(x,j,2)
                    self.sp_outlier[j] += posterior_value
                    prev_mu = self.mu_outlier[j]
                    self.mu_outlier[j] = self.mu_outlier[j] + (posterior_value/self.sp_outlier[j])*(np.array(x)-self.mu_outlier[j])
                    self.C_outlier[j] = self.C_outlier[j] - np.matmul((self.mu_outlier[j]-prev_mu),(self.mu_outlier[j]-prev_mu).T) + (posterior_value/self.sp_outlier[j])*(np.matmul((np.array(x)-self.mu_outlier[j]),(np.array(x)-self.mu_outlier[j]).T)-self.C_outlier[j])
                total_sum = np.sum(self.sp_outlier)
                for j in range(len(self.pi_outlier)):
                    self.pi_outlier[j] = self.sp_outlier[j]/total_sum
        return
    
    def checkoutlier(self):
        for i in range(len(self.pi_outlier)):
            if(self.sp_outlier[i] >= self.t0):
                return i
        return -1
    
    def update_outlierdata(self,index):
        dataset = list()
        for x in self.outlierdata:
            index_mx = -1
            mx_prob = 0
            for i in range(len(self.pi_outlier)):
                post_val = self.posterior_prob(x,i,2)
                if(mx_prob < post_val):
                    mx_prob = post_val
                    index_mx = i
            if(index_mx == index):
                dataset.append(x)
                self.outlierdata.remove(x)
        return dataset
    
    def objupdate(self, x):
        createnew = self.createnewornot(x,1)
        if(createnew == True):
            self.mu_obj.append(np.array(x))
            self.C_obj.append((self.sigma_ini**2)*np.eye(self.dim))
            self.sp_obj.append(1)
            
            total_sum = np.sum(self.sp_obj)
            for j in range(len(self.sp_obj)-1):
                self.pi_obj[j] = self.sp_obj[j]/total_sum
            self.pi_obj.append(self.sp_obj[len(self.sp_obj)-1]/total_sum)
        else:
            for j in range(len(self.sp_obj)):
                posterior_value = self.posterior_prob(x,j,1)
                self.sp_obj[j] += posterior_value
                prev_mu = self.mu_obj[j]
                self.mu_obj[j] = self.mu_obj[j] + (posterior_value/self.sp_obj[j])*(np.array(x)-self.mu_obj[j])
                self.C_obj[j] = self.C_obj[j] - np.matmul((self.mu_obj[j]-prev_mu),(self.mu_obj[j]-prev_mu).T) + (posterior_value/self.sp_obj[j])*(np.matmul((np.array(x)-self.mu_obj[j]),(np.array(x)-self.mu_obj[j]).T)-self.C_obj[j])
            total_sum = np.sum(self.sp_obj)
            for j in range(len(self.pi_obj)):
                self.pi_obj[j] = self.sp_obj[j]/total_sum
        return
    
    def outlierupdate(self, x):
        createnew = self.createnewornot(x,2)
        if(createnew == True):
            self.mu_outlier.append(np.array(x))
            self.C_outlier.append((self.sigma_ini**2)*np.eye(self.dim))
            self.sp_outlier.append(1)
            
            total_sum = np.sum(self.sp_outlier)
            for j in range(len(self.sp_outlier)-1):
                self.pi_outlier[j] = self.sp_outlier[j]/total_sum
            self.pi_outlier.append(self.sp_outlier[len(self.sp_outlier)-1]/total_sum)
        else:
            for j in range(len(self.sp_outlier)):
                posterior_value = self.posterior_prob(x,j,2)
                self.sp_outlier[j] += posterior_value
                prev_mu = self.mu_outlier[j]
                self.mu_outlier[j] = self.mu_outlier[j] + (posterior_value/self.sp_outlier[j])*(np.array(x)-self.mu_outlier[j])
                self.C_outlier[j] = self.C_outlier[j] - np.matmul((self.mu_outlier[j]-prev_mu),(self.mu_outlier[j]-prev_mu).T) + (posterior_value/self.sp_outlier[j])*(np.matmul((np.array(x)-self.mu_outlier[j]),(np.array(x)-self.mu_outlier[j]).T)-self.C_outlier[j])
            total_sum = np.sum(self.sp_outlier)
            for j in range(len(self.pi_outlier)):
                self.pi_outlier[j] = self.sp_outlier[j]/total_sum
        return    
    
    def insert_update(self,dataset):
        for x in dataset:
            self.objupdate(x)
        return
    
    def remove_update(self):
        self.pi_outlier = list()
        self.mu_outlier = list()
        self.C_outlier = list()
        self.sp_outlier = list()
        for x in self.outlierdata:
            self.outlierupdate(x)
        return
        
    def run(self):
        for x in self.X:
            if(self.objclustermatch(x)):
                self.igmm_update(x,1)
            else:
                self.igmm_update(x,2)
                outlier_to_obj = self.checkoutlier()
                if(outlier_to_obj):
                    datatomodify = self.update_outlierdata(outlier_to_obj)
                    self.insert_update(datatomodify)
                    self.remove_update()
        return
                    


# In[ ]:




