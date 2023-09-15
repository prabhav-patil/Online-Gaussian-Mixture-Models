import numpy as np
import math
from scipy.stats import multivariate_normal
class igmm:
    def __init__(self, X, dim, sigma_ini, tau, cluster_threshold):
        self.X = X
        self.dim = dim
        self.pi = list()
        self.mu = list()
        self.C = list()
        self.sp = list()
        self.temp = list()
        self.n0 = cluster_threshold
        self.sigma_ini = sigma_ini
        self.tau = tau
        
    def posterior_prob(self, x, component):
        mvn = multivariate_normal(mean = np.array(self.mu[component][0]), cov = np.array(self.C[component]))
        pdf_value = mvn.pdf(x)
        return self.pi[component]*pdf_value
    
    def posterior_prob_temp(self, x, component):
        mvn = multivariate_normal(mean = np.array(self.temp[component][0][0]), cov = np.array(self.temp[component][1]))
        pdf_value = mvn.pdf(x)
        total_sum = np.sum(self.temp[:,2])
        return self.temp[component][2]/total_sum*pdf_value
    
    def createnewornot(self, x):
        createnew = True
        for i in range(len(self.pi)):
            novelty_criterion = self.tau/(((2*math.pi)**(self.dim/2))*math.sqrt(np.linalg.det(np.array(self.C[i]))))
            mvn = multivariate_normal(mean = self.mu[i][0], cov = np.array(self.C[i]))
            pdf_value = mvn.pdf(x)
            if(pdf_value >= novelty_criterion):
                createnew = False
                return createnew
        return createnew
    
    def createnewtemp(self, x):
        createnew = True
        for i in range(len(self.temp)):
            novelty_criterion = self.tau/(((2*math.pi)**(self.dim/2))*math.sqrt(np.linalg.det(np.array(self.temp[i][1]))))
            mvn = multivariate_normal(mean = self.temp[i][0], cov = np.array(self.temp[i][1]))
            pdf_value = mvn.pdf(x)
            if(pdf_value >= novelty_criterion):
                createnew = False
                return createnew
        return createnew
    
    def update(self, x):
        createnew = self.createnewornot(x)
        if(createnew == True):
            createnewtemp = self.createnewtemp(x)
            if(createnewtemp == True):
                self.temp.append([np.array(x),(self.sigma_ini**2)*np.eye(self.dim), 1])
            else: 
                for j in range(len(self.temp)):
                    posterior_value = self.posterior_prob(x,j)
                    self.temp[j][2] += posterior_value
                    prev_mu = self.temp[0][j]
                    self.temp[j][0] = self.temp[j][0] + (posterior_value/self.temp[j][2])*(np.array(x)-self.temp[j][0])
                    self.temp[j][1] = self.temp[j][1] - np.matmul((self.temp[j][0]-prev_mu),(self.temp[j][0]-prev_mu).T) + (posterior_value/self.temp[j][2])*(np.matmul((np.array(x)-self.temp[j][0]),(np.array(x)-self.temp[j][0]).T)-self.temp[j][1])
                
                for j in range(len(self.temp)):
                    if(self.temp[2][j] >= self.n0):
                        self.mu.append(self.temp[j][0])
                        self.C.append(self.temp[j][1])
                        self.sp.append(self.temp[j][2])
            
                        total_sum = np.sum(self.sp)
                        for k in range(len(self.sp)-1):
                            self.pi[k] = self.sp[k]/total_sum
                        self.pi.append(self.sp[len(self.sp)-1]/total_sum)
                        
                        del self.temp[j]
                        return
        else:
            for j in range(len(self.sp)):
                posterior_value = self.posterior_prob(x,j)
                self.sp[j] += posterior_value
                prev_mu = self.mu[j]
                self.mu[j] = self.mu[j] + (posterior_value/self.sp[j])*(np.array(x)-self.mu[j])
                self.C[j] = self.C[j] - np.matmul((self.mu[j]-prev_mu),(self.mu[j]-prev_mu).T) + (posterior_value/self.sp[j])*(np.matmul((np.array(x)-self.mu[j]),(np.array(x)-self.mu[j]).T)-self.C[j])
            total_sum = np.sum(self.sp)
            for j in range(len(self.pi)):
                self.pi[j] = self.sp[j]/total_sum
        return
    
    def fit(self):
        iter = 1
        for x in self.X:
            self.update(x)
            print_pi = np.array(self.pi)
            print_mean = np.array(self.mu)
            print_cov = np.array(self.C)
            print_sp = np.array(self.sp)
            iter+=1
        return



