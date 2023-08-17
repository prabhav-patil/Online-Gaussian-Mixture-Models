import numpy as np
import math
from scipy.stats import multivariate_normal
class igmm:
    def __init__(self, X, dim, sigma_ini, tau):
        self.X = X
        self.dim = dim
        self.pi = list()
        self.mu = list()
        self.C = list()
        self.sp = list()
        self.sigma_ini = sigma_ini
        self.tau = tau
        
    def posterior_prob(self, x, component):
        mvn = multivariate_normal(mean = np.array(self.mu[component][0]), cov = np.array(self.C[component]))
        pdf_value = mvn.pdf(x)
        return self.pi[component]*pdf_value
    
    def createnewornot(self, x):
        createnew = True
        for i in range(len(self.pi)):
            novelty_criterion = self.tau/(((2*math.pi)**(dim/2))*math.sqrt(np.linalg.det(np.array(self.C[i]))))
            mvn = multivariate_normal(mean = self.mu[i][0], cov = np.array(self.C[i]))
            pdf_value = mvn.pdf(x)
            if(pdf_value >= novelty_criterion):
                createnew = False
                return createnew
        return createnew
    
    def update(self, x):
        createnew = self.createnewornot(x)
        if(createnew == True):
            self.mu.append(np.array(x))
            self.C.append((self.sigma_ini**2)*np.eye(dim))
            self.sp.append(1)
            
            total_sum = np.sum(self.sp)
            for j in range(len(self.sp)-1):
                self.pi[j] = self.sp[j]/total_sum
            self.pi.append(self.sp[len(self.sp)-1]/total_sum)
        else:
            for j in range(len(self.sp)):
                posterior_value = self.posterior_prob(x,j)
                self.sp[j] += posterior_value
                prev_mu = self.mu[j]
                self.mu[j] = self.mu[j] + (posterior_value/self.sp[j])*(np.array(x)-self.mu[j])
                self.C[j] = self.C[j] - np.matmul(self.mu[j]-prev_mu,(self.mu[j]-prev_mu).T) + (posterior_value/self.sp[j])*(np.matmul((np.array(x)-self.mu[j]),(np.array(x)-self.mu[j]).T)-self.C[j])
            total_sum = np.sum(self.sp)
            for j in range(len(self.pi)):
                self.pi[j] = self.sp[j]/total_sum
        return
    
    def fit(self):
        iter = 1
        for x in self.X:
            self.update(x)
            print(f"\n\n\033[1mFor next sample {iter}:\033[0m")
            print_pi = np.array(self.pi)
            print_mean = np.array(self.mu)
            print_cov = np.array(self.C)
            print_sp = np.array(self.sp)
            for j in range(len(self.mu)):
                print(f"\033[1mComponent #{j+1}:\033[0m\nPrior Probability: {print_pi[j]}\nMean: {print_mean[j][0]}\nContribution of dataset: {print_sp[j]}\nCovariance:\n {print_cov[j]}")
            iter+=1
        return
