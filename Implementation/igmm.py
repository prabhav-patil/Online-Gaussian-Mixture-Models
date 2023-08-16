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
            #print(f"Is this not a vector of length 2: {self.mu[i]}\n Dimensions: {np.shape(self.mu[i])}")
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

class gen_samples:
    def __init__(self, pi, mu, C):
        self.X = list()
        self.pi = pi
        self.mu = mu
        self.C = C
        return
    
    def generate_samples(self, n_samples):
        iter = 1
        while(iter <= n_samples):
            z_i = np.argmax(np.random.multinomial(1, self.pi))
            sample = np.random.multivariate_normal(self.mu[z_i], self.C[z_i], 1)
            self.X.append(np.array(sample))
            iter += 1
        return

#Generation of Random Dataset
pi_value = [1/4, 3/4]
pi_value = np.array(pi_value)
mu_value = [[-5,0,0], [10,0,5]]
mu_value = np.array(mu_value)
C_value = [[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]]]
C_value = np.array(C_value)
sample_generator = gen_samples(pi_value, mu_value, C_value)
sample_generator.generate_samples(1000)
X = sample_generator.X
X = np.array(X)

#Setting up the parameters
sigma_threshold = (np.max(X)-np.min(X))/10
tau = 0.01
dim = 3
incremental_model = igmm(X,dim,sigma_threshold,tau)
incremental_model.fit()
