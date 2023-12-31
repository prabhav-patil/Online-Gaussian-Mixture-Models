#IGMM Model
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

#Generate Data Sample from Source Mixture
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

#Plot (1-D) Datasets
import matplotlib.pyplot as plt
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

#Implementation of IGMM
dim = 1
pi_value = [1/2, 1/2]
pi_value = np.array(pi_value)
mu_value = [[-10],[10]]
mu_value = np.array(mu_value)
C_value = [[4],[1]]
C_value = np.array(C_value)
sample_generator = gen_samples(pi_value, mu_value, C_value,dim)
sample_generator.generate_samples(1000)
X = sample_generator.X
X = np.array(X)
plot_generated_samples(X,pi_value,mu_value,C_value,dim)

sigma_threshold = (np.max(X)-np.min(X))/10
tau = 0.1
incremental_model = igmm(X,dim,sigma_threshold,tau)
incremental_model.fit()

#Removing Spurious Components Model
import numpy as np
import math
from scipy.stats import chi2

class deletespurious:
    def __init__(self, params, dim, confidence):
        self.params = params
        self.dim = dim
        self.confidence = confidence
    
    def delete_trivial(self):
        continue_update = True
        while(continue_update):
            continue_update = False
            num_components = len(self.params)
            delete_threshold = (math.pi/10)*(math.e ** (-1*(math.pi/10)*num_components))
            for k in range(num_components):
                if(self.params[k][0] <= delete_threshold):
                    del self.params[k]
                    continue_update = True
                    break
        return
    
    def inconfidenceornot(self, i, j):
        mahalanobis_distance = np.matmul(np.matmul(self.params[i][1]-self.params[j][1],np.linalg.inv(self.params[i][2])),(self.params[i][1]-self.params[j][1]).T)
        lower_critical_value = chi2.ppf((1-self.confidence)/2, dim)
        upper_critical_value = chi2.ppf(1-(1-self.confidence)/2, dim)
        if(lower_critical_value <= mahalanobis_distance <= upper_critical_value):
            return 1
        return 0
    
    def createLM(self):
        LM = []
        num_components = len(self.params)
        for i in range(num_components):
            ith_entries = list()
            for j in range(num_components):
                ith_entries.append(self.inconfidenceornot(i,j))
            LM.append(ith_entries)
        LM = np.array(LM)
        return LM
    
    def updateLM(self, LM, sum_LM, index_to_remove):
        updated_LM = np.delete(np.delete(LM, index_to_remove, axis=0), index_to_remove, axis=1)
        updated_sum_LM = np.delete(sum_LM, index_to_remove)
        del self.params[index_to_remove]
        return updated_LM, updated_sum_LM
    
    def deleteLMspurious(self):
        sorted_params = sorted(self.params, key=lambda x: x[0], reverse=True)
        self.params = sorted_params
        self.delete_trivial()
        LM = self.createLM()
        sum_LM = np.sum(LM, axis=1)
        continue_update = True
        while(continue_update):
            continue_update = False
            num_components = len(sum_LM)
            for i in range(num_components):
                for j in range(num_components):
                    if(LM[i][j]==1):
                        if(sum_LM[j]>=2):
                            continue_update = True
                            updated_LM, updated_sumLM = self.updateLM(LM, sum_LM, i)
                            LM = updated_LM
                            sum_LM = updated_sumLM
                        elif(sum_LM[j]==1 and self.params[i][0] <= self.params[j][0]):
                            continue_update = True
                            updated_LM, updated_sumLM = self.updateLM(LM, sum_LM, i)
                            LM = updated_LM
                            sum_LM = updated_sumLM
                        break
                if(continue_update):
                    break
        total_spsum = sum(x[3] for x in self.params)
        for i in range(len(self.params)):
            self.params[i][0] = self.params[i][3]/total_spsum
        return

#Plot (1-D) Output Mixture Model based on Dataset
def plot_output_fit(data_points,pi_value,mu_value,C_value,dim):
    if dim==1:
        x_vals = np.linspace(min(data_points) - 1, max(data_points) + 1, 500)
        y_mixture = np.zeros_like(x_vals)
        for i in range(len(pi_value)):
            y_vals = pi_value[i]*gaussian(x_vals,mu_value[i][0],C_value[i][0])
            y_mixture = y_mixture + y_vals
        plt.plot(x_vals,y_mixture,label='Output Mixture')
        plt.scatter(data_points, np.zeros_like(data_points), color='red', label='Generated Samples')
        plt.xlabel('X')
        plt.ylabel('Density')
        plt.title('Estimated Mixture from Generated Samples')
        plt.legend()
        plt.grid()
        plt.show()
    return

#Implementation of Removal Phase
M = len(incremental_model.pi)
params = []
for k in range(M):
    current_components = list()
    current_components.append(incremental_model.pi[k])
    current_components.append(incremental_model.mu[k])
    current_components.append(incremental_model.C[k])
    current_components.append(incremental_model.sp[k])
    params.append(current_components)
confidence = 0.95
remove_spurious = deletespurious(params, dim, confidence)
remove_spurious.deleteLMspurious()
output_prior = []
output_mean = []
output_C = []
for k in range(len(remove_spurious.params)):
    output_prior.append(remove_spurious.params[k][0])
    output_mean.append(remove_spurious.params[k][1])
    output_C.append(remove_spurious.params[k][2])
    print(f"\033[1mComponent #{k+1}:\033[0m\nPrior Probability:{remove_spurious.params[k][0]}\nMean:{remove_spurious.params[k][1]}\nContribution:{remove_spurious.params[k][3]}\nCovariance\n{remove_spurious.params[k][2]}")
plot_output_fit(X, output_prior, output_mean, output_C, dim)
