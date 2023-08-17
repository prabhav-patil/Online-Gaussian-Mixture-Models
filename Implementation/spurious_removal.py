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
    
    def updateLM(self, LM, sum_LP, index_to_remove):
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
                        num_components = len(sum_LM)
                        i-=1
                        if(j>=i):
                            j-=1
        total_spsum = sum(x[3] for x in self.params)
        for i in range(len(self.params)):
            self.params[i][0] = self.params[i][3]/total_spsum
        return
