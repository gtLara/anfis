import numpy as np
import time

class cmenas:
    def __init__(self, k, m = 2):
        # k : numero de clusters
        # C : matriz q guarda o espaco dos centros
        # U : matriz q mapeia as amostras nos grupos
        # J : funcao objetiva, de custo
        # m : parametro expoente de peso, normalmente usa-se m = 2
        self.k = k
        self.C = np.array([])
        self.U = np.array([])
        self.J = 0
        self.m = m

    def train(self, data, MAX, tol, log = True):
        # data : conjunto de treinamento
        # MAX : numero maximo de epocas
        # tol : tolerancia da funcao de custo
        num_amostras, num_atributos = data.shape
        #STEP 1: initialize random centers and add noisy
        index = np.random.choice(a = num_amostras, size = self.k, 
                                 replace = False)
        self.C = data[index, :]
        self.C += np.random.normal(size = self.C.shape) * 0.001
        #iteration
        ite = 0
        self.J = 1e5 + tol
        while (True):
            start_time = time.time()
            #STEP 2: compute distance matrix D and new U
            D = self.calc_dist(data = data, C = self.C)
            self.U = self.calc_memb(data = data, D = D, k = self.k, 
                                    m = self.m)
            #STEP 3: calculate cost function
            J_old = self.J
            self.J = self.calc_cost(U = self.U, D = D, m = self.m)
            #STEP 4: calculate centers
            self.C = self.calc_cent(data = data, U = self.U, k = self.k, 
                                    m = self.m)
            #condicao de parada
            ite += 1
            deltat = time.time() - start_time
            if (log): print(ite, 'loss:', self.J, 'duration:', deltat)
            if (ite >= MAX or tol > abs(self.J - J_old)): break

    def calc_dist(self, data, C):
        #dij = euclidean distance(ci - xj)
        D = np.einsum('ijk->ij',(C - data[:, np.newaxis, :]) ** 2)
        return np.sqrt(D)

    def calc_memb(self, data, D, k, m):
        #equation 15.9: uij = 1 / sum((dij / dkj) ^ (2 / (m - 1)))
        num_amostras, num_atributos = data.shape
        U = np.zeros([num_amostras, k])
        for i in range(k):
            dij = D[:, i]
            u = np.zeros([num_amostras, 1])
            for kk in range(k):
                dkj = D[:, kk]
                u += (dij / dkj).reshape(-1, 1) ** (2 / (m - 1))
            U[:, i:i+1] = 1 / u
        return U

    def calc_cost(self, U, D, m):
        #equation 15.6: J = sum(sum(uij ^ m * dij ^ 2))
        J = (U ** m * D ** 2).sum()
        return J

    def calc_cent(self, data, U, k, m):
        #equation 15.8: ci = sum(uij ^ m * xj) / sum(uij ^ m)
        C = np.matmul(np.transpose(U ** m), data)
        C /= np.sum(U ** m, axis = 0).reshape(-1, 1)
        return C