import numpy as np

class ANNBFIS:
    '''
    ANNBFIS - Artificial Neural Network Based on Fuzzy Inference System

    Attributes
    ----------
    data: np.ndarray
        training dataset
        (cases in rows, features in columns, output in the last column)
    nrule: int
        number of IF-THEN rules
    
    Methods
    -------
    ...
    '''
    def __init__(self, data: np.ndarray, nrule: int) -> None:
        self.data = data
        self.nrule = nrule
    
    def annbfis(self) -> tuple:
        n1, m1 = np.shape(self.data)
        iter = 100
        # c, s, cc = self.cclust(self.data[:,:-1], self.nrule, self.nrule, 4)
        c, s, cc = self.cclust(4)

        c = c[:,:m1-1].T
        s = s[:,:m1-1].T
        m = cc
        ss = 0.01
        a = np.zeros((m1, m))
        ww = 2* np.ones((1,m))
        co = 0
        EEMIN = 1E100
        print(EEMIN)
        lasti = 1
        lastd = 1
        EE = np.zeros((iter,1))

        print('- loop under learning epoch.\n')
        for I in range(iter):
            Ist = ' '
            if co == 0:
                P = 10E6 * np.array(np.eye(m1*m), dtype=float)
                ak = np.zeros((m1*m,1))

            if co == 1:
                gc = np.zeros((m1-1,m))
                gs = np.zeros((m1-1,m))
                gw = np.zeros((1,m))
            
            # LOOP under data cases
            for n in range(n1):
                d1 = np.dot(self.data[n,:m1-1].T.reshape((-1,1)), np.ones((1,m))) - c
                d2 = d1 / s
                d3 = np.copy(d2) * np.copy(d2)
                d4 = -0.5 * np.sum(np.copy(d3), axis=0)
                R = np.exp(np.copy(d4))
                mi = (R * ww) / 2
                [[z]] = np.dot(np.ones((1,m)), mi.T)
                aa = np.dot(self.data[n,:m1-1], a[:m1-1,:]) + a[m1-1,:]
                [y] = (np.dot(aa, mi.T)) / z
                e = np.abs(self.data[n,m1-1] - y)
                EE[I,0] = EE[I,0] + e*e
                if co == 0:
                    mi1 = mi / z
                    M = np.append(self.data[n,:m1-1].T,1).reshape(-1,1) * mi1
                    Rk = M.flatten('F').reshape(-1,1)   
                    P = P - (np.dot(np.dot(np.dot(P, Rk), Rk.T), P) / (np.dot(np.dot(Rk.T, P), Rk) + 1))
                    
                    ak = ak + np.dot(np.dot(P, Rk), (self.data[n,m1-1] - np.dot(Rk.T, ak)))
                    
                    if n == (n1-1):
                        a[:] = ak.reshape(-1,a.shape[0]).T
            
            
                if co == 1:     # gradient method
                    ay = (np.ones((m1-1,1))*aa - y) / z
                    gc = gc + np.dot((self.data[n,m1-1] - y), ay) * (np.ones((m1-1,1)) * mi) * (d1 / (s*s))
                    gs = gs + np.dot((self.data[n,m1-1] - y), ay) * (np.ones((m1-1,1)) * mi) * ((d1*d1) / (s*s*s))
                    gw = gw + np.dot((self.data[n,m1-1] - y), ((aa - y) / z)) * (R/2)

            if co == 1:     # a huge if
                if EEMIN > EE[I,0]:
                    Ist = '<--'
                    self.w = (a, ww, c, s)   # parameters to output in a tuple

                if (I - lasti) > 8:
                    if (EE[I,0] < EE[I-2,0]) and (EE[I-2,0] < EE[I-4,0]) and (EE[I-4,0] < EE[I-6,0]) and (EE[I-6,0] < EE[I-8,0]):
                        ss = ss*1.1
                        print(f'Step increasing {ss}\n')
                        lasti = I
                if (I - lastd) > 8:
                    if (EE[I,0] < EE[I-2,0]) and (EE[I-2,0] > EE[I-4,0]) and (EE[I-4,0] < EE[I-6,0]) and (EE[I-6,0] > EE[I-8,0]):
                        ss = ss*0.9
                        print(f'Step decreasing {ss}\n')
                        lastd = I
                
                Sss = ss / np.sqrt(np.sum(gc*gc) + np.sum(gs*gs) + np.sum(gw*gw))
                c = c + Sss * gc
                s = s + Sss * gs
                ww = ww + Sss * gw
            co = np.mod(co+1,2)
            print(f'Optimization method change into {co}\n')
            print(f'{I:4d}, {EE[I,0]/n1:.2e} {m:d} {Ist}')
        
        return self.w

    def cclust(self, ni: int = 4) -> tuple:
        '''
        Finds the best partition
            requires:
            --------------------------------------------
            mincl_n, maxcl_n: int
                minimal, maximal number of clusters (it 
                is set that mincl_n == maxcl_n)
            ni: int
                number of iteration
            
            returns:
            --------------------------------------------
            c: matrix centers
            s: cluster variability
            cc: optimal number of clusters
        '''

        # def cclust(data: np.ndarray, mincl_n: int, maxcl_n: int, ni: int):
        wsk = 1E30
        mincl_n = self.nrule
        maxcl_n = self.nrule
        cc = mincl_n

        for i in range(mincl_n, maxcl_n+1):
            for j in range(ni):
                c1, s1, U, XB, FS, SC, VAL = self.cluster(i)
                if VAL < wsk:
                    wsk = VAL
                    c = c1
                    s = s1
                    cc = i
        print(f'Minimal Value of index: {wsk:4f}\n')
        print(f'Number of Clusters: {cc}\n')
        
        return c, s, cc
    
    def cluster(self, n_clust: int) -> tuple:
        '''
        Fuzzy c-means clustering
            requires:
            --------------------------------------------
            dataset: np.ndarray
                cases in rows - this version of the 
                cluster method takes class attribute 
                and removes output from the last column
            n_clust: int
                number of clusters
            
            returns:
            --------------------------------------------
            V: cluster centers
            S: cluster variability
            U: partition matrix
            XB: Xie-Beni index
            FS: Fukuyama-Sugeno index
            SC: Bezdek Partition Index
            VAL: new index
        '''
        data = np.copy(self.data[:,:-1])
        # initial parameters
        n1, n2 = np.shape(data)
        m = 2
        iter = 500
        Jm_d = 10**(-5)
        Jm = np.zeros((iter,1))
        U = np.random.uniform(0, 1, (n_clust,n1))
        sum_c = U.sum(axis=0)
        temp_U = np.zeros((n_clust,n1))
        temp_U[:,:] = sum_c
        U = U/temp_U

        for i in range(iter):
            Um = U**(m)
            V = np.dot(Um, data) / (((np.ones((n2,1))) * (np.sum(Um.T, axis=0))).T)

            DD = np.zeros((n_clust, n1))
            for k in range(n_clust):
                DD[k,:] = np.sqrt(np.sum(((data - np.ones((n1,1)) * V[k,:])**(2)).T, axis=0))
            
            Jm[i,0] = np.sum(np.sum(((DD**(2))*Um), axis=0))
            U_new = DD**(-2/(m-1))
            U = U_new / (np.ones((n_clust,1)) * sum(U_new))

            if i > 0:
                if np.abs(Jm[i,0] - Jm[i-1,0]) < Jm_d:
                    break
        
        Um = U**(2)
        V2 = Jm[i,0] / n1
        V1 = 1E23
        for i in range(n_clust-1):
            for j in range(i+1,n_clust):
                tm = np.sum(((V[i,:] - V[j,:])**(2)).T)
                if tm < V1:
                    V1 = np.copy(tm)
        
        XB = V2 / V1
        VG = np.mean(data, axis=0)
        tm = np.sum(Um.T, axis=0)
        FS = Jm[i,0] - np.sum(tm * np.sum((V - (np.ones((n_clust,1)) * VG))**(2), axis=1))
        PI = (np.sum(((DD**(2)) * Um).T, axis=0)) / (np.sum(U.T, axis=0))
        
        SI = np.zeros((1,n_clust))
        for i in range(n_clust):
            SI[0,i] = np.sum(np.sum(((V - np.ones((n_clust,1)) * V[i,:])**(2)).T, axis=0), axis=0)
        
        SC = np.sum(PI/SI)
        PI = (np.sum(((DD**(2)) * Um).T, axis=0)) / (np.sum(Um.T, axis=0))
        PI = PI * np.sum(U.T, axis=0)
        nt = np.sum(U.T, axis=0)

        for i in range(n_clust):
            ni = nt + nt[i]
            SI[0,i] = np.sum(np.sum(((V - (np.ones((n_clust,1)) * V[i,:]))**(2)).T, axis=0) * ni, axis=0)
        
        VAL = np.sum(PI/SI)
        print(f'X-B: {XB:4f}, F-S: {FS:4f}, SC: {SC:4f}, VAL: {VAL:4f}\n')

        S = np.zeros((n_clust,n2))
        for k in range(n_clust):
            squared_diff = (data - np.ones((n1,1)) * V[k,:])**(2)
            temp = U[k,:] * squared_diff.T

            S[k,:] = np.sum(temp, axis=1)
            S[k,:] = S[k,:] / np.sum(U[k,:])
        S = np.sqrt(S)


        return V, S, U, XB, FS, SC, VAL
    
    def annbfise(self) -> np.ndarray:
        ''' 
        Method for evaluation of the ANNBFIS model

        IMPORTANT
        ---------------------------------------------------
            for now, this version of the annbfise method 
            uses the same dataset for testing and training
            (it's just a temporary solution to verify the 
            structure of the algorithm)
        ---------------------------------------------------
        '''

        if not hasattr(self, 'w'):
            self.w = self.annbfis()        
        
        (a, ww, c, s) = self.w
        n1, m1 = np.shape(self.data)
        _, m = np.shape(ww)         # just to get no. if-then rules

        out = np.zeros((n1,1))
        for n in range(n1):
            d1 = self.data[n,:m1-1].T.reshape(-1,1) * np.ones((1,m)) - c
            d2 = d1 / s
            d3 = d2 * d2
            d4 = -0.5 * np.sum(d3, axis=0)
            R = np.exp(d4)
            mi = (R * ww) / 2
            [[z]] = np.dot(np.ones((1,m)),mi.T)
            aa = (np.dot(self.data[m,:m1-1], a[:m1-1,:])) + a[m1-1,:]
            [out[n,0]] = (np.dot(aa,mi.T)) / z
        
        return out
