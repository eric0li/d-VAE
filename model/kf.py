import numpy as np
from numpy.linalg import pinv,inv
import pdb

class kf(object):
    def __init__(self,cfg):
        self.A_ = None
        self.Q_ = None
        self.H_ = None
        self.R_ = None
        self.alpha = 1e-6

    def fit(self, train_X, train_Y):
        N = train_Y.shape[1]
        # add bias to velocity Y
        train_Y = np.concatenate([np.ones([1,N]),train_Y],axis=0)
        train_Y = np.mat(train_Y)
        train_X = np.mat(train_X)
        # calculate A, Q
        Y_pre = train_Y[:,:-1]
        Y_post = train_Y[:,1:]
        A = Y_post*Y_pre.T*inv(Y_pre*Y_pre.T)
        A = np.mat(A)
        state_e = Y_post - A * Y_pre
        Q = np.dot(state_e, state_e.T)/(N-1)
        self.A = A
        self.Q = np.mat(Q)
        # calucate H, R
        H = train_X*train_Y.T*pinv(train_Y*train_Y.T)
        H = np.mat(H)
        ob_e = train_X[:,1:] - H * Y_post
        R = np.dot(ob_e,ob_e.T)/(N-1)+np.eye(ob_e.shape[0])*self.alpha
        self.H = H
        self.R = np.mat(R)
        # initialize Q
        self.P = np.mat(np.cov(Y_pre))
    def test(self, test_Y):
        pred = []
        x = np.array([1,0,0]).reshape([-1,1])
        x = np.mat(x)
        test_Y = np.mat(test_Y)
        pred.append(np.squeeze(np.asarray(x[1:,0])))
        N = test_Y.shape[1]
        for i in range(1,N):
            ya = self.A * x
            Pa = self.A * self.P * self.A.T + self.Q
            K = Pa * self.H.T * pinv(self.H * Pa * self.H.T + self.R)
            x = ya + K * (test_Y[:,i] - self.H * ya)
            self.P = Pa - K * self.H * Pa
            pred.append(np.squeeze(np.asarray(x[1:,:])))
        pred = np.transpose(np.asarray(pred))
        return pred
