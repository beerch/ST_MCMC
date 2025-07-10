import numpy as np
import time
from Helpers import *

#Recreating blog post results
# based on magnetization updates

class MNESS_model:
    def __init__(self,N=64,D=32,B=1,
                 init_x=None,
                 init_J=None,
                 init_m=None,
                 norm='blog'):
        self.N, self.D, self.B = N, D, B
        self.R = (D/2 - 1) ** 0.5

        if init_x is None:
            self.x_0 = np.random.normal(loc=0, scale=1, size=(self.N, self.D))
            self.x_0 = self.R * self.x_0 / np.linalg.norm(self.x_0, axis=-1, keepdims=True)
        else:
            self.x_0 = init_x

        if init_J is None:
            self.J = self.N**-0.5 * np.random.normal(loc=0, scale=1, size=(self.N, self.N))
        else:
            self.J = init_J

        if init_m is None:
            self.m_0 = np.ones((self.N, self.D))
            self.m_0 = self.m_0 / np.linalg.norm(self.m_0, axis=-1, keepdims=True)
        else:
            self.m_0 = init_m

        self.norm = norm

    def update_naive_mf(self,m_prev):
        theta = self.x_0 + np.matmul(self.J,m_prev)
        if self.norm == 'none':
            return theta
        elif self.norm == 'blog':
            gamma = np.sqrt(1 + self.B**2 * (np.linalg.norm(theta,axis=1)**2) / self.R**2)
            return self.B*theta/(1+gamma[:,None])
        if type(self.norm) == type(1.) or type(1):
            return theta*self.norm

    def single_run(self,steps=50):
        m_hist = np.zeros((steps, self.N, self.D))
        m_hist[0] = self.m_0
        for t in range(steps-1):
                m_hist[t + 1] = self.update_naive_mf(m_hist[t])
        return m_hist