import numpy as np
import matplotlib.pyplot as plt
import time
from Helpers import *

class SpinIsingMC:
    def __init__(self, N=64, D=32, B=1,
                 init_x=None,
                 init_J=None,
                 init_m=None,
                 x_method='simpledot',
                 x_var=1/(np.sqrt(32)*64),
                 attn_version='v1',
                 attn_norm = 'blog',
                 sxy2_toggle= True,
                 proposal_method='normal',
                 B_prime=None):

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

        self.x_method = x_method
        self.x_var = x_var
        self.attn_version = attn_version
        self.attn_norm = attn_norm
        self.sxy2_toggle= sxy2_toggle
        self.proposal_method = proposal_method
        self.B_prime = B_prime

    def MetropolisIteration(self,mx_,prev_system_state):
        '''Choose a random site'''
        my_ = np.array(mx_)
        index = np.random.randint(self.N)
        s_t0 = np.array(my_[index])

        if self.proposal_method == 'indiv':
            dim = np.random.randint(self.D)
            s_t1 = np.array(s_t0)
            new_ = np.random.normal(loc=s_t0[dim],scale=self.x_var)
            s_t1[dim] = new_

        '''Propose a new site'''
        if self.proposal_method == 'normal':
            s_t1 = []
            for x in s_t0:
                s_t1.append(np.random.normal(loc=x,scale=self.x_var))
            s_t1 = np.array(s_t1)


        '''Accept/reject proposition based on calculated energy'''
        'dot product method'
        if self.x_method == 'simpledot':
            delta_s = s_t0 - s_t1
            attention = self.x_0[index] + np.matmul(self.J[index], prev_system_state)

            A = 1
            try:
                A = float(self.attn_norm)
            except ValueError:
                #print(f'attn not a float; {type(A)} : {A}')
                if self.attn_norm == 'blog':
                    A = self.B / (1 + np.sqrt(1 + (self.B ** 2) * (np.linalg.norm(attention) ** 2) / (self.R ** 2)))

            if self.attn_version == 'v1':
                delta_H = A*np.matmul(delta_s,attention.T) + A*np.matmul(attention,delta_s.T)
                if self.sxy2_toggle:
                    delta_H += np.matmul(s_t1,s_t1.T) - np.matmul(s_t0,s_t0.T)

        if self.x_method == 'fulldot':
            A = B = 1
            try:
                A = B = float(self.attn_norm)
            except:
                if self.attn_norm == 'state':
                    A = 1/np.linalg.norm(self.x_0 + np.matmul(self.J,my_),axis=1)[:,None]
                    B = 1/np.linalg.norm(self.x_0 + np.matmul(self.J,mx_),axis=1)[:,None]
            my_[index] = s_t1
            term1 = np.matmul(my_,my_.T) - np.matmul(mx_,mx_.T)        #my^2 mx^2
            term2 = 2*np.matmul((np.matmul(self.J,(A*A*my_ - B*B*mx_)) - A*my_ + B*mx_),self.x_0.T)   #2J[]x
            term3 = 2*np.matmul(self.J,(B*np.matmul(mx_,mx_.T) - A*np.matmul(my_,my_.T)))        #2J[]
            term4 = np.matmul(np.matmul(self.J,(A*A*np.matmul(my_,my_.T) - B*B*np.matmul(mx_,mx_.T))),self.J.T)        #J[]J.T
            delta_H = np.trace(term1+term2+term3+term4)

        if self.x_method == 'direct':
            my_[index] = s_t1
            term1 = np.matmul((my_ - mx_),self.x_0.T)
            term2 = np.matmul((np.matmul(my_,my_.T) - np.matmul(mx_,mx_.T)),self.J.T)
            delta_H = np.trace(term1+term2)

        update_bool = False
        if delta_H < 0:
            update_bool = True
        elif delta_H > 0:
            if self.B_prime is not None:
                probability = np.exp(-1 * self.B_prime * delta_H)
            else:
                probability = np.exp(-1 * self.B * delta_H)
            if np.random.rand() < probability:
                update_bool = True

        if update_bool:
            my_[index] = s_t1
        else:
            my_[index] = s_t0
        return my_

    def single_run(self,steps=50000):
        m_hist = np.zeros((steps,self.N,self.D))
        m_hist[0] = self.m_0
        for t in range(steps-1):
            if t == 0:
                prev_m_system = m_hist[t]
            else:
                prev_m_system = m_hist[t-1]
            m_hist[t+1] = self.MetropolisIteration(m_hist[t], prev_m_system)
        return m_hist