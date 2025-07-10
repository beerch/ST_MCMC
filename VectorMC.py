import numpy as np
import matplotlib.pyplot as plt
from Helpers import *
from matplotlib.gridspec import GridSpec
from BlogModel import MNESS_model
from MC_TAP import *
from numba import njit


class VectorSpinMC:
    def __init__(self, N=64, D=32, B=1,
                 init_xJm=None,
                 x_var=1/(np.sqrt(32)*64),
                 attn_norm = 'state',
                 reduction_type='block',
                 proposal_type='local',
                 threshold = 0.005
                 ):

        self.N, self.D, self.B = N, D, B
        if init_xJm is None:
            self.x0,self.J,self.m0 = load_generic_xJM(N,D)
        else:
            self.x0,self.J,self.m0 = init_xJm
        self.R = np.sqrt(self.D/2 -1)
        self.x_var = x_var
        self.attn_norm = attn_norm
        self.reduction_type = reduction_type
        self.proposal_type = proposal_type
        self.threshold = threshold
        norm_func = norm_state if self.attn_norm == 'state' else norm_blog
        if self.proposal_type == 'global':
            proposal_func = proposal_global
        elif self.proposal_type == 'global_fixR':
            proposal_func = proposal_global_fixR
        elif self.proposal_type == 'glocal':
            proposal_func = proposal_global_fixR
            print(f'glocal')
        elif self.proposal_type == 'local_flip':
            proposal_func = proposal_local_flip
        else:
            proposal_func = proposal_local
        self.fast_metropolis = make_fast_metropolis(
            self.x0, self.J, self.B, self.R, self.x_var, norm_func, proposal_func
        )

    def fast_long_run(self,million_steps=10,m_0=None,equilibrate=False,verbose=True):
        steps = million_steps*1e6
        if self.reduction_type == 'N':
            steps *= 64
        reduction = self.reduction()
        m_hist = np.empty((int(steps // reduction), self.N, self.D))
        idx = 0

        if m_0 is None:
            m_t = self.m0.copy()
        else:
            m_t = m_0.copy()

        threshold = self.threshold
        for t in range(int(steps)):
            m_t = self.fast_metropolis(m_t)
            if t % reduction == 0:
                m_hist[idx] = m_t
                idx += 1
                if equilibrate and idx>40 and idx % 10 == 0:
                    mag_hist = np.mean(np.linalg.norm(m_hist[idx-40:idx],axis=2),axis=1)
                    if np.std(mag_hist) < threshold:
                        break

        if verbose:
            if self.reduction_type == 'N':
                print(f"Completed {int(idx * self.reduction() / 1e6)}M/{int(idx)}k steps/time for run B={self.B}")
            else:
                print(f"Completed {int(idx*self.reduction()/1e6)}M steps for run B={self.B}")
        return m_hist[:idx]
    
    def glocal_run(self,steps_global=10,steps_local=1,loop=1):
        steps_g = steps_global * 1e6
        steps_l = steps_local * 1e6
        reduction = self.reduction()
        m_hist = np.empty((int(loop*(steps_g+steps_l) // reduction), self.N, self.D))
        idx = 0
        m_t = self.m0.copy()

        for l in range(loop):
            self.fast_metropolis = make_fast_metropolis(
                self.x0, self.J, self.B, self.R, self.x_var, norm_state, proposal_global_fixR)
            for t in range(int(steps_g)):
                m_t = self.fast_metropolis(m_t)
                if t % reduction == 0:
                    m_hist[idx] = m_t
                    idx += 1

            self.fast_metropolis = make_fast_metropolis(
                self.x0, self.J, self.B, self.R, self.x_var, norm_state, proposal_local)
            for t in range(int(steps_l)):
                m_t = self.fast_metropolis(m_t)
                if t % reduction == 0:
                    m_hist[idx] = m_t
                    idx += 1
            
        return m_hist[:idx]

    def short_run(self,steps=6400,m_0=None):
        if m_0 is None:
            s_t = self.m0.copy()
        else:
            s_t = m_0.copy()
        m_hist = np.zeros(shape=(steps, self.N, self.D))
        for t in range(int(steps)):
            s_t = self.fast_metropolis(s_t)
            m_hist[t] = s_t
        return m_hist

    # Declaring theta & beta
    def norm(self,m_):
        alpha = 1
        if self.attn_norm == 'state':
            alpha = self.R / np.linalg.norm(self.x0 + np.matmul(self.J, m_), axis=1)[:, None]
        if self.attn_norm == 'state_squared':
            alpha = (self.R**2) / (np.linalg.norm(self.x0 + np.matmul(self.J, m_), axis=1)**2)[:, None]
        elif self.attn_norm == 'blog':
            alpha = self.B / (1 + np.sqrt(
                1 + (self.B ** 2) * (np.linalg.norm(self.x0 + np.matmul(self.J, m_), axis=1) ** 2) / (self.R ** 2))[:, None])
        return alpha

    def energy_ising_loss(self,mh):
        steps, N, D = np.shape(mh)
        energys_ising = np.zeros((steps,))
        energys_loss = np.zeros((steps,))
        for t in np.arange(len(mh)):
            attn = self.norm(mh[t]) * (self.x0 + np.matmul(self.J, mh[t]))
            energys_ising[t] = -np.trace(np.matmul(mh[t], attn.T))
            loss = mh[t] - attn
            energys_loss[t] = np.trace(np.matmul(loss,loss.T))
        return energys_ising, energys_loss

    def reduction(self):
        if self.reduction_type == 'block':
            return 1e4
        if self.reduction_type == 'AC':
            return 1e3
        if self.reduction_type == 'normal':
            if self.B >= 10:
                return 1e4
            elif self.B < 10:
                return 1e5
        if self.reduction_type == 'N':
            return 64000

    def cs_bal_ness(self, m_hist):
        mbal = np.matmul(np.linalg.inv(np.identity(len(self.J)) - self.J), self.x0)
        mness = MNESS_model(B=self.B, init_x=self.x0, init_J=self.J, init_m=self.m0).single_run(steps=30)[-1]
        return cossim_m(m_hist, mbal, mness)

    def cs_ness_tap(self, m_hist):
        mness = MNESS_model(B=self.B, init_x=self.x0, init_J=self.J, init_m=self.m0).single_run(steps=30)[-1]
        mtap_hist = np.array(jaxtap(jnp.array(self.x0),jnp.array(self.J),jnp.array([self.m0]),jnp.arange(0, 100),self.B,(self.D / 2 - 1) ** 0.5))
        mtap = np.mean(mtap_hist, axis=0)[0]
        return cossim_m(m_hist, mness, mtap)

    def plot_TEs0(self, m_run, title=None, tap=False):
        if len(m_run) > 1000:
            m_run = redu(m_run, 1000)
        H_ising, H_loss = self.energy_ising_loss(m_run)
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 2.5))
        axes[0].plot(H_ising)
        axes[0].set_title(f"H_ising")
        axes[0].set_ylabel('H')
        axes[1].plot(H_loss)
        axes[1].set_title("H_loss")
        axes[1].set_ylabel('H')
        axes[2].plot(np.linalg.norm(m_run,axis=1))
        axes[2].set_title("euclidean norms")
        if tap:
            cs2,cs1 = self.cs_ness_tap(m_run)
            axes[3].plot(cs1[:,0], c='blue',label='tap')
        else:
            cs1,cs2 = self.cs_bal_ness(m_run)
            axes[3].plot(cs1[:,0], c='blue',label='lin')
        axes[3].plot(cs1[:,1:],c='blue')
        axes[3].plot(cs2[:,0], c='purple',label='ness')
        axes[3].plot(cs2[:,1:],c='purple')
        axes[3].set_title('cs similarity')
        axes[3].legend()
        fig.subplots_adjust(top=0.8)
        if title is not None:
            fig.suptitle(title)
        plt.show()

    def plot_TEs1(self, m_run, title=None):
        if len(m_run) > 1000:
            m_run = redu(m_run, 1000)
        H_ising = self.ness_sim(m_run)

        fig = plt.figure(figsize=(10,4))
        gs = GridSpec(2, 2, width_ratios=[1, 2])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[:,1:])

        ax1.plot(H_ising)
        ax1.set_ylim(-1,np.mean(H_ising)*0.9)
        ax1.set_ylabel('Alignment')

        ax2.plot(np.linalg.norm(m_run,axis=2))
        ax2.set_ylabel("Magnitudes")

        x, J, m = load_generic_xJM(self.N, self.D)
        mbal = np.matmul(np.linalg.inv(np.identity(len(self.J)) - self.J), self.x0)
        mnMF_Jz = MNESS_model(B=self.B, init_x=self.x0, init_J=self.J, init_m=self.m0).single_run(steps=30)[-1]
        mnMF_J = MNESS_model(B=self.B, init_x=x, init_J=J, init_m=m).single_run(steps=30)[-1]

        if self.B < 3.5:
            cossim_labels = ['m_bal', 'm_nMF(J)', 'm_tap(J)', 'm_nMF(Jzeta)', 'm_tap(Jzeta)', 'm(t-1)']
            colors = "bgrcmy"
            mtap_ = jaxtap(jnp.array(self.x0), jnp.array(self.J), jnp.array([self.m0]), jnp.arange(0, 30), self.B, (self.D / 2 - 1) ** 0.5)
            mTAP_Jz = np.transpose(mtap_, (1, 0, 2, 3))[0][-1]
            mtap_ = jaxtap(jnp.array(x), jnp.array(J), jnp.array([m]), jnp.arange(0, 100), self.B,self.R)
            mTAP_J = np.transpose(mtap_, (1, 0, 2, 3))[0][-1]
            ms = [mbal,mnMF_J,mTAP_J,mnMF_Jz,mTAP_Jz]
        else:
            cossim_labels = ['m_bal', 'm_nMF(J)', 'm_nMF(Jzeta)', 'm(t-1)']
            colors = "bgcy"
            ms = [mbal,mnMF_J,mnMF_Jz]

        css = cossim_ms(m_run,ms,m_t=True)
        for i,cs in enumerate(css):
            ax3.plot(np.mean(cs,axis=-1),c=colors[i],label=cossim_labels[i])
            ax3.plot(cs,c=colors[i],lw=0.2,alpha=0.3)
            # ax3.fill_between(range(len(m_run)),np.min(cs,axis=-1),np.max(cs,axis=-1),color=colors[i],alpha=0.3)
        ax3.set_title('Cosine similarities')
        ax3.legend()
        fig.subplots_adjust(top=0.8)
        if title is not None:
            fig.suptitle(title)
        plt.show()

    def plot_TEs2(self, m_run, title=None):
        H_ising = self.ness_sim(m_run)

        fig = plt.figure(figsize=(10,4))
        gs = GridSpec(2, 2, width_ratios=[1, 2])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[:,1:])

        ax1.plot(H_ising)
        ax1.set_ylabel('Alignment')

        ax2.plot(np.linalg.norm(m_run,axis=1))
        ax2.set_ylabel("Magnitudes")

        x, J, m = load_generic_xJM(self.N, self.D)
        mbal = np.matmul(np.linalg.inv(np.identity(len(self.J)) - self.J), self.x0)
        nmf_ness = MNESS_model(B=self.B, init_x=self.x0, init_J=self.J, init_m=self.m0).single_run(steps=30)[-1]

        cossim_labels = ['m_bal', 'nMF ness', 'm(t-1)']
        colors = "bgy"
        ms = [mbal,nmf_ness]
        css = cossim_ms(m_run,ms,m_t=True)
        for i,cs in enumerate(css):
            ax3.plot(np.mean(cs,axis=-1),c=colors[i],label=cossim_labels[i])
            ax3.plot(cs,c=colors[i],lw=0.2,alpha=0.3)
        ax3.set_title('Cosine similarities')
        ax3.legend()
        fig.subplots_adjust(top=0.8)
        if title is not None:
            fig.suptitle(title)
        plt.show()

    def ness_sim(self,m_hist):
        H_ising_normalised = np.zeros((np.shape(m_hist)[0],))
        for t in np.arange(len(m_hist)):
            A = self.norm(m_hist[t])
            attn = A * (self.x0 + np.matmul(self.J, m_hist[t]))
            H_ising = -np.trace(np.matmul(m_hist[t], attn.T))
            H_ising_normalised[t] = H_ising/(np.linalg.norm(m_hist[t])*np.linalg.norm(attn))
        return H_ising_normalised


    path = ("C:\\Users\\ilove\\PycharmProjects\\MScThesis1_SpinTransformer\\AC_data\\")

def make_fast_metropolis(x0, J, B, R, x_var, norm_func, proposal_func):
    @njit(cache=True, fastmath=True)
    def fast_metropolis(mx):
        N, D = mx.shape
        my = mx.copy()
        idx = np.random.randint(0, N)
        s_ty = proposal_func(mx[idx], x_var, R)
        # mag = np.sqrt(np.sum(s_ty**2, axis=-1))
        # print(mag)
        my[idx] = s_ty

        attn_T = x0 + J @ mx
        T = norm_func(attn_T, B, R)
        attn_A = x0 + J @ my
        A = norm_func(attn_A, B, R)
        delta_H =  0.5*(np.sum((my - A * (attn_A)) ** 2) - np.sum((mx - T * (attn_T)) ** 2))

        if delta_H < 0 or np.random.rand() < np.exp(-B * delta_H):
            return my
        else:
            return mx
    return fast_metropolis

@njit
def norm_state(attn, B, R):
    norm = np.sqrt(np.sum(attn**2, axis=1))[:, None]
    return R / norm

@njit
def norm_blog(attn, B, R):
    norm_sq = np.sum(attn**2, axis=1)[:, None]
    inner = 1 + (B ** 2) * norm_sq / (R ** 2)
    return B / (1 + np.sqrt(inner))

@njit
def proposal_local(s_t0, x_var,R):
    return s_t0 + np.random.normal(0.0, x_var, size=s_t0.shape)

@njit
def proposal_local_flip(s_t0, x_var,R):
    a = s_t0 + np.random.normal(0.0, x_var, size=s_t0.shape)
    if np.random.rand() < 0.5:
        return a
    else:
        return -a

from scipy.stats import uniform_direction

@njit
def proposal_global(s_t0, x_var,R):
    D = s_t0.shape
    prop_angle, prop_mag = np.random.normal(loc=0,scale=1,size=D), np.random.normal(loc=R,scale=x_var)
    s_t1 = prop_angle / np.sqrt(np.sum(prop_angle**2)) * prop_mag
    return s_t1

@njit
def proposal_global_fixR(s_t0, x_var,R):
    D = s_t0.shape
    prop_angle = np.random.normal(loc=0,scale=1,size=D)
    s_t1 = prop_angle / np.sqrt(np.sum(prop_angle**2)) * R
    return s_t1

# std = R,R,R,R
# mag = sqrt(R**2 + R**2 + R**2 + R**2)
# sqrt((r**2)/D) = R

