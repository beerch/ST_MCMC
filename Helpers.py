import numpy as np
import matplotlib.pyplot as plt
import time
# from BlogModel import *
# from KineticIsingModel import *

path = ("C:\\Users\\ilove\\PycharmProjects\\MScThesis1_SpinTransformer\\AC_data\\")

def sphered(input,R):
    norm = np.linalg.norm(input,axis=1)
    return input / norm[:,None] * R

def cos_sim(a,b):
    return [float((a[v] @ b[v].T)/(np.linalg.norm(a[v])*np.linalg.norm(b[v]))) for v in range(len(a))]

def plot_system(m_system,R):
    vx,vy = m_system[:,0],m_system[:,1]
    t0 = np.linspace(0, 2 * np.pi, len(vx),endpoint=False)
    x0 = R * np.cos(t0)
    y0 = R * np.sin(t0)
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.axis('square')
    plt.axis('off')
    plt.quiver(x0, y0, vx, vy, color='black')
    plt.plot(x0, y0, "o", ms=0.1, color='black')
    plt.show()

def cossim_m(hist,mbal,mness):
    csHist_mbal, csHist_mness = [], []
    for t in range(len(hist)):
        csHist_mbal.append(cos_sim(mbal, hist[t]))
        csHist_mness.append(cos_sim(mness, hist[t]))
    return np.array(csHist_mbal), np.array(csHist_mness)

def cossim_ms(hist,ms,m_t=False,N=64):
    if m_t:
        mness_s = np.zeros((len(ms)+1,len(hist),N))
    else:
        mness_s = np.zeros((len(ms),len(hist),N))
    for t in range(len(hist)):
        for i,m in enumerate(ms):
            mness_s[i,t,:] = cos_sim(m, hist[t])
        if m_t and t != 0:
            mness_s[-1,t,:] = cos_sim(hist[t-1],hist[t])
    return mness_s

def cossim_hist(hist, x_0, mness=None):
    csHist_mt, csHist_m0, csHist_x0 = [], [], []
    if mness is not None:
        csHist_MNESS = []
    for t in range(len(hist)):
        csHist_mt.append(cos_sim(hist[t - 1], hist[t]))
        csHist_m0.append(cos_sim(hist[0], hist[t]))
        csHist_x0.append(cos_sim(x_0, hist[t]))
        if mness is not None:
            csHist_MNESS.append(cos_sim(mness,hist[t]))
    if mness is not None:
        return np.array(csHist_mt), np.array(csHist_m0), np.array(csHist_x0), np.array(csHist_MNESS)
    return np.array(csHist_mt), np.array(csHist_m0), np.array(csHist_x0)

def plot_cs_norm(m_run,x,title='',mult=1):
    if len(m_run) > 10000:
        m_run = redu(m_run,10000)
    cs3 = cossim_hist(m_run, x)
    m_norms = np.linalg.norm(m_run, axis=2)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for i,color in enumerate(['green','yellow','blue']):
        axes[0].plot(np.arange(len(cs3[i]))*mult,cs3[i],c=color,lw=0.1)
    axes[0].set_title("cosine similarity")
    axes[1].plot(np.array(range(len(m_norms)))*mult, m_norms, c='blue', lw=0.1)
    axes[1].set_title("norm")
    fig.suptitle(title, fontsize=15)
    plt.show()

def plot_cs_norms(m_runs, cs_runs, run_mean=True, site_mean=False, title='',mult=1):
    norms = np.linalg.norm(m_runs, axis=0)
    if site_mean and run_mean:
        cs_runs = np.mean(cs_runs, axis=(3, 2))
        m_norms = np.mean(norms, axis=(1, 2))
    elif site_mean:
        cs_runs = np.mean(cs_runs, axis=3)
        m_norms = np.mean(norms, axis=2)
    elif run_mean:
        cs_runs = np.mean(cs_runs, axis=2)
        m_norms = np.mean(norms, axis=1)
    else:
        cs_runs = cs_runs.reshape(cs_runs.shape[0], cs_runs.shape[1], -1)
        m_norms = norms.reshape(norms.shape[0], -1)
    csHist_mt, csHist_m0, csHist_x0 = cs_runs

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].plot(np.array(range(len(csHist_mt))[1:])*mult, csHist_mt[1:], c='green', lw=0.1)
    axes[0].plot(np.array(range(len(csHist_m0)))*mult, csHist_m0, c='yellow', lw=0.1)
    axes[0].plot(np.array(range(len(csHist_x0)))*mult, csHist_x0, c='blue', lw=0.1)
    axes[0].set_title("cosine similarity")
    axes[1].plot(np.array(range(len(m_norms)))*mult, m_norms, c='blue', lw=0.1)
    axes[1].set_title("norm")
    fig.suptitle(title, fontsize=15)
    plt.show()

def simulate(model,steps,runs=1,xJ=False):
    if runs == 1:
        m_hist = model.single_run(steps=steps)
        print(f'shape1:{type(m_hist)}, len:{len(m_hist)}')
    else:
        m_runs = []
        for run in range(runs):
            m_hist = model.single_run(steps=steps)
            m_runs.append(m_hist)

def energy_state(state,prev_state,x0,J,prev=True):
    H = -np.trace(np.matmul(state,(x0 + np.matmul(J,state)).T))
    return H

def energy(mh,x0,J,prev=True):
    steps,N,D = np.shape(mh)
    energys = np.zeros((steps,))
    for t in np.arange(1,len(mh)):
        energys[t]=(energy_state(mh[t],mh[t-1],x0,J,prev=prev))
    return energys

def generic_xJm(N,D):
    x_0 = np.random.normal(loc=0, scale=1, size=(N, D))
    x = ((D/2 - 1) ** 0.5) * x_0 / np.linalg.norm(x_0, axis=-1, keepdims=True)
    J = N**-0.5 * np.random.normal(loc=0, scale=1, size=(N, N))
    m_0 = np.ones((N, D))
    m = m_0 / np.linalg.norm(m_0, axis=-1, keepdims=True)
    return x,J,m

def load_generic_xJM(N=64,D=32):
    try:
        x = np.load(path+'sample_x.npy')
        J = np.load(path+'sample_J.npy')
        m = np.load(path+'sample_m.npy')
        return x,J,m
    except:
        print('generating sample x,J,m')
        xJm = generic_xJm(N,D)
        for i,tag in enumerate(['x','J','m']):
            np.save(path+f'sample_{tag}.npy',xJm[i])
        return xJm

def normalize_state(state,A,method='per site'):
    norm_state = np.copy(state)
    if method == 'per site':
        for i,site in enumerate(state):
            norm_state[i] = site * A/np.linalg.norm(site)
    elif method == 'whole':
        norm_state = norm_state * A / np.mean(np.linalg.norm(norm_state,axis=1))
    return norm_state

def redu(m,l=100000,method='avg'): #reduce by averaging
    if method == 'avg':
        return m.reshape(-1,int(len(m)/l),np.shape(m)[1],np.shape(m)[2]).mean(axis=1)
    elif method == 'skip':
        return m[::int(len(m) / l)]

def AutoCorrelation(data):
    t = len(data)
    autocorr_0 = (((1 / t) * np.sum(data**2)) - ((1 / t) ** 2 * np.sum(data) ** 2))
    autocorr = [autocorr_0/autocorr_0]
    for k in np.arange(1, t):
        a = t - k
        sum1 = np.sum(data[:a] * data[k:])
        sum2 = np.sum(data[:a]) * np.sum(data[k:])
        corr = (1 / a) * sum1 - ((1 / a) ** 2) * sum2
        if corr < 0:
            break
        autocorr.append(corr/autocorr_0)
    return autocorr

def state_visual(states,titles=None):
    fig, axs = plt.subplots(1,len(states))
    if len(states)==1:
        axs.xaxis.set_ticklabels([])
        axs.yaxis.set_ticklabels([])
        c = axs.imshow(states[0], cmap='viridis')
        axs.set(ylabel='Site')
        axs.set(xlabel='D')
        fig.colorbar(c,ax=axs)
    else:
        for i,state in enumerate(states):
            axs[i].xaxis.set_ticklabels([])
            axs[i].yaxis.set_ticklabels([])
            c = axs[i].imshow(state, cmap='viridis')
            if titles is not None:
                axs[i].set_title(titles[i])
        fig.colorbar(c, ax=axs.ravel().tolist(), orientation='horizontal')
    plt.show()

