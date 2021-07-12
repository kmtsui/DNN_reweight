from __future__ import absolute_import, division, print_function

import math

import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping

print(tf.__version__)

# Define default plot styles  

from matplotlib import rc
import matplotlib.font_manager

rc('font', family='serif')
rc('text', usetex=True)
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

plot_style_0 = {
    'histtype': 'step',
    'color': 'black',
    'linewidth': 2,
    'linestyle': '--',
    'density': False
}

plot_style_1 = {
    'histtype': 'step',
    'color': 'black',
    'linewidth': 2,
    'density': False
}

plot_style_2 = {'alpha': 0.5, 'density': False}


# Use uproot for ROOT I/O instead
import uproot
def getarray(filename):
    f = uproot.open(filename)
    t = f["RES"]
    array = np.transpose(np.asarray(t.arrays(["Enu","Q2","p_mu","costh_mu",
                          "W_had","gamma_had","p_pi","costh_pi","costh_mu_pi","mode","norm"], library="np", how=tuple)))
    weight = array[0,-1]
    array=array[array[:,-2]==11]
    return array[:,:-2], weight

array11_Eb0, w0 =getarray('nuwro_res_C_Eb0.root')
array11_Eb27, w27=getarray('nuwro_res_C_Eb27.root')


# Load the pre-trained model
from tensorflow import keras
from tensorflow_addons.optimizers import AdamW
model = keras.models.load_model('DNN_Eb27_example')

# from NN (DCTR)
def reweight(events):
    f = model.predict(events, batch_size=100)
    weights = f / (1. - f)
    weights[weights>10]=10
    return np.squeeze(np.nan_to_num(weights))

# get the reweight value
w_model = reweight(array11_Eb0[:,0:6])

# make plots
for plotNum in range(9):

        w_norm=1./(np.sum(w_model)/len(w_model))*w27*len(array11_Eb27)/w0/len(array11_Eb0) #properly weight to normalize the reweighted events

        bins_Enu = np.linspace(0.4, 6, 41)
        bins_q2 = np.linspace(0, 2, 21)
        bins_mumom = np.linspace(0, 6, 41)
        bins_mutheta = np.linspace(-1, 1, 41)
        bins_W = np.linspace(1.08, 1.6, 41)
        bins_gamma = np.linspace(1., 5, 41)
        bins_pimom = np.linspace(0, 1.500, 41)
        bins_pitheta = np.linspace(-1, 1, 41)
        bins_mupitheta = np.linspace(-1, 1, 41)
        

        xlabel = (r"$E_\nu (GeV)$",r"$Q2 (GeV^2)$",r"$p_\mu (GeV)$",r"$\cos\theta_\mu$",r"W (GeV)",r"$\gamma$",r"$p_\pi (GeV)$",r"$\cos\theta_\pi$",r"$\cos\theta_{\mu\pi}$")

        figname = ("Enu","Q2","pmu","costh_mu","W","gamma","ppi","costh_pi","costh_mupi")

        bins_array = (bins_Enu,bins_q2,bins_mumom,bins_mutheta,bins_W,bins_gamma,bins_pimom,bins_pitheta,bins_mupitheta)
        legend_loc = ('lower right','center right','center left','center left','center left','center left','center left','center left','center right')

        bins_w= np.linspace(0, 4, 101)

        fig, ax = plt.subplots(2,
                            2,
                            figsize=(12, 10),
                            constrained_layout=True,
                            #sharey='row'
                            )

        binwidth = bins_array[plotNum][1]-bins_array[plotNum][0]

        ax[0,0].set_xlabel(xlabel[plotNum])
        ax[0,0].set_ylabel(r'$d\sigma/dvar$')
        h1=ax[0,0].hist(array11_Eb0[:,plotNum], bins=bins_array[plotNum], **plot_style_2, label='Eb=0',weights=w0/binwidth*np.ones(len(array11_Eb0)),color="blue")
        h2=ax[0,0].hist(array11_Eb27[:,plotNum], bins=bins_array[plotNum], **plot_style_2, label='Eb=27MeV',weights=w27/binwidth*np.ones(len(array11_Eb27)),color="red")
        legend = ax[0,0].legend(
                    title='NuWro MC mode11',
                    loc=legend_loc[plotNum],
                    frameon=False)
        plt.setp(legend.get_title(), multialignment='center')
        ax2 = ax[0,0].twinx()
        ax2.set_ylabel('ratio')
        bincenter = 0.5 * (h1[1][1:] + h1[1][:-1])
        plt.errorbar(bincenter, h1[0]/h1[0], fmt='-', color="blue")
        plt.errorbar(bincenter, h2[0]/h1[0], fmt='-', color="red")

        ax[0,1].set_xlabel(xlabel[plotNum])
        ax[0,1].set_ylabel(r'$d\sigma/dvar$')
        #ax[0,1].hist(array11_Eb0[:,2], bins=bins_mutheta, **plot_style_2, label='Eb=0',weights=array11_Eb0[:,6]/40000000/0.05)
        #ax[0,1].hist(array11_Eb27[:,2], bins=bins_mutheta, **plot_style_2, label='Eb=27MeV',weights=array11_Eb27[:,6]/40000000/0.05)
        h3=ax[0,1].hist(array11_Eb0[:,plotNum], bins=bins_array[plotNum], **plot_style_2, label='Eb=0',weights=w0/binwidth*np.ones(len(array11_Eb0)),color="blue")
        h4=ax[0,1].hist(array11_Eb0[:,plotNum], bins=bins_array[plotNum], **plot_style_2, label='Eb=27MeV (Reweight)',
                        weights=w0/binwidth*np.double(w_model)*w_norm,color="orange")
        ax4 = ax[0,1].twinx()
        ax4.set_ylabel('ratio')
        plt.errorbar(bincenter, h3[0]/h3[0], fmt='-', color="blue")
        plt.errorbar(bincenter, h4[0]/h3[0], fmt='-', color="orange")
        plt.errorbar(bincenter, h2[0]/h1[0], fmt='-', color="red")
        plt.errorbar(bincenter,  h4[0]/h3[0]/(h2[0]/h1[0]), fmt='-', color="black",label="Ratio of Ratio")
        legend = ax[0,1].legend(
                    title='NuWro MC mode11',
                    loc=legend_loc[plotNum],
                    frameon=False)
        ax4.legend(loc='upper right',frameon=False)
        plt.setp(legend.get_title(), multialignment='center')

        ax[1,0].set_xlabel('reweight value')
        ax[1,0].hist(w_model, bins=bins_w, **plot_style_2, label='Eb=0',color="blue")

        ax[1,1].set_xlabel('reweight value')
        ax[1,1].set_ylabel(xlabel[plotNum])
        ax[1,1].hist2d(w_model, array11_Eb0[:,plotNum], bins=(bins_w,bins_array[plotNum]))

        #ax[2,0].hist2d(array11_Eb0[:,0], array11_Eb0[:,1], bins=(bins_mumom,bins_mumom))

        #ax[2,1].hist2d(array11_Eb27[:,0], array11_Eb27[:,1], bins=(bins_mumom,bins_mumom))

        fig.show()

        print("Saving", figname[plotNum]+".pdf")
        plt.savefig(figname[plotNum]+".pdf")