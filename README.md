# DNN_reweight

In some physics problems we want to reweigh the MC dataset generated from one model to another with different underlying parameters. The conventional way is to construct the probability density of the variables of interest in each model (say p(x) and q(x)), and create weights as w(x)=p(x)/q(x) so that it produces a statistically identical dataset after reweight. The downside is that the reweight performance is strongly dependent on the binning definition. 

On the other hand, it is known that neural networks learn to approximate the likelihood ratio = q(x)/p(x) (or something monotonically related to it in a known way), so we can just train a neural network to distinguish the two datasets. This turns the problem of density estimation (hard) into a problem of classification (easy).

This repository contains an example of model reweight inspired by the OmniFold algorithm (see https://github.com/hep-lbdl/OmniFold). The procedure includes training a deep neural network (DNN) in tensorflow for classification, and use the DNN to derive reweight functions. Additionally there is an example of using the tensorflow model in C++ with the help of the cppflow api.

## DNN training and reweight in tensorflow
### DNN_training.py
Example code to train the DNN. `nuwro_res_C_Eb0.root` and `nuwro_res_C_Eb27.root` contain the neutrino resonant event vectors on carbon produced by the NuWro generator, with binding energy Eb=0 and Eb=27 MeV respectively. Python module `uproot` is used to read the `.root` files. The training varaibles are neutrino energy `Enu`, four momentum transfer squared `Q2`, muon momentum `p_mu`, muon cosine angle `costh_mu`, hadronic invariant mass `W_had`, Lorentz factor of the hadronic system `gamma_had`. The DNN is constructed and trained by the `tensorflow` module to classify the Eb=0 and Eb=27 MC datasets. Finally the trained model is saved in the folder `DNN_Eb27`.

### DNN_reweight.py
Example code to reweigh the MC dataset using the trained DNN. The DNN gives a prediction `f` for each event which is the probability of being a true Eb=27 MeV MC event. To reweigh the Eb=0 MC dataset, each event is reweighed by `w=f/(1-f)`. The variable distributions of the Eb=0 and Eb=27 MeV MC are plotted against the reweighed Eb=0 MC to show the reweight performace. Apart from the `DNN_Eb27` model produced by `DNN_training.py` above, this repository also contains the `DNN_Eb27_example` model which is trained using the same configurations but with a larger dataset.

### DNN_reweight.cc
Effectively do the same job as `DNN_reweight.py` but in C++. The cppflow api (see https://github.com/serizba/cppflow) is used to run tensorflow models in C++. The api is slightly modified to make it C++11 compatible. `cmake` is used to compile the program.

Before the build, download the [Tensorflow C API](https://www.tensorflow.org/install/lang_c), extract the library to a directory you like, and set the environment variable `$TENSORFLOW_C` to that directory.

From within this cloned repository

```
$ mkdir build; cd build; cmake ../
$ make 
```

Run the progarm
```
$ ./DNN_reweight -i ../nuwro_res_C_Eb0.root -m ../DNN_Eb27_example/ -n 0.8745
```
