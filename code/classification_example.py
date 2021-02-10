#!/usr/bin/env python3
# General imports
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from modules import RC_model

#my imports
from visualize import plot_AB
import pickle

# ============ RC model configuration and hyperparameter values ============
config = {}
config['dataset_name'] = 'JpVow'

config['seed'] = 1
np.random.seed(config['seed'])

# Hyperarameters of the reservoir
config['n_internal_units'] = 450        # size of the reservoir
config['spectral_radius'] = 0.59        # largest eigenvalue of the reservoir
config['leak'] = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.25           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 1.0           # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 0                    # transient states to be dropped
config['bidir'] = False                  # if True, use bidirectional reservoir
config['circ'] = False                  # use reservoir with circle topology

config['IP'] = True
config['ab'] = (1.0, 0.0)
config['ip_parameters'] = (0.0005, 0.0, 0.1) #(eta, mu, sigma)

# Dimensionality reduction hyperparameters
config['dimred_method'] = None       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = None                    # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
config['mts_rep'] = 'last'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Type of readout
config['readout_type'] = 'identity'          # readout used for classification: {'lin', 'mlp', 'svm'} add identity

# Linear readout hyperparameters
config['w_ridge'] = 5.0                 # regularization of the ridge regression readout

# SVM readout hyperparameters
config['svm_gamma'] = 0.005             # bandwith of the RBF kernel
config['svm_C'] = 5.0                   # regularization for SVM hyperplane

# MLP readout hyperparameters
config['mlp_layout'] = (10,10)          # neurons in each MLP layer
config['num_epochs'] = 2000             # number of epochs 
config['w_l2'] = 0.001                  # weight of the L2 regularization
config['nonlinearity'] = 'relu'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}

print(config)

# ============ Load dataset ============
#data = scipy.io.loadmat('../dataset/'+config['dataset_name']+'.mat')
dataX = np.loadtxt('danni/'+'XtrD32_MackeyGlass_t17.txt').T
dataY = np.loadtxt('danni/'+'YtrD32_MackeyGlass_t17.txt').T
X = np.zeros((len(dataX[:,0]), 1, len(dataX[0,:])))
X[:,0,:] = dataX
Y = dataY
#print("X shape input: ", X.shape)
#print(Y.shape)
dataXte = np.loadtxt('danni/'+'XtsD32_MackeyGlass_t17.txt').T
dataYte = np.loadtxt('danni/'+'YtsD32_MackeyGlass_t17.txt').T
Xte = np.zeros((len(dataXte[:,0]), 1, len(dataXte[0,:])))
Xte[:,0,:] = dataXte
Yte = dataYte

print('Loaded '+config['dataset_name']+' - Tr: '+ str(X.shape)+', Te: '+str(Xte.shape))

# One-hot encoding for labels
#onehot_encoder = OneHotEncoder(sparse=False)
#Y = onehot_encoder.fit_transform(Y)
#Yte = onehot_encoder.transform(Yte)

# ============ Initialize, train and evaluate the RC model ============
classifier =  RC_model(
                        reservoir=None,     
                        n_internal_units=config['n_internal_units'],
                        spectral_radius=config['spectral_radius'],
                        leak=config['leak'],
                        connectivity=config['connectivity'],
                        input_scaling=config['input_scaling'],
                        noise_level=config['noise_level'],
                        circle=config['circ'],
                        n_drop=config['n_drop'],
                        bidir=config['bidir'],
                        IP=config['IP'],
                        ab=config['ab'],
                        ip_parameters=config['ip_parameters'],
                        #dimred_method=config['dimred_method'],
                        #n_dim=config['n_dim'],
                        mts_rep=config['mts_rep'],
                        w_ridge_embedding=config['w_ridge_embedding'],
                        readout_type=config['readout_type'],            
                        w_ridge=config['w_ridge'],              
                        #mlp_layout=config['mlp_layout'],
                        #num_epochs=config['num_epochs'],
                        #w_l2=config['w_l2'],
                        #nonlinearity=config['nonlinearity'],
                        #svm_gamma=config['svm_gamma'],
                        #svm_C=config['svm_C']
                        )

classifier.trainIP(X, 2)
plot_AB(2, config['n_internal_units'])

tr_time = classifier.train(X,Y)
print('Training time = %.2f seconds'%tr_time)

accuracy, f1 = classifier.test(Xte, Yte)
print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))

classifier.save_trained_model("model.pickle")

"""
with open("model.pickle", 'rb') as file:
    classifier = pickle.load(file)
"""