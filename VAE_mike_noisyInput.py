# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:51:36 2022

@author: quick2063706271 
"""

import numpy as np
from sklearn import preprocessing

#%%
mean = np.array([20, 30])
cov = [[1, 0], [0, 100]]  # diagonal covariance

dataset = np.random.multivariate_normal(mean, cov, 10000)
dataset = dataset.astype(np.float32)
dim = dataset.shape[1]



#%%
# make sure of good reconstruction
def mse_loss(y_pred, y_true):
    loss = nn.MSELoss(reduction='sum', size_average=False)
    return loss(y_pred, y_true)

# make sure that the latent space is continuous and standard normal distributed
def kld_Loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

# loss functino of the VAE
def loss_function(y_pred, y_true, input_dim):
    recon_x, mu, logvar = y_pred
    x = y_true
    KLD = kld_Loss(mu, logvar)
    MSE = mse_loss(recon_x, x)
    return KLD + MSE




#%%
###########################################################################################################
#                                               model defination                                          #
###########################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms

class VAE(nn.Module):
    def __init__(self, zdim, input_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 20)
        self.fc21 = nn.Linear(20, zdim) 
        self.fc22 = nn.Linear(20, zdim) 
        self.fc3 = nn.Linear(zdim, 20)
        self.fc4 = nn.Linear(20, input_dim)
        self.input_dim = input_dim
    # encoder
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    # generating the latent layer values
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn(mu.size(0),mu.size(1)) # assume eps normally distributed ~N(0,1)
            z = mu+ eps*std
            return z
        
    # decoder
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    

#%%
def train(model, num_epochs = 1, batch_size = 64, learning_rate = 0.0002):
    model.train() #train mode
    torch.manual_seed(42)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    
    for epoch in range(num_epochs):
      for data in train_loader:  # load batch
          recon_mu_logvar = model(data) # recon_mu_logvar contains recon_x, mu, and logvar
          loss = loss_function(recon_mu_logvar, data, 5) # calculate loss
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
  
def generate(num_data_generated,latent_shape_0, latent_shape_1, VAE_model):
    result = []
    for i in range(num_data_generated):
      rinpt = torch.randn(latent_shape_0, latent_shape_1)
      with torch.no_grad():
        si = VAE_model.decode(rinpt).numpy()
      result.append(si)
    return result
#%%
###########################################################################################################
#                                              training starts                                            #
###########################################################################################################
batch_size = 256
latent_size = 2
input_size = 2

model = VAE(latent_size, input_size)
train(model, num_epochs = 600, batch_size = batch_size, learning_rate = 0.005)
#%%
num_data_generated = 2000
latent_shape_0 = 1
latent_shape_1 = 2
VAE_model = model
result = generate(num_data_generated,latent_shape_0, latent_shape_1, VAE_model)

# result = []
# for i in range(2000):
#   rinpt = torch.randn(1, 2)
#   with torch.no_grad():
#     si = model.decode(rinpt).numpy()
#   result.append(si)
  

#%%
testing_dataset = np.random.multivariate_normal(mean, cov, 3000)
testing_dataset = testing_dataset.astype(np.float32)


#%%
result_new = np.array(result)[:,0,:]
result_new.shape
#%%
import matplotlib.pyplot as plt

plt.scatter(result_new[:, 0],  result_new[:, 1])
plt.title("VAE generated points 2d gaussian")
plt.show()
plt.scatter(dataset[:, 0], dataset[:, 1], c="r")
plt.title("VAE label points 2d gaussian")
plt.show()

#%%
var_0 = result_new[:,0].var()
mean_0 = result_new[:,0].mean()
var_1 = result_new[:,1].var()
mean_1 = result_new[:,1].mean()

result_mean = np.array([mean_0, mean_1])
result_var = np.diag(np.array([var_0, var_1]))


#%%
def kl_divergence(mean1, var1, mean2, var2):
    return np.log(var2/var1) + (var1 ** 2 + (mean1 - mean2) ** 2) / (2 * var2 ** 2) - 1/2


def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 



#%%
Kl_mvn = kl_mvn(mean, cov, result_mean, result_var)