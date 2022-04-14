import numpy as np
import torch
from dafx22_fno.generators.string_solver import StringSolver
from dafx22_fno.modules.fno_rnn import FNO_RNN_1d
from dafx22_fno.modules.fno_gru import FNO_GRU_1d
from dafx22_fno.modules.fno_ref import FNO_Markov_1d
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

dur = 0.0025
fs = 48000
delta_x = 5e-3
d1 = 1e-1

num_variations = 4
###############################################################################################################

stringSolver = StringSolver(dur = dur, Fs = fs, delta_x = delta_x, d1 = d1)
training_input = torch.zeros((num_variations,1,stringSolver.numXs,2))
training_output = torch.zeros((num_variations,stringSolver.numT -1 ,stringSolver.numXs,2))
for i in range(num_variations):
    if (i < num_variations // 2):
        pos = np.random.rand(1)
        fe_x = stringSolver.create_pluck(pos)
    else:
        fe_x = stringSolver.create_random_initial()
    y_x, y_defl_x = stringSolver.solve(fe_x)
    training_input[i,:,:,:] = torch.tensor(np.stack([y_x[:,0], y_defl_x[:,0]], axis = -1 )).unsqueeze(0)
    training_output[i,:,:,:] = torch.tensor(np.stack([y_x[:,1:].transpose(), y_defl_x[:,1:].transpose()], axis = -1 )).unsqueeze(0)
normalization_multiplier = 1/torch.std(training_output, dim = (0,1,2))
training_input *= normalization_multiplier
training_output *= normalization_multiplier
max_val = training_output.abs().max()
training_output /= max_val
training_input /= max_val

fig_width = 237/72.27 # Latex columnwidth expressed in inches
figsize = (fig_width, 0.618*fig_width)
fig = plt.figure(figsize = figsize)
plt.rcParams.update({
    'axes.titlesize': 'small',
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,
    "font.serif": ["Times"]})
gs = fig.add_gridspec(2, 4, hspace=0.04, wspace=0.05, height_ratios=[1,2])

axs = gs.subplots(sharex=True, sharey='row')
axs[0,0].plot(training_input[0,0,:,0], linewidth = 0.7)
axs[0,0].plot(training_input[0,0,:,1], linewidth = 0.7)
axs[0,1].plot(training_input[1,0,:,0], linewidth = 0.7)
axs[0,1].plot(training_input[1,0,:,1], linewidth = 0.7)
axs[0,2].plot(training_input[2,0,:,0], linewidth = 0.7)
axs[0,2].plot(training_input[2,0,:,1], linewidth = 0.7)
axs[0,3].plot(training_input[3,0,:,0], linewidth = 0.7)
axs[0,3].plot(training_input[3,0,:,1], linewidth = 0.7)
axs[1,0].imshow(training_output[0,:,:,0].detach().cpu().numpy(),cmap = 'viridis', aspect = 'auto', interpolation = 'none')
axs[1,1].imshow(training_output[1,:,:,0].detach().cpu().numpy(),cmap = 'viridis', aspect = 'auto', interpolation = 'none')
axs[1,2].imshow(training_output[2,:,:,0].detach().cpu().numpy(),cmap = 'viridis', aspect = 'auto', interpolation = 'none')
pos=axs[1,3].imshow(training_output[3,:,:,0].detach().cpu().numpy(),cmap = 'viridis', aspect = 'auto', interpolation = 'none')

cbar_min = training_output.min()
cbar_max = training_output.max()
cbar = fig.colorbar(pos, ax=axs.ravel().tolist())

axs[0,0].set_yticks([])
axs[0,0].set_yticklabels([])

#axs[0].set(title = 'FGRU')
#axs[1].set(title = 'FRNN')
#axs[2].set(title = 'Ref')
#axs[3].set(title = 'Truth')

axs[1,0].set(ylabel = " $\leftarrow$ t (/s)")
axs[1,0].set(xlabel = " x (/m)")
for i in range(len(axs)):
    for j in range(len(axs[0])):
        axs[i,j].label_outer()
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
for j in range(len(axs[0])):
    axs[1,j].get_images()[0].set_clim(-1, 1)
    
cbar.set_ticks([-1,0,1])
cbar.set_ticklabels(["low","0","high"])
    
plt.savefig("1d_string_examples.pdf",bbox_inches='tight')