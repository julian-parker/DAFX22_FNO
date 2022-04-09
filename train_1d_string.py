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

num_variations = 1024
validation_split = 0.1

width = 16
epochs = 5000
batch_size = 350
device = 'cuda'

num_example_timesteps = 500

#######################################################################################################################
stringSolver = StringSolver(dur = dur, Fs = fs, delta_x = delta_x, d1 = d1)
training_input = torch.zeros((num_variations,1,stringSolver.numXs,2))
training_output = torch.zeros((num_variations,stringSolver.numT -1 ,stringSolver.numXs,2))
for i in range(num_variations):
    if (i < num_variations //2):
        pos = np.random.rand(1)
        fe_x = stringSolver.create_pluck(pos)
    else:
        fe_x = stringSolver.create_random_initial()
    y_x, y_defl_x = stringSolver.solve(fe_x)
    training_input[i,:,:,:] = torch.tensor(np.stack([y_x[:,0], y_defl_x[:,0]], axis = -1 )).unsqueeze(0)
    training_output[i,:,:,:] = torch.tensor(np.stack([y_x[:,1:].transpose(), y_defl_x[:,1:].transpose()], axis = -1 )).unsqueeze(0)
normalization_multiplier = 1/torch.std(training_input)
training_input *= normalization_multiplier
training_output *= normalization_multiplier

num_validation = np.int(np.ceil(validation_split * num_variations))
validation_input = training_input[-num_validation:,...]
validation_output = training_output[-num_validation:,...]
training_input = training_input[:-num_validation,...]
training_output = training_output[:-num_validation,...]
#######################################################################################################################


model_gru = FNO_GRU_1d   (in_channels = 2, out_channels = 2, spatial_size = training_output.shape[2], width = width).to(device)
model_rnn = FNO_RNN_1d   (in_channels = 2, out_channels = 2, spatial_size = training_output.shape[2], depth = 3, width = width).to(device)
model_ref = FNO_Markov_1d(in_channels = 2, out_channels = 2, spatial_size = training_output.shape[2], depth = 3, width = width).to(device)

learning_rate = 1e-4

params = list(model_gru.parameters()) + list(model_rnn.parameters()) + list(model_ref.parameters())
dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(training_input, training_output), batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, epochs=epochs, steps_per_epoch = len(dataloader))

loss_history = np.zeros((epochs,3))

for ep in range(epochs):
  tic = time.time()
  for input, output in dataloader:
    input, output = input.to(device), output.to(device)
    optimizer.zero_grad()
    model_input = input[:,0,:,:]
    pred_gru = model_gru(model_input, num_time_steps = training_output.shape[1])
    loss_gru = torch.log10(torch.nn.functional.mse_loss(pred_gru, output))
    loss_gru.backward()
    pred_rnn = model_rnn(model_input, num_time_steps = training_output.shape[1])
    loss_rnn = torch.log10(torch.nn.functional.mse_loss(pred_rnn, output))
    loss_rnn.backward()
    pred_ref = model_ref(model_input, num_time_steps = training_output.shape[1])
    loss_ref = torch.log10(torch.nn.functional.mse_loss(pred_ref, output))
    loss_ref.backward()
    torch.nn.utils.clip_grad_norm_(params, 0.5)
    optimizer.step()
    scheduler.step()
  loss_history[ep,0] = np.power(10,loss_gru.detach().cpu().numpy())
  loss_history[ep,1] = np.power(10,loss_rnn.detach().cpu().numpy())
  loss_history[ep,2] = np.power(10,loss_ref.detach().cpu().numpy())
  elapsed = time.time() - tic
  time_remaining = elapsed * (epochs - ep) / (60.0 * 60.0)
  print("\r", f"epochs:{ep}, gru_loss:{loss_history[ep,0]:.5f}, rnn_loss:{loss_history[ep,1]:.5f}, ref_loss:{loss_history[ep,2]:.5f}, epoch_time(s):{elapsed:.2f}, time_remaining(hrs):{time_remaining:.2f}", end = "")

#######################################################################################################################
now = datetime.now()
directory = "1d_string_" + now.strftime("%H_%M_%S")
os.mkdir(directory)
plt.plot(loss_history)
plt.savefig(directory + "/loss_history.pdf")

path = directory + "/model_gru.pt" 
torch.save(model_gru, path)
path = directory + "/model_rnn.pt" 
torch.save(model_rnn, path)
path = directory + "/model_ref.pt" 
torch.save(model_ref, path)

#######################################################################################################################
validation_input = validation_input.to(device)
validation_output = validation_output.to(device)

val_gru_out = model_gru(validation_input[:,0,...], validation_output.shape[1])
val_rnn_out = model_rnn(validation_input[:,0,...], validation_output.shape[1])
val_ref_out = model_ref(validation_input[:,0,...], validation_output.shape[1])

val_gru_mse = torch.nn.functional.mse_loss(val_gru_out, validation_output).detach().cpu().numpy()
val_rnn_mse = torch.nn.functional.mse_loss(val_rnn_out, validation_output).detach().cpu().numpy()
val_ref_mse = torch.nn.functional.mse_loss(val_ref_out, validation_output).detach().cpu().numpy()

with open(directory + "/validation.txt", 'w') as f:
    f.write(f"GRU validation MSE:{val_gru_mse:.8f} || RNN validation MSE:{val_rnn_mse:.8f} || Ref validation MSE:{val_ref_mse:.8f}")
    f.close()

#######################################################################################################################

dur = (num_example_timesteps+1)/fs
stringSolver = StringSolver(dur = dur, Fs = fs,delta_x = delta_x, d1 = d1)

fe_x = stringSolver.create_pluck(0.49)
y_x, y_defl_x = stringSolver.solve(fe_x)
model_input = torch.tensor(np.stack([y_x[:,0], y_defl_x[:,0]], axis = -1 )).unsqueeze(0).to(device)
model_input *= 1 / model_input.std()
y_x *= 1 / model_input.std().cpu().numpy()
output_sequence_gru = model_gru(model_input, num_example_timesteps)
output_sequence_rnn = model_rnn(model_input, num_example_timesteps)
output_sequence_ref = model_ref(model_input, num_example_timesteps)

fig = plt.figure()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"]})
gs = fig.add_gridspec(1, 4, hspace=0, wspace=0.05)
fig_width = 237/72.27 # Latex columnwidth expressed in inches
figsize = (fig_width, fig_width * 0.618)
axs = gs.subplots(sharex='row', sharey=True, figsize=figsize)
axs[0].imshow(output_sequence_gru[0,:,:,0].detach().cpu().numpy(),cmap = 'Greys', aspect = 'auto')
axs[1].imshow(output_sequence_rnn[0,:,:,0].detach().cpu().numpy(),cmap = 'Greys', aspect = 'auto')
axs[2].imshow(output_sequence_ref[0,:,:,0].detach().cpu().numpy(),cmap = 'Greys', aspect = 'auto')
axs[3].imshow(y_x[:,1:].transpose()                              ,cmap = 'Greys', aspect = 'auto')

axs[0].set(title = 'FGRU')
axs[1].set(title = 'FRNN')
axs[2].set(title = 'Ref')
axs[3].set(title = 'Ground Truth')

axs[0].set(ylabel = "t(samples)")
for ax in axs:
    ax.label_outer()
    ax.set_xticks([])
    
plt.savefig(directory + "/1d_string_outputs.pdf")