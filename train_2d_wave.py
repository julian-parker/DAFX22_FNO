import numpy as np
import torch
import sys
import time
from dafx22_fno.generators.wave_solver_2d import WaveSolver2D
from dafx22_fno.modules.fno_rnn import FNO_RNN_2d
from dafx22_fno.modules.fno_gru import FNO_GRU_2d
from dafx22_fno.modules.fno_ref import FNO_Markov_2d
import matplotlib.pyplot as plt

dur = 0.0007
num_variations = 1024
validation_split = 0.1

fs = 48000
num_points = 40
simulated_modes = 25
room_size = 1


if(len(sys.argv) == 1):
    epochs = 5000
else:
    epochs = int(sys.argv[1])
print("\r",f"Starting training for {epochs} epochs", end = "")

width = 16
device = 'cuda'
batch_size = 400

num_example_timesteps = 100

#######################################################################################################################
solver = WaveSolver2D(dur = dur, Fs = fs, lx = room_size, ly = room_size, spatial_delta = room_size/num_points, modes = simulated_modes)

training_input = torch.zeros((num_variations,1,solver.numXs,solver.numYs,3))
training_output = torch.zeros((num_variations,solver.numT -1 ,solver.numXs,solver.numYs,3))
for i in range(num_variations):
    if (i < num_variations //2):
        pos_x = np.random.rand(1)
        pos_y = np.random.rand(1)
        fe_x = solver.create_impulse(pos_x,pos_y)
    else:
        fe_x = solver.create_random_initial()
    _,_,y_sp, y_vx, y_vy = solver.solve(fe_x)
    training_input[i,:,:,:,:] = torch.tensor(np.stack([y_sp[:,:,0], y_vx[:,:,0], y_vy[:,:,0]], axis = -1 )).unsqueeze(0)
    training_output[i,:,:,:,:] = torch.tensor(np.stack([y_sp[:,:,1:].transpose(2,0,1), y_vx[:,:,1:].transpose(2,0,1), y_vy[:,:,1:].transpose(2,0,1)], axis = -1 )).unsqueeze(0)
    
normalization_multiplier = 1/torch.std(training_output, dim = (0,1,2,3))
training_input *= normalization_multiplier
training_output *= normalization_multiplier

num_validation = np.int(np.ceil(validation_split * num_variations))
validation_input = training_input[-num_validation:,...]
validation_output = training_output[-num_validation:,...]
training_input = training_input[:-num_validation,...]
training_output = training_output[:-num_validation,...]

learning_rate = 1e-4

model_gru = torch.nn.DataParallel(FNO_GRU_2d   (in_channels = 3, out_channels = 3, spatial_size_x = training_output.shape[2], spatial_size_y = training_output.shape[3], width = width)).to(device)
model_rnn = torch.nn.DataParallel(FNO_RNN_2d   (in_channels = 3, out_channels = 3, spatial_size_x = training_output.shape[2], spatial_size_y = training_output.shape[3], depth = 3, width = width)).to(device)
model_ref = torch.nn.DataParallel(FNO_Markov_2d(in_channels = 3, out_channels = 3, spatial_size_x = training_output.shape[2], spatial_size_y = training_output.shape[3], depth = 3, width = width)).to(device)

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
    model_input = input[:,0,...]
    pred_gru = model_gru(model_input, num_time_steps = training_output.shape[1])
    loss_gru = torch.log10(torch.nn.functional.mse_loss(pred_gru, output))
    loss_gru.backward()
    del pred_gru
    pred_rnn = model_rnn(model_input, num_time_steps = training_output.shape[1])
    loss_rnn = torch.log10(torch.nn.functional.mse_loss(pred_rnn, output))
    loss_rnn.backward()
    del pred_rnn
    pred_ref = model_ref(model_input, num_time_steps = training_output.shape[1])
    loss_ref = torch.log10(torch.nn.functional.mse_loss(pred_ref, output))
    loss_ref.backward()
    del pred_ref
    torch.nn.utils.clip_grad_norm_(params, 0.5)
    optimizer.step()
    scheduler.step()
  loss_history[ep,0] = np.power(10,loss_gru.detach().cpu().numpy())
  loss_history[ep,1] = np.power(10,loss_rnn.detach().cpu().numpy())
  loss_history[ep,2] = np.power(10,loss_ref.detach().cpu().numpy())
  elapsed = time.time() - tic
  time_remaining = elapsed * (epochs - ep) / (60.0 * 60.0)
  print("\r", f"epochs:{ep}, gru_loss:{loss_history[ep,0]:.5f}, rnn_loss:{loss_history[ep,1]:.5f}, ref_loss:{loss_history[ep,2]:.5f}, epoch_time(s):{elapsed:.2f}, time_remaining(hrs):{time_remaining:.2f}", end = "")

from datetime import datetime
import os
now = datetime.now()
directory = "2d_wave_" + now.strftime("%H_%M_%S")
os.mkdir(directory)
plt.plot(loss_history)
plt.savefig(directory + "/loss_history.pdf")

path = directory + "/model_gru.pt" 
torch.save(model_gru, path)
path = directory + "/model_rnn.pt" 
torch.save(model_rnn, path)
path = directory + "/model_ref.pt" 
torch.save(model_ref, path)
path = directory + "/norms.pt"
torch.save(normalization_multiplier, path)

del input
del output
del dataloader
del optimizer
del params
torch.cuda.empty_cache()
#######################################################################################################################
validation_input = validation_input.to(device)
validation_output = validation_output.to(device)

val_gru_out = model_gru(validation_input[:,0,...], validation_output.shape[1])
val_gru_mse = torch.nn.functional.mse_loss(val_gru_out, validation_output).detach().cpu().numpy()
del val_gru_out
val_rnn_out = model_rnn(validation_input[:,0,...], validation_output.shape[1])
val_rnn_mse = torch.nn.functional.mse_loss(val_rnn_out, validation_output).detach().cpu().numpy()
del val_rnn_out
val_ref_out = model_ref(validation_input[:,0,...], validation_output.shape[1])
val_ref_mse = torch.nn.functional.mse_loss(val_ref_out, validation_output).detach().cpu().numpy()
del val_ref_out

with open(directory + "/validation.txt", 'w') as f:
    f.write(f"GRU validation MSE:{val_gru_mse:.8f} || RNN validation MSE:{val_rnn_mse:.8f} || Ref validation MSE:{val_ref_mse:.8f}")
    f.close()

#######################################################################################################################
display_timestep = num_example_timesteps -1

dur = (num_example_timesteps+1)/fs
solver = WaveSolver2D(dur = dur, Fs = fs, lx = room_size, ly = room_size, spatial_delta = room_size/num_points, modes = simulated_modes)

fe_x = solver.create_impulse(0.5,0.1)
_,_,y_x, y_vx, y_vy = solver.solve(fe_x)
model_input = torch.tensor(np.stack([y_x[:,:,0], y_vx[:,:,0], y_vy[:,:,0]], axis = -1 )).unsqueeze(0).to(device)
model_input *= normalization_multiplier.to(device)
y_x *= normalization_multiplier[0].cpu().numpy()
output_sequence_gru = model_gru(model_input, num_example_timesteps)
output_sequence_rnn = model_rnn(model_input, num_example_timesteps)
output_sequence_ref = model_ref(model_input, num_example_timesteps)

plot_norm = 1/np.max(np.abs(y_x[:,:,10:]))
output_sequence_gru *= plot_norm
output_sequence_rnn *= plot_norm
output_sequence_ref *= plot_norm
y_x *= plot_norm

fig_width = 237/72.27 # Latex columnwidth expressed in inches
figsize = (fig_width, fig_width * 0.75)
fig = plt.figure(figsize = figsize)
plt.rcParams.update({
    'axes.titlesize': 'small',
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "font.serif": ["Times"]})
gs = fig.add_gridspec(3, 4, hspace=0.0, wspace=0.05)
axs = gs.subplots(sharex='row', sharey=True)
axs[0,0].imshow(output_sequence_gru[0,0,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[0,1].imshow(output_sequence_rnn[0,0,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[0,2].imshow(output_sequence_ref[0,0,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[0,3].imshow(y_x[...,1].transpose()                                           ,cmap = 'viridis', aspect = 'equal')
axs[1,0].imshow(output_sequence_gru[0,display_timestep // 2,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[1,1].imshow(output_sequence_rnn[0,display_timestep // 2,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[1,2].imshow(output_sequence_ref[0,display_timestep // 2,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[1,3].imshow(y_x[...,display_timestep // 2 + 1].transpose()                                       ,cmap = 'viridis', aspect = 'equal')
axs[2,0].imshow(output_sequence_gru[0,display_timestep,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[2,1].imshow(output_sequence_rnn[0,display_timestep,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[2,2].imshow(output_sequence_ref[0,display_timestep,:,:,0].detach().cpu().numpy().transpose(),cmap = 'viridis', aspect = 'equal')
axs[2,3].imshow(y_x[...,display_timestep + 1].transpose()                                        ,cmap = 'viridis', aspect = 'equal')

axs[0,0].set(title = 'FGRU')
axs[0,1].set(title = 'FRNN')
axs[0,2].set(title = 'Ref.')
axs[0,3].set(title = 'Truth')

axs[0,0].set(ylabel = "t = 0")
axs[1,0].set(ylabel = f"t = {display_timestep // 2}")
axs[2,0].set(ylabel = f"t = {display_timestep}")

for i in range(len(axs)):
    for j in range(len(axs[0])):
        axs[i,j].get_images()[0].set_clim(-1, 1)
        axs[i,j].label_outer()
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

plt.savefig(directory + "/2d_wave_outputs.pdf",bbox_inches='tight')