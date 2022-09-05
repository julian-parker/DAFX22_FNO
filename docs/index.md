<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
		tex2jax: {
			inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
  }
});
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<!-- ... -->

<link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous" />
<link rel="stylesheet" href="{{ site.baseurl}}/css/trackswitch.min.css" />




    
# Supplemental Material for Physical Modeling using Recurrent Neural Networks with Fast Convolutional Layers


## Example 1 - Linear String

<video width="640" height="480" controls>
  <source src="{{ site.baseurl}}/examples/videos/1d_string_anim.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video> 
 
### Linear string GRU
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/string_gru.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/string_gru.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>

### Linear string Real
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/string_real.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/string_real.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>

### Linear string Reference
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/string_ref.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/string_ref.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>

### Linear string RNN
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/string_rnn.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/string_rnn.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>




## Example 2 - Nonlinear String

<video width="640" height="480" controls>
  <source src="{{ site.baseurl}}/examples/videos/1d_nonlinear_string_anim.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video> 


### Nonlinear string GRU
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_gru.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_gru.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>

### Nonlinear string Real
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_real.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_real.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>

### Nonlinear string Reference
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_ref.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_ref.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>

### Nonlinear string RNN
<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_rnn.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_rnn.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>


## Example 3 - 2D Wave Equation

<video width="640" height="480" controls>
  <source src="{{ site.baseurl}}/examples/videos/2d_wave_anim.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video> 


## Physical parameters used for dataset generation

### Linear String

| Quantity | Value |
|:--------:|:-----:|
|$$E$$ | 5.4 e9|              
|$\rho_s$ | 1140|             
|$l$ | 0.65|              
|$A$ | 0.5188 e-6|  
|$I$ | 0.171 e-12|
|$d_1$ |8 e-1|
|$d_3$ |1.4 e-5|
|$T_s$ |60.97|
|$\nu$ |50|   

### 2d wave equation

| Quantity | Value |
|:--------:|:-----:|
|$l_x$ | 1 |
|$l_y$ | 0.95 |
|$c_0$ | 340 |
|$\rho_o$ | 1.2041|

### Nonlinear Tension Modulated String

| Quantity | Value |
|:--------:|:-----:|
|$l$ | 0.65       |
|$A$   | 0.5188e-6  |
|$I$   | 0.171e-12  |
|$\rho$ | 1140       |
|$E$   | 5.4e9      |
|$d_1$  | 1e-2       |
|$d_3$  | 6e-5       |
|$T_{s0}$ | 60.97      |

## Network/Training Hyperparameters

Visible in:
`train_1d_string.py` 
`train_2d_wave.py`  
`train_1d_nonlinear_string.py`

