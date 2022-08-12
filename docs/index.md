<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
		tex2jax: {
			inlineMath: [ ["\\(","\\)"] ],
      processEscapes: true
  }
});
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<!-- ... -->

<link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous" />
<link rel="stylesheet" href="{{ site.baseurl}}/css/trackswitch.min.css" />




    
# Supplemental Material for Physical Modeling using Recurrent Neural Networks with Fast Convolutional Layers

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

## Network Hyperparameters


## Sound Examples

<audio controls>
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_gru.wav" type="audio/ogg">
  <source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_gru.wav" type="audio/mpeg">
  Your browser does not support the audio tag. 
</audio>


nonlinear_string_real

nonlinear_string_ref

nonlinear_string_rnn

string_gru
string_real
string_ref
string_rnn

## Animations

1d_string_anim

<video width="320" height="240" controls>
  <source src="{{ site.baseurl}}/examples/videos/1d_string_anim.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video> 



<div class="player">
  <p>
      Example trackswitch.js instance.
  </p>
  <ts-track title="Drums">
      <ts-source src="{{ site.baseurl}}/examples/sounds/nonlinear_string_gru.wav" type="audio/mpeg"></ts-source>
  </ts-track>
</div>




### Credits

[Trackswitch.js](https://audiolabs.github.io/trackswitch.js/) was developed by Nils Werner, Stefan Balke, Fabian-Rober Stöter, Meinard Müller and Bernd Edler. 

<script src="https://cdn.rawgit.com/download/polymer-cdn/1.5.0/lib/webcomponentsjs/webcomponents-lite.min.js"></script>
<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script src="{{ site.baseurl}}/js/trackswitch.js"></script>
<script type="text/javascript">
	var $j = jQuery.noConflict();
    $j(document).ready(function() {
        // $j(".customplayer").trackswitch({ onlyradiosolo: true, repeat: true });
        $j(".player").trackSwitch({ onlyradiosolo: true, repeat: true });
    });
</script>	

