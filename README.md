This is an experimental positional embeddings approach based on Fourier shifted signals to replace transformer attention layer.

I found that Square signal representation (inspired from biological spiking neurons) perform well.

The next token prediction, in this approach, is the max frequency of the fft of output predicted signal. Instead of the max probability of output softmax.
