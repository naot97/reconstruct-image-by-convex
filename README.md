# Reconstructing RGB image by Convex optimization method

Use total variation method to create convex models for denoising, debluring, inpainting RGB images. 
#The paper for this project can be found in [here](https://link.springer.com/chapter/10.1007/978-981-33-4370-2_26)

## Getting Started
### Installing
* Install python >= 3.6.2
### Running
* Type ```python denoise.py``` in cmd or terminal to denoising.
* Type ```python deblur.py``` in cmd or terminal to debluring.
* Type ```python inpaint.py``` in cmd or terminal to inpainting.
### Testing
* folder ```denoising_benchmark_data``` consists data to benchmark for denoisinng.
* function ```denoising_real()``` in file ```denoise.py``` is used to benchmark this method with mean filter, median filter and threshold in Fourier domain.
