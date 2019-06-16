import numpy as np
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.linalg as linalg
from timeit import default_timer
from utils import ROFImg


class Denoise(ROFImg):
    def __init__(self):
        ROFImg.__init__(self)
    def denoise_smoothed_sq(self, x, b, l=1.1):
        return 0.5*linalg.norm(b - x)**2 + l*(linalg.norm(self.Dx.dot(x))**2 + linalg.norm(self.Dy.dot(x))**2)

    def denoise_smoothed_sq_grad(self, x, b, l=1.1):
        return 2*(0.5*(x - b) + l*(self.Dx.T.dot(self.Dx.dot(x)) + self.Dy.T.dot(self.Dy.dot(x))))

    def denoising(self, noise ):
        start = default_timer()
        noise = np.rot90(noise,4)
        b = noise.flatten()
        l = self.l
        optim_output = optimize.minimize(lambda x: self.denoise_smoothed_sq(x,b,l),
                                    np.zeros(self.M * self.N),
                                    method='L-BFGS-B',
                                    jac=lambda x: self.denoise_smoothed_sq_grad(x,b,l),
                                    options={'disp':True,'ftol' : 1e-30 },callback=lambda xk : self.f.append(self.denoise_smoothed_sq(xk,b,l)))

        out = np.rot90(optim_output['x'].reshape((self.M,self.N)),4)
        
        return out,default_timer()-start
        
    def denoising_simulate(self):
        #ori = self.get_image(self.fname)
        #noise = self.get_simulate_data(ori)
        #out,t1 = self.denoising(noise)
        ori = self.get_rgb(self.fname)
        noise = self.get_simulate_data(ori)
        out,t1 = self.denoising_rgb(noise)
        print('mse noisy :', self.eval_mse(ori,noise))
        print('mse denoising :', self.eval_mse(ori,out))
        self.show_figure(ori,noise,out.astype(int))
        
    def denoising_rgb(self,noise):
        start = default_timer()
        out = np.zeros_like(noise)
        for i in range(3):
            b = noise[:,:,i].flatten()
            l = self.l
            optim_output = optimize.minimize(lambda x: self.denoise_smoothed_sq(x,b,l),
                                    np.zeros(self.M * self.N),
                                    method='L-BFGS-B',
                                    jac=lambda x: self.denoise_smoothed_sq_grad(x,b,l),
                                    options={'disp':True,'ftol' : 1e-30, 'maxiter' : 150 },callback=lambda xk : self.f.append(self.denoise_smoothed_sq(xk,b,l)))

            out[:,:,i] = np.rot90(optim_output['x'].reshape((self.M,self.N)),4)
        
        return out,default_timer()-start
    def denoising_real_rgb(self):
        n = 16
        self.mse_noise = 0
        self.mse_denoise = 0
        self.time_denoise = 0
        
        for i in range(1,n+1):
            ori = self.get_rgb("denoising_benchmark_data/batch"+str(i)+"/clean.bmp")
            noisy = self.get_rgb("denoising_benchmark_data/batch"+str(i)+"/noisy.bmp")
            out,t1 = self.denoising_rgb(noisy)

            self.time_denoise = self.time_denoise + t1
            self.mse_noise = self.mse_noise + self.eval_mse(noisy,ori)
            self.mse_denoise = self.mse_denoise + self.eval_mse(out,ori)
            
        print( 'MSE of noise :',  self.mse_noise/n )
        print( 'MSE of ROF model :', self.mse_denoise/n )
        print( 'PSNR of noise :',  self.eval_pnsr(self.mse_noise/n))
        print( 'PSNR of ROF model :', self.eval_pnsr(self.mse_denoise/n) )
        print( 'Time of ROF model :', self.time_denoise/n )
        
        self.show_figure(ori,noisy,out)
    def denoising_real(self):
        n = 16
        self.mse_noise = 0
        self.mse_denoise = 0
        self.mse_median = 0
        self.mse_mean = 0
        self.mse_fourier = 0
        
        self.time_denoise = 0
        self.time_median = 0
        self.time_mean = 0
        self.time_fourier = 0
        for i in range(n,n+1):
            ori = self.get_image("denoising_benchmark_data/batch"+str(i)+"/clean.bmp")
            noisy = self.get_image("denoising_benchmark_data/batch"+str(i)+"/noisy.bmp")
            out,t1 = self.denoising(ori,noisy)
            out_median,t2 = self.median_filter(noisy,3)
            out_mean,t3 = self.mean_filter(noisy)
            out_fourier,t4 = self.denoise_fourier(noisy)

            self.time_denoise = self.time_denoise + t1
            self.time_median = self.time_median + t2
            self.time_mean = self.time_mean + t3
            self.time_fourier = self.time_fourier + t4

            self.mse_noise = self.mse_noise + self.eval_mse(noisy,ori)
            self.mse_denoise = self.mse_denoise + self.eval_mse(out,ori)
            self.mse_median = self.mse_median + self.eval_mse(out_median,ori)
            self.mse_mean = self.mse_mean + self.eval_mse(out_mean,ori)
            self.mse_fourier = self.mse_fourier + self.eval_mse(out_fourier,ori)

        print( 'MSE of noise :',  self.mse_noise/n )
        print( 'MSE of ROF model :', self.mse_denoise/n )
        print( 'MSE of median filter', self.mse_median/n )
        print( 'MSE of mean filter', self.mse_mean/n )
        print( 'MSE of fourier filter', self.mse_fourier/n )

        print( 'PSNR of noise :',  self.eval_pnsr(self.mse_noise/n))
        print( 'PSNR of ROF model :', self.eval_pnsr(self.mse_denoise/n) )
        print( 'PSNR of median filter', self.eval_pnsr(self.mse_median/n) )
        print( 'PSNR of mean filter', self.eval_pnsr(self.mse_mean/n) )
        print( 'PSNR of fourier filter', self.eval_pnsr(self.mse_fourier/n) )

        print( 'Time of ROF model :', self.time_denoise/n )
        print( 'Time of median filter', self.time_median/n )
        print( 'Time of mean filter', self.time_mean/n )
        print( 'Time of fourier filter', self.time_fourier/n )

        self.show_figure(ori,noisy,out)

test = Denoise()
test.fname = "images/lena.bmp"

##### code for denoise #####
lambdas = [1.2]
f = list()
for l in lambdas:
    test.l = l
    test.denoising_simulate()
    f.append(test.f)
    test.f = list()
#test.graph(f[0],f[1],f[2],lambdas[0],lambdas[1],lambdas[2])