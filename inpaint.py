import numpy as np
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.linalg as linalg
from timeit import default_timer
from detectnoise import *
from utils import ROFImg

class Inpaint(ROFImg):
    def __init__(self):
        ROFImg.__init__(self)
    def inpainting_smoothed_sq(self,x, a,b, l=0.5):
        #print(l)
        return (linalg.norm(self.Dx.dot(x))**2 + linalg.norm(self.Dy.dot(x))**2) + 0.5 *l* a.dot(linalg.norm(b - x)**2)

    def inpainting_smoothed_sq_grad(self,x,a, b, l=0.5):
        return 2*(0.5*a*l*(x - b) + (self.Dx.T.dot(self.Dx.dot(x)) + self.Dy.T.dot(self.Dy.dot(x))))
    def denoise_smoothed_sq(self, x, b, l=1.1):
        return 0.5*linalg.norm(b - x)**2 + l*(linalg.norm(self.Dx.dot(x))**2 + linalg.norm(self.Dy.dot(x))**2)

    def denoise_smoothed_sq_grad(self, x, b, l=1.1):
        return 2*(0.5*(x - b) + l*(self.Dx.T.dot(self.Dx.dot(x)) + self.Dy.T.dot(self.Dy.dot(x))))
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
    def inpainting_simulate(self):
        #ori = self.get_rgb('./images/clean.bmp')
        ori = self.get_rgb(self.fname)
        damaged = self.get_simulate_data(ori,"s&p")
        #damaged = self.get_rgb('./images/noisy.bmp')
        rows0,cols0,rows1,cols1,rows2,cols2 = algorithm(3,damaged)
        
        out,t = self.denoising_rgb(damaged)

        noisy = damaged + 0
        noisy = damaged.astype(int)
        noisy[rows0, cols0,:] = 0
        # convex

        out1=damaged.copy()
        out1[rows0, cols0,:]= out[rows0, cols0,:]
        out1[rows1, cols1,:]= out[rows1, cols1,:]
        out1[rows2, cols2,:]= out[rows2, cols2,:]
        # median
        median,t = self.median_filter(damaged,3)
        median1=damaged.copy()
        median1[rows0, cols0,:]= median[rows0, cols0,:]
        median1[rows1, cols1,:]= median[rows1, cols1,:]
        median1[rows2, cols2,:]= median[rows2, cols2,:]
        print('noisy :',self.eval_mse(ori,damaged))
        print('median filter  :',self.eval_mse(ori,median))
        print('median filter + detect noise :', self.eval_mse(ori, median1) )
        print('convex :',self.eval_mse(ori,out))
        print('convex + detect noisy :',self.eval_mse(ori,out1))
        self.show_figure(ori,damaged,noisy,out,out1)

    def inpainting(self,noise, rows0, cols0, rows1, cols1, rows2, cols2):


        a = np.ones((self.M,self.N))
        a[rows0, cols0] = 0
        a = a.flatten()

        b = np.ones((self.M,self.N))
        b[rows1, cols1] = 0
        b = b.flatten()
        
        c = np.ones((self.M,self.N))
        c[rows2, cols2] = 0
        c = c.flatten()

        #print('a',rows0.shape)

        #A=[a,b,c]
        A=[a,a,a]


        image_result = np.zeros_like(noise)

        #t= noise+0  
        #t[rows,cols,:]=0  
        #return t  

        for i in range(3):
            b = noise[:,:,i].flatten()

        
            l=10

            optim_output = optimize.minimize(lambda x: self.inpainting_smoothed_sq(x,A[i],b,l),
                                    np.zeros(self.M * self.N),
                                    method='L-BFGS-B',
                                    jac=lambda x: self.inpainting_smoothed_sq_grad(x,A[i],b,l),
                                    options={'disp':False, 'ftol' : 1e-30},callback= lambda xk : self.f.append(self.inpainting_smoothed_sq(xk,a,b,l)[0]))

            image_smooth = optim_output['x']
            image_result[:,:,i] = image_smooth.reshape((self.N,)*2)
        
        return image_result.astype(int)
        

test = Inpaint()
test.fname = "./images/lana.jpg"
#test.fname = "./images/sieuam.png"

# #### code for inpating #####
lambdas = [1.2]
f = list()
tt=0
for l in lambdas:
    test.l = l
    test.inpainting_simulate()
    f.append(test.f)
    test.f = list()
