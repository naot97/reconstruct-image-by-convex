import numpy as np
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.linalg as linalg
from timeit import default_timer
from utils import ROFImg
from detectnoise import *

class Inpaint(ROFImg):
    def __init__(self):
        ROFImg.__init__(self)
    def inpainting_smoothed_sq(self,x, a,b, l=1.1):
        return l*(linalg.norm(self.Dx.dot(x))**2 + linalg.norm(self.Dy.dot(x))**2) + 0.5 * a.dot(linalg.norm(b - x)**2)

    def inpainting_smoothed_sq_grad(self,x,a, b, l=1.1):
        return 2*(0.5*a*(x - b) + l*(self.Dx.T.dot(self.Dx.dot(x)) + self.Dy.T.dot(self.Dy.dot(x))))
    
    def inpainting_simulate(self):
        ori = self.get_rgb(self.fname)
        noise = self.get_text_data()
        #rows, cols = np.where((noise[:,:,0] == ori[:,:,0]) & (noise[:,:,1] == ori[:,:,1]) & (noise[:,:,2] == ori[:,:,2]))
        noise_index = algorithm(5,ori)
        print(noise_index.shape)
        rows = noise_index[:,0]
        cols = noise_index[:,1]
        out = self.inpainting(noise,rows,cols)
        self.show_figure(ori,noise,out)

    def inpainting(self,noise, rows, cols):
        a = np.zeros((self.M,self.N))

        a[rows, cols] = 1
        a = a.flatten()
        image_result = np.zeros_like(noise)

        for i in range(3):
            b = noise[:,:,i].flatten()

            l = self.l

            optim_output = optimize.minimize(lambda x: self.inpainting_smoothed_sq(x,a,b,l),
                                    np.zeros(self.M * self.N),
                                    method='L-BFGS-B',
                                    jac=lambda x: self.inpainting_smoothed_sq_grad(x,a,b,l),
                                    options={'disp':True, 'ftol' : 1e-30},callback= lambda xk : self.f.append(self.inpainting_smoothed_sq(xk,a,b,l)[0]))

            image_smooth = optim_output['x']
            image_result[:,:,i] = image_smooth.reshape((self.N,)*2)
        
        return image_result

test = Inpaint()
test.fname = "./images/monalisa.png"

# #### code for inpating #####
lambdas = [1.2]
f = list()
for l in lambdas:
    test.l = l
    test.inpainting_simulate()
    f.append(test.f)
    test.f = list()
test.graph(f[0],f[1],f[2],lambdas[0],lambdas[1],lambdas[2])