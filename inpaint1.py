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
    def inpainting_smoothed_sq(self,x, a,b, l=0.5):
        print(l)
        return (linalg.norm(self.Dx.dot(x))**2 + linalg.norm(self.Dy.dot(x))**2) + 0.5 *l* a.dot(linalg.norm(b - x)**2)

    def inpainting_smoothed_sq_grad(self,x,a, b, l=0.5):
        return 2*(0.5*a*l*(x - b) + (self.Dx.T.dot(self.Dx.dot(x)) + self.Dy.T.dot(self.Dy.dot(x))))
    
    def inpainting_simulate(self):
        ori = self.get_rgb(self.fname)
        #self.M = ori.shape[0]
        #self.N = ori.shape[1]

        noise = self.get_simulate_data(ori,"gauss")
        #noise = ori
        #mean = 0.0   # some constant
        #std = 1.0    # some constant (standard deviation)
        #noise = ori + np.random.normal(mean, std, ori.shape)
        #noise = np.clip(noise, 0, 255)  
        #rows, cols = np.where((noise[:,:,0] == ori[:,:,0]) & (noise[:,:,1] == ori[:,:,1]) & (noise[:,:,2] == ori[:,:,2]))
        rows0,cols0,rows1,cols1,rows2,cols2 = algorithm(5,noise)
        #rows0=cols0=rows1=cols1=rows2=cols2 = np.array([])
        #print(rows.shape,cols.shape)
        out = self.inpainting(noise,rows0,cols0,rows1,cols1,rows2,cols2)
        print(self.eval_mse(ori,noise))
        print(self.eval_mse(ori,out))
        self.show_figure(ori,noise,out,noise)

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

        A=[a,b,c]


        image_result = np.zeros_like(noise)

        #t= noise+0  
        #t[rows,cols,:]=0  
        #return t  

        for i in range(3):
            b = noise[:,:,i].flatten()

        
            l=8

            optim_output = optimize.minimize(lambda x: self.inpainting_smoothed_sq(x,A[i],b,l),
                                    np.zeros(self.M * self.N),
                                    method='L-BFGS-B',
                                    jac=lambda x: self.inpainting_smoothed_sq_grad(x,a,b,l),
                                    options={'disp':True, 'ftol' : 1e-30},callback= lambda xk : self.f.append(self.inpainting_smoothed_sq(xk,a,b,l)[0]))

            image_smooth = optim_output['x']
            image_result[:,:,i] = image_smooth.reshape((self.N,)*2)
        
        return image_result
        

test = Inpaint()
test.fname = "./images/ivanka.jpg"

# #### code for inpating #####
lambdas = [1.2]
f = list()
tt=0
for l in lambdas:
    tt+=1
    test.l = l
    test.inpainting_simulate()
    f.append(test.f)
    test.f = list()
test.graph(f[0],f[1],f[2],lambdas[0],lambdas[1],lambdas[2])
print("##",tt)
