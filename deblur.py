import numpy as np
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.linalg as linalg
from timeit import default_timer
import scipy.io
from scipy import fftpack
from utils import ROFImg

class Deblur(ROFImg):
    def __init__(self):
        ROFImg.__init__(self)

    def deblurred_smoothed_sq(self,x, A, b, l=1e-25):
         return 0.5*linalg.norm(b - A.dot(x))**2 + l*(linalg.norm(self.Dx.dot(x))**2 + linalg.norm(self.Dy.dot(x))**2)
        
    def deblurred_smoothed_sq_grad(self,x, A, b, l=1e-25):
        return 2*(0.5*A.T.dot(A.dot(x) - b) +l*(self.Dx.T.dot(self.Dx.dot(x)) + self.Dy.T.dot(self.Dy.dot(x))))
    def deblurred(self,ori,noisy,kernel):
        l = self.l
        nitems = kernel.size
        
        curr_1d_kernel = kernel
        print(kernel.shape)
        ## Gaussian 1D kernel as matrix
        T = self.toeplitz(curr_1d_kernel, self.N)
        row_mat = sparse.kron(sparse.eye(self.N), T)
        col_mat = sparse.kron(T, sparse.eye(self.N+nitems-1))

        a = np.zeros((self.N + nitems -1,self.N + nitems -1))
        w = int(nitems / 2) - 2
        a[w:self.N + w,w:self.N + w] = noisy
        new_blurred = a
        b =  new_blurred.flatten()

        G = col_mat.dot(row_mat)
       
        optim_output = optimize.minimize(lambda x: self.deblurred_smoothed_sq(x,G,b,l),
                                         np.zeros(self.N**2),
                                         method='L-BFGS-B',
                                         jac=lambda x: self.deblurred_smoothed_sq_grad(x,G,b,l),
                                         options={'disp':True, 'ftol' : 1e-30},callback=lambda xk : self.f.append(self.deblurred_smoothed_sq(xk,b,l)))

    def deblurred_real(self):
        n = 2
        for i in range(n,n+1):
            ori = self.get_image("deblured_data/image_"+ str(i) +".jpg")
            noisy = self.get_image("deblured_data/blurry_"+ str(i) +".jpg")
            mat = scipy.io.loadmat('deblured_data/kernel_GT_1.mat')
            kernel = mat.get('kernel').flatten()
            self.deblurred(ori,noisy,kernel)

    def deblurred_simulate(self):
        l = self.l
        nitems, sigma = 9, 5
        curr_1d_kernel = self.gaussian1d(nitems, sigma)

        ## Gaussian 1D kernel as matrix
        T = self.toeplitz(curr_1d_kernel, 512)

        row_mat = sparse.kron(sparse.eye(self.N), T)
        col_mat = sparse.kron(T, sparse.eye(self.N+8))

        ori = self.get_image(self.fname)
        new_blurred = col_mat.dot(row_mat.dot(ori.flatten()))
        b =  new_blurred.flatten()

        G = col_mat.dot(row_mat)

        optim_output = optimize.minimize(lambda x: self.deblurred_smoothed_sq(x,G,b,l),
                                         np.zeros(self.N**2),
                                         method='L-BFGS-B',
                                         jac=lambda x: self.deblurred_smoothed_sq_grad(x,G,b,l),
                                         options={'disp':True, 'ftol' : 1e-30},callback=lambda xk : self.f.append(self.deblurred_smoothed_sq(xk,G,b,l)))

        out = np.rot90(optim_output['x'].reshape((self.M,self.N)),4)
        noise = new_blurred.reshape((self.N+8,)*2)
        self.show_figure(ori,noise,out)



test = Deblur()
test.fname = "images/lena.bmp"

##### code for deblured #####
lambdas = [1e-25,1e-15,1e-5]
f = list()
for l in lambdas:
    test.l = l
    test.deblurred_simulate()
    f.append(test.f)
    test.f = list()
test.graph(f[0],f[1],f[2],lambdas[0],lambdas[1],lambdas[2])