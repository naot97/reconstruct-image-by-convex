import numpy as np
import cv2
import argparse
import math
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.linalg as linalg
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from imageio import imread
import scipy.ndimage.filters as fi
from scipy.signal import convolve2d
from scipy.ndimage import imread
from timeit import default_timer
from PIL import Image, ImageOps
import scipy.io
from scipy import fftpack

np.random.seed(21)

class ROFImg:
    def __init__(self):
        self.mean = 0
        self.var = 500
        self.sigma = self.var ** 0.5
        self.M=512
        self.N=512 
        self.f = list()
        self.init_size()

    def init_size(self):
        L = np.zeros((self.M-1, self.N))
        i,j = np.indices(L.shape)
        L[i==j] = 1
        L[i==j-1] = -1

        self.Dx = sparse.kron(sparse.eye(self.M), L)
        self.Dy = sparse.kron(L, sparse.eye(self.N))

    def denoise_fourier(self,im_noise):
        start = default_timer()
        im_fft = fftpack.fft2(im_noise)

        keep_fraction = 0.1
        im_fft2 = im_fft.copy()
        r, c = im_fft2.shape
        im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
        im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
        im_new = fftpack.ifft2(im_fft2).real
        return im_new,default_timer() - start

    def median_filter(self,data, filter_size):
        start = default_timer()
        temp = []
        indexer = filter_size // 2
        data_final = []
        data_final = np.zeros((len(data),len(data[0])))
        for i in range(len(data)):

            for j in range(len(data[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(data[i + z - indexer][j + k - indexer])

                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
        return data_final,default_timer() - start

    def gaussian_kernel(self,k_len = 5, sigma = 3):
        d_mat = np.zeros((k_len, k_len))
        d_mat[k_len//2, k_len//2] = 1
        return fi.gaussian_filter(d_mat, sigma)

    def noise_generator (self,noise_type,image):
        if noise_type == "gauss":       
            gauss = np.random.normal(self.mean, self.sigma, image.shape) 
            gauss = gauss.reshape(image.shape)
            noisy = image + gauss
            return noisy.astype(int)
        elif noise_type == "s&p":
            s_vs_p = 0.5
            amount = 0.004
            out = image
            # Generate Salt '1' noise
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
            out[coords] = 255
            # Generate Pepper '0' noise
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
            out[coords] = 0
            return out.astype(int)
        elif noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy.astype(int)
        elif noise_type =="speckle":
            gauss = np.random.randn(image.shape)
            gauss = gauss.reshape(image.shape)        
            noisy = image + image * gauss
            return noisy.astype(int)
        else:
            return image.astype(int)
    def load_text(self):
        fname = './images/text.png'
        #print(np.mean(imread(fname), axis=-1))
        text = cv2.cvtColor(imread(fname), cv2.COLOR_BGRA2RGB) 
        
        return text.astype(int)

    def load_mona_image(self):  
        return self.get_image(self.fname)

    
    def get_simulate_data(self,orig):
        corrupted = self.noise_generator('gauss',orig)
        
        return corrupted

    def get_image(self, dir):
        im = imread(dir)
        if len(im.shape) == 3:
            im  = self.rgb2gray(im)
        
        self.M = self.N = min(im.shape)
        self.init_size()
        return im[:self.M,:self.N]

    def get_text_data(self):
        original = self.get_rgb(self.fname)
        text = self.load_text()
        corrupted = original + text
        return corrupted

    def toeplitz(self,b, n):
        m = len(b)
        T = np.zeros((n+m-1, n))
        for i in range(n+m-1):
            for j in range(n):
                if 0 <= i-j < m:
                    T[i,j] = b[i-j]
        return T

    def eval_mse(self,x,y):
        return np.mean( (x-y)**2 )
        
    def rgb2gray(self, img):
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    def show_figure(self,ori,damaged,out):
        f = plt.figure()
        f.add_subplot(1,3, 1 )
        plt.title('Original')
        plt.imshow(ori)
        f.add_subplot(1,3, 2 )
        plt.title('Damaged')
        plt.imshow(damaged)
        
        f.add_subplot(1,3, 3)
        plt.title('Reconstructed l = ' + str(self.l))
        plt.imshow(out)
        plt.show(block=True)
        
    def mean_filter(self,noisy):
        start = default_timer()
        kernel = np.ones((3,3),np.float32)/9
        processed_image = cv2.filter2D(noisy,-1,kernel)
        return processed_image,default_timer()-start

    def eval_pnsr(self,mse):
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def get_rgb(self,dir):
        return imread(dir)[:self.M,:self.N]

    def graph(self,f1,f2,f3,l1,l2,l3):
        plt.plot( range(1,len(f1) + 1) ,f1, color='red', linewidth=2,label='lambda = ' + str(l1))
        plt.plot( range(1,len(f2) + 1) ,f2, color='green', linewidth=2, linestyle='dashed',label='lambda = ' + str(l2))
        plt.plot( range(1,len(f3) + 1) ,f3, color='purple', linewidth=2,linestyle='dashed',label='lambda = ' + str(l3))
        plt.xlabel('Vòng lặp') 
        plt.ylabel('Hàm mục tiêu')
        plt.title('Đồ thị biểu thị giá trị hàm mục tiêu theo số vòng lặp')
        plt.legend()
        plt.show() 
        


