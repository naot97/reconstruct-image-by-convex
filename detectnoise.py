import cv2
import numpy as np
from imageio import imread,imwrite

#def caculateUa(central,img):
m = 40

def caculateT1(windowsize,lef,img,cp):
    n=int(windowsize/2)
    sum1 = 0
    cent=(lef[0]+n,lef[1]+n)
    #print("lef",lef)
    #print("cent",cent)
    #print('A',img)
    for i in range(windowsize):
        for j in range(windowsize):
            if lef[0]+i ==cent[0] and lef[1]+j==cent[1]:
                continue
            else:
                sum1+=img[lef[0]+i,lef[1]+j]
                #print("x",i,"y",j)
                #print("s",sum1)
   
    Ua=sum1/(windowsize*windowsize-1)
    #print("sum1",sum1)
    #print("u1",Ua)
    sum2 = 0

    #print("---------")
    for i in range(windowsize):
        for j in range(windowsize):
            if lef[0]+i ==cent[0] and lef[1]+j==cent[1]:
                continue
            else: 
                sum2+=(img[lef[0]+i][lef[1]+j] - Ua)*(img[lef[0]+i][lef[1]+j] - Ua)
    T1 = Ua+np.sqrt(sum2/(windowsize*windowsize-1)) 
    #print("sum2",sum2)
    #print("---------")

    sum3=0
    for i in range(windowsize):
        for j in range(windowsize):
                sum3+= abs(img[lef[0]+i][lef[1]+j] - img[cent[0]][cent[1]])
    NS=sum3/(windowsize*windowsize-1)
    #print("sum3",sum3)
    #print("NS",NS)
   
    return  True if NS >= T1 and (cp<=m or cp>=255-m)else False 
    
    
def algorithm(windowsize,img):          
    h, w = img.shape[:2]
    n=int(windowsize/2)

    print("h",h)
    print("w",w)
    print("n",n)


    #img_p = np.zeros([h+2*padding, w+2*padding])
    #img_p[padding:padding+h, padding:padding+w] = img
    #for i in range(w):
     #   for j in range(h):
    sumnoise=0
    rows = np.array([])
    cols =  np.array([])
    for i in range(h-windowsize+1):
        for j in range(w-windowsize+1):
            if caculateT1(windowsize,(i,j),img[:,:,0],img[i,j,0]) and caculateT1(windowsize,(i,j),img[:,:,1],img[i,j,1]) and caculateT1(windowsize,(i,j),img[:,:,2],img[i,j,2]):
                #noise_index = np.append(noise_index,[[i + (windowsize/2),j + (windowsize/2)]])
                rows = np.append(rows,[i + (windowsize/2)])
                cols = np.append(cols,[j + (windowsize/2)])
                sumnoise+=1
    return (rows,cols)     
    			
            

def apply_on_3_channels(img):
    layer_blue = algorithm(5,img[:,:,0])
    layer_green = algorithm(5,img[:,:,1])
    layer_red = algorithm(5,img[:,:,2])
    #new_img = np.zeros(list(layer_blue.shape) + [3])
    #new_img[:,:,0], new_img[:,:,1], new_img[:,:,2] = layer_blue, layer_green, layer_red
    #layer_blue = algorithm(5,img)
    print("***")
    return img

if __name__ == "__main__":
    img = imread("./images/monalisa.png")
    A = np.array([[1,2,3,4], 
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]])    
    print('Shape test.png:', img.shape)


    #new_img = apply_on_3_channels(img)'
    new_img = algorithm(5,img)
    
    #imwrite('./images/test_saved.jpg', new_img)
    #print("a",A[1][0])
    #print('Shape mona_new.png:', new_img.shape)
    print('Saved new image @ mona_new.png')

    #print('------------')
    
    #lighten_blur_img = apply_sliding_window_on_3_channels(img, kernel=[[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]], padding=1, stride=1)
    #imwrite('./images/monalisa_lighten_blur.png', lighten_blur_img)
    #print('Shape mona.png:', img.shape)
    #print('Shape mona_lighten_blur.png:', lighten_blur_img.shape)
    #print('Saved new image @ mona_lighten_blur.png')

