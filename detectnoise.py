import cv2
import numpy as np
from imageio import imread,imwrite
#xlnntn châp6 bài 1,3 phương pháp ppmi
#knowlege representation and reasoning( cac chuong lq btl 7,11,12)
#def caculateUa(central,img):
m = 40

def detect(im,lef,windowsize):
    n = int(windowsize/2)
    cent=(n,n)

    #chuyen 2 chieu thanh 1 chieu
    arr = im[ lef[0]:lef[0] + windowsize ,lef[1]:lef[1] + windowsize ]
    cp = arr[cent[0], cent[1]]
    arr = arr.flatten()
    arr = np.delete(arr,n * windowsize + n,0)


    #tinh mean_a, sigma_a, mean_q
    ua = np.mean(arr)
    sa = np.std(arr)
    p = abs(arr - ua)
    up = np.mean(p)
    sp = np.std(p)


    NS = np.mean(abs(arr - cp))

    #dieu kien 1
    T1 = up + sp
    if (NS >= T1):
        return False


    #dieu kien 2
    T2max = ua + 0.5 * sa
    T2min = ua - 0.5 * sa

    if (cp >= T2max or cp <= T2min):
        return False

    #dieu kien 3
    arr.sort()
    l  = windowsize*windowsize - 1
    Q1 = arr[int((l + 1)/4)]
    Q3 = arr[int((3*l +3)/4)]

    if cp>=Q3 or cp<=Q1:
        return False

    return True

def caculateT1(matrix,windowsize,lef,img,cp):
    n=int(windowsize/2)
    sum1 = 0
    cent=(lef[0]+n,lef[1]+n)

    lst=np.array([])
    N = (windowsize*windowsize -1)
    for i in range(windowsize):
        for j in range(windowsize):
            #lst=np.append(lst,img[lef[0]+i,lef[1]+j])
            if (lef[0]+i ==cent[0] and lef[1]+j==cent[1]) or matrix[lef[0]+i,lef[1]+j]==0:
                continue
            else:
                lst=np.append(lst,img[lef[0]+i,lef[1]+j])
                sum1+=img[lef[0]+i,lef[1]+j]         
                     
    ua=sum1/N

    sum2 = 0
    sum3 = 0
    sum4 = 0
    sa=0

    #print("---------")
    for i in range(windowsize):
        for j in range(windowsize):
            if (lef[0]+i ==cent[0] and lef[1]+j==cent[1])or matrix[lef[0]+i,lef[1]+j]==0:
                continue
            else:
                temp =  abs(img[lef[0]+i][lef[1]+j] - ua)
                sum3 += temp*temp
                sum2 += temp
                
    sa = np.sqrt(sum3/N) 
    up = sum2/N

    sum5=0
    sum4=0
    sp=0
    for i in range(windowsize):
        for j in range(windowsize):
            if (lef[0]+i ==cent[0] and lef[1]+j==cent[1]) or matrix[lef[0]+i,lef[1]+j]==0:
                continue
            else:
                sum5+= abs(img[lef[0]+i][lef[1]+j] - img[cent[0]][cent[1]])
                temp1 =  abs(img[lef[0]+i][lef[1]+j] - ua)
                temp2 =  abs(temp1 - up)
                sum4 += temp2*temp2

    NS=sum5/N
    sp=np.sqrt(sum4/N)

    T1=up+sp
    if NS>=T1 :
        return False

    T2max = ua + 0.5 * sa
    T2min = ua - 0.5 * sa
    if cp<=T2min or cp>=T2max:
        return False

    lst.sort()
    Q1 = lst[int((N + 1)/4)]
    Q3 = lst[int((3*N +3)/4)]

    if cp>=Q3 or cp<=Q1:
        return False

    return True


    
    
def algorithm(windowsize,img):          
    h, w = img.shape[:2]
    n=int(windowsize/2)
    #h=256
    #w=256

    print("h",h)
    print("w",w)
    print("n",n)

    sumOK = 0

    rows = np.array([])
    cols =  np.array([])

    rows0 = np.array([])
    cols0 =  np.array([])

    matrix = np.ones((h,w))

    for i in range(h-windowsize+1):
        for j in range(w-windowsize+1):
            cp1=int(i + int(windowsize/2))
            cp2=int(j + int(windowsize/2))
            #x= (img[cp1,cp2,0] >= 0) and img[cp1,cp2,1]>=0 and img[cp1,cp2,2]>=0 
            #y= img[cp1,cp2,0] <= 255 and img[cp1,cp2,1]<= 255 and img[cp1,cp2,2]<= 255

            #if  x and y :
            #if caculateT1(matrix,windowsize,(i,j),img[:,:,0],img[cp1,cp2,0]) and caculateT1(matrix,windowsize,(i,j),img[:,:,1],img[cp1,cp2,1]) and caculateT1(matrix,windowsize,(i,j),img[:,:,2],img[cp1,cp2,2]) : 
            if detect(img[:,:,0], (i,j), windowsize) and detect(img[:,:,1], (i,j), windowsize) and detect(img[:,:,2], (i,j), windowsize): 
                sumOK+=1
            else :
                rows0= np.append(rows0,[cp1])
                cols0= np.append(cols0,[cp2])
                    

                    
    rate0 = sumOK/(h*w)
    print("noise rate : ",1-rate0)

    return rows0.astype(int),cols0.astype(int) 
    			
            

def apply_on_3_channels(img):
    layer_blue = algorithm(5,img[:,:,0])
    layer_green = algorithm(5,img[:,:,1])
    layer_red = algorithm(5,img[:,:,2])
    print("***")
    return img

if __name__ == "__main__":
    img = imread("./images/lana.jpg")
    A = np.array([
    [[1,2,3,4], 
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]],
    [[1,2,3,4], 
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]],
     [[1,2,3,4], 
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]]
    
    ])    
    print('Shape cat.png:', img)



