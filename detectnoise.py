import cv2
import numpy as np
from imageio import imread,imwrite
#xlnntn châp6 bài 1,3 phương pháp ppmi
#knowlege representation and reasoning( cac chuong lq btl 7,11,12)
#def caculateUa(central,img):
m = 40

def caculateT1(windowsize,lef,img,cp):
    n=int(windowsize/2)
    sum1 = 0
    cent=(lef[0]+n,lef[1]+n)
    #print("lef",lef)
    #print("cent",cent)
    #print('A',img)
    lst=np.array([])
    for i in range(windowsize):
        for j in range(windowsize):
            lst=np.append(lst,img[lef[0]+i,lef[1]+j])
            if lef[0]+i ==cent[0] and lef[1]+j==cent[1]:
                continue
            else:
                sum1+=img[lef[0]+i,lef[1]+j]
                #print("x",i,"y",j)
                #print("s",sum1)
   
    ua=sum1/(windowsize*windowsize-1)
    #print("sum1 : ",sum1)
    #print("u1 : ",ua)
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sa=0

    #print("---------")
    for i in range(windowsize):
        for j in range(windowsize):
            if lef[0]+i ==cent[0] and lef[1]+j==cent[1]:
                continue
            else:
                temp =  abs(img[lef[0]+i][lef[1]+j] - ua)
                sum3 += temp*temp
                sum2 += temp
                
    sa = np.sqrt(sum3/(windowsize*windowsize-1)) 
    up = sum2/(windowsize*windowsize-1) 
    #print("sum2 : ",sum2)
    #print("sum3 : ",sum3)
    #print("sa",sa)
    #print("ua",ua)

    sum5=0
    sum4=0
    sp=0
    for i in range(windowsize):
        for j in range(windowsize):
            if lef[0]+i ==cent[0] and lef[1]+j==cent[1]:
                continue
            else:
                sum5+= abs(img[lef[0]+i][lef[1]+j] - img[cent[0]][cent[1]])
                temp1 =  abs(img[lef[0]+i][lef[1]+j] - ua)
                temp2 =  abs(temp1 - up)
                sum4 += temp2*temp2

    NS=sum5/(windowsize*windowsize-1)
    sp=np.sqrt(sum4/(windowsize*windowsize-1))
    #print("sum4",sum4)
    #print("sum5",sum5)
    #print("sp",sp)
    #print("up",up)
    #print("NS",NS)
    T1=up+sp
    #return  True if NS >= T1 and (cp<=m or cp>=255-m)else False 
    #return  True if NS >= T1 else False 
    if NS>=T1 :
        return True

    else :
        T2max = ua + 1.5 * sa
        T2min = ua - 1.5 * sa
        #print("cp ",cp )
        if cp<=T2min or cp>=T2max:
            return True
        else:
        # #   mid = int(windowsize*windowsize/2)
            lst = np.sort(lst)
        #     Q1 = (lst[5]+lst[6])/2
        #     Q3 = (lst[18]+lst[19])/2
            N = (windowsize*windowsize -1)
            Q1 = lst[int((N + 1)/4)]
            Q3 = lst[int((3*N +3)/4)]

        #     #print("lst",np.sort(lst))
            if cp>=Q3 or cp<=Q1: 
        #         #print("T3" ,Q1, '--',Q3, "    ",lst)
                return True


    return False
    
    
def algorithm(windowsize,img):          
    h, w = img.shape[:2]
    n=int(windowsize/2)
    #h=256
    #w=256

    print("h",h)
    print("w",w)
    print("n",n)


    #img= np.ones((h+2*n, w+2*n,3))
    #img[n:n+h, n:n+w,:] = img1
    #img = img.astype(int)
    #print(img.shape,"====",img1.shape)

    #for i in range(w):
    #  for j in range(h):
    sumnoise0=0
    sumnoise1=0
    sumnoise2=0

    rows = np.array([])
    cols =  np.array([])

    rows0 = np.array([])
    cols0 =  np.array([])

    rows1 = np.array([])
    cols1 =  np.array([])

    rows2 = np.array([])
    cols2 =  np.array([])

    for i in range(h-windowsize+1):
        for j in range(w-windowsize+1):
    
    #for i in range(h):
    #    for j in range(w):
            cp1=int(i + int(windowsize/2))
            cp2=int(j + int(windowsize/2))
            x= (img[cp1,cp2,0] >= 0) and img[cp1,cp2,1]>=0 and img[cp1,cp2,2]>=0 
            y= img[cp1,cp2,0] <= 255 and img[cp1,cp2,1]<= 255 and img[cp1,cp2,2]<= 255

            if  x and y:
                if caculateT1(windowsize,(i,j),img[:,:,0],img[cp1,cp2,0]): 
                    #noise_index = np.append(noise_index,[[i + (windowsize/2),j + (windowsize/2)]])
                    sumnoise0+=1
                    rows0= np.append(rows0,[int(i + int(windowsize/2))])
                    cols0 = np.append(cols0,[int(j + int(windowsize/2))])

                    rows= np.append(rows,[int(i + int(windowsize/2))])
                    cols= np.append(cols,[int(j + int(windowsize/2))])
                    
                    #print("noise at : ",i+int(windowsize/2),",",j+int(windowsize/2))
                elif caculateT1(windowsize,(i,j),img[:,:,1],img[cp1,cp2,1]): 
                    #noise_index = np.append(noise_index,[[i + (windowsize/2),j + (windowsize/2)]])
                    sumnoise1+=1
                    rows1= np.append(rows1,[int(i + int(windowsize/2))])
                    cols1 = np.append(cols1,[int(j + int(windowsize/2))])

                    rows= np.append(rows,[int(i + int(windowsize/2))])
                    cols= np.append(cols,[int(j + int(windowsize/2))])
                    
                elif caculateT1(windowsize,(i,j),img[:,:,2],img[cp1,cp2,2]): 
                    #noise_index = np.append(noise_index,[[i + (windowsize/2),j + (windowsize/2)]])
                    sumnoise2+=1
                    rows2= np.append(rows2,[int(i + int(windowsize/2))])
                    cols2 = np.append(cols2,[int(j + int(windowsize/2))])

                    rows= np.append(rows,[int(i + int(windowsize/2))])
                    cols= np.append(cols,[int(j + int(windowsize/2))])
                    
    rate0 = sumnoise0/(h*w)
    rate1 = sumnoise1/(h*w)
    rate2 = sumnoise2/(h*w)
    print("noise rate : ",rate0 ,"  ",rate1," ",rate2)

    #print(img)
    #print(img)
    return rows.astype(int),cols.astype(int),rows1.astype(int),cols1.astype(int),  rows2.astype(int),cols2.astype(int)
    			
            

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
    img = imread("./images/cat.jpg")
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

