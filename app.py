import numpy as np
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from numba import jit,objmode
import re
from os import path
import os
import requests
import gzip
import time
import argparse

def get_mean(arr):
    s=np.sum(arr)
    l=arr.shape[0] * arr.shape[1]
    u=s/l
    return u

def get_std(arr,u):
    # u_mean=np.subtract(arr,u)
    a=arr-u
    b=np.square(a)
    c=np.sum(b)
    d=arr.shape[0] * arr.shape[1]
    return np.sqrt(c/d)


def addSpeckle(img,gamma,sigma):
    gaussSample=np.zeros_like(img)
    gaussSample=cv.randn(gaussSample,0,sigma)
    
    # vgamma=np.power(gaussSample,gamma)
    speckleNoise=np.multiply(img,gaussSample)
    speckleImg=np.add(img,speckleNoise)
    
    return speckleImg


def addGaussianNoise(img,mean,sigma):
    img=np.asarray(img,np.float32)
    ni=np.zeros_like(img)
    gaussSample=cv.randn(ni,mean,sigma)
    noisy_img=cv.add(img,gaussSample)
    # noisy_img=np.clip(noisy_img,0,1)
    
    return noisy_img

def getGaussianKernel(size,dim):
    return tuple([size]*dim)

def addSnPNoise(img,thresh):
    
    output=img.copy()
    randSample=np.random.rand(img.shape[0],img.shape[1])
    output[randSample < thresh]=0
    output[randSample > (1-thresh)]=1
    return output
        
def psnr(img,img_n,peak=255):
    a=img
    b=img_n    
    rmse=np.sqrt(np.mean((a-b)**2))
    return peak_signal_noise_ratio(a,b),rmse

def snr(img,img_n):
    a=img
    b=img_n
    rmse=np.sqrt(np.mean((a-b)**2))
    n=np.sum(np.add(np.square(a) , np.square(b))) 
    mse=np.sum(np.square(np.subtract(a,b)))
    r=10*np.log10(n/mse)
    
    return (r,rmse)

def snr2(img,img_n):
    s1=np.std(img)
    s2=np.std(img_n)
    print(s1,s2)
    return s1/s2

def display(imgs,mv,size=(10,10),*args):
    plt.figure(figsize=size)    
    plt.subplot(2,3,1)
    plt.imshow(imgs[0],cmap='gray')
    plt.title("original")
    
    for i,img_n in enumerate(imgs[1:]):
        plt.subplot(2,3,i+2)
        plt.imshow(img_n,cmap='gray')
        (n,p)=snr(imgs[0],img_n)
        
        plt.title('Noise= {:0.2f}db, RMSE= {:0.1f}%'.format(n,p*100),loc='center',fontdict={'size':10})
        # plt.title()
        # plt.text(2,0.65,s='Time: {:0.2f}s'.format(float(args[0])))
        plt.xlabel('Time: {:0.2f}s'.format(float(args[0])),fontdict={'color':'darkred'})
        if not args[1]:
            plt.close()
    if args[1]:#display figures
        plt.show()
    return n #latest snr
    

@jit(nopython=True)
def createPaddedImg(img,pad):
    paddedImg=np.zeros((img.shape[0]+pad*2,img.shape[1]+pad*2),np.float32)
    paddedImg[pad:pad+img.shape[0],pad:pad+img.shape[1]]=img.copy()#center

    paddedImg[pad:pad+img.shape[0],0:pad]=np.fliplr(img[:,0:pad])#left
    paddedImg[pad:pad+img.shape[0],pad+img.shape[1]:]=np.fliplr(img[:,img.shape[1]-pad:])#right

    paddedImg[0:pad,:]=np.flipud(paddedImg[pad:pad*2,:])#up
    paddedImg[pad+img.shape[0]:,:]=np.flipud(paddedImg[paddedImg.shape[0]-2*pad:paddedImg.shape[0]-pad,:])#down

    return paddedImg

def remove_background(img_gr):
    sobel_x=[[-1,-2,-1],
             [0,0,0],
             [1,2,1]]
    sobel_y=[
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ]
    # gx=signal.convolve2d(img_gr,sobel_x)
    # gy=signal.convolve2d(img_gr,sobel_y)
    # G=np.sqrt(np.square(gx),np.square(gy))

    # G = cv.adaptiveThreshold(img_gr, 1,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)
    
    # px=img_gr.shape[0]
    # py=img_gr.shape[1]
    # Gx=G.shape[0]
    # Gy=G.shape[1]
    # px=Gx-px
    # py=Gy-py
    # return G[py//2:img_gr.shape[0]+px//2,px//2:img_gr.shape[1]+px//2]
    mask=img_gr < 0.07
    return mask
    

@jit(nopython=True)
def nonLocalMeans(img,u1,var,bW,sW,h,optimized=False,voxel_selection=False,bayesian=False):
    pad=bW//2 #pad length on each size
    paddedImg=createPaddedImg(img,pad)
    img_mean=np.zeros_like(paddedImg)
    img_var=np.zeros_like(paddedImg)
    sWr=sW//2
    totalIterations = img.shape[1]*img.shape[0]*(bW - sW)**2
    iter=0
    _mean=u1#0.6
    _var=var+1e-7# 0.4+1e-7
    gamma=0.5
    epsilon=10^-13
    zepsilon=0.0001
    
    # return paddedImg[pad:pad+img.shape[0],pad:pad+img.shape[1]]
    if optimized:
        
        for px in range(pad,pad+img.shape[1]):
            for py in range(pad,pad+img.shape[0]):
                img_mean[py,px]=_local_means(img,px,py,sWr)
                img_var[py,px]=_local_variance(img,img_mean[py,px],px,py,sWr)
    
    # volWin=len(range(-sW//2,sW//2+1))**2
    
    with objmode(start='f8'):
        start=time.time()
    for imgX in range(pad,pad+img.shape[1]):
        for imgY in range(pad,pad+img.shape[0]):
            bWX=imgX-pad #big window X point (corner point)
            bWY=imgY-pad #big window Y point (corner point)

            pixelNbrW=paddedImg[imgY-sWr:imgY+sWr+1,imgX-sWr:imgX+sWr+1]#pixel neighbor window
            totalWeight=0
            pixel=0
            weight=0
            
            if optimized and ((img_mean[imgY,imgX] < zepsilon) or (img_var[imgY,imgX] < zepsilon)):
                weight=1
                totalWeight+=weight
                pixel=paddedImg[imgY,imgX]#same value
            else:
                wmax=0
                for sWX in range(bWX,bWX+bW-sW):
                    for sWY in range(bWY,bWY+bW-sW):
                        
                        if optimized:
                            if (img_mean[sWY,sWX] < zepsilon) or (img_var[sWY,sWX] < zepsilon): #rule out zero division
                                continue
                            if (sWX == imgX) and (sWY == imgY): # condition 2
                                continue
                            if voxel_selection:
                                tu=img_mean[imgY,imgX] / (img_mean[sWY,sWX])
                                tv=img_var[imgY,imgX] / (img_var[sWY,sWX])

                        sNbrW=paddedImg[sWY:sWY+sW+1,sWX:sWX+sW+1]
                        if not bayesian:
                            Dist=np.sum(np.square(pixelNbrW - sNbrW))
                        else:#Calculate Pearson Distance
                            n=np.square(pixelNbrW - sNbrW)
                            d=np.power(sNbrW,2*gamma)+epsilon
                            # d=2*d * (0.4)**2
                            Dist=np.sum(n/d)

                        
                            
                        if (voxel_selection and (_mean < tu < 1/_mean) and (_var < tv < 1/_var)) or not voxel_selection:
                            
                            weight=np.exp(-Dist/(h))# / volWin / h)
                            # if weight > wmax or not optimized:
                            wmax=weight
                            totalWeight+=weight
                            pixel+=weight * paddedImg[sWY+sWr,sWX+sWr]

                            percentage=iter * 100 / totalIterations
                            if percentage % 5 ==0:
                                print(percentage)
                            iter+=1
                    
            
                
            if totalWeight!=0:
                restoredPixel=pixel/totalWeight
                paddedImg[imgY,imgX]=restoredPixel
    
    with objmode(ttime='f8'):
        ttime=time.time()-start
        
        
        
    return paddedImg[pad:pad+img.shape[0],pad:pad+img.shape[1]],ttime
@jit(nopython=True)
def _local_means(img,x,y,sWr):
    
    p=0
    for px in range(x,x+ (2*sWr)+1):
        for py in range(y,y+ (2*sWr)+1):
            p+=img[py,px]
    vol=len(range(x,x+ (2*sWr)+1))*len(range(y,y+ (2*sWr)+1))
    
    return p/vol
@jit(nopython=True)
def _local_variance(img,mean,x,y,sWr):
    
    cnt=0
    p=0
    for px in range(x,x+ (2*sWr)+1):
        for py in range(y,y+ (2*sWr)+1):
            
                p+=(img[py,px] - mean)**2
                cnt+=1
    
    return p/(cnt-1)

@jit(nopython=True)
def calculate_pearson_distance(img,x,y,bx,by,sWr):
    gamma=0.5
    epsilon=10^-13
    n=np.zeros((2*sWr+1,2*sWr+1),np.float64)
    d=np.zeros((2*sWr+1,2*sWr+1),np.float64)
    for sx in range(0,2*sWr+1): #small window in x-dir
        for sy in range(0,2*sWr+1): #small window in y-dir
            
            pxi=x+sx #mapping window on image
            pyi=y+sy #mapping window on image
            # bxj=bx+sx
            # byj=by+sy
            bxj=bx+sx #padding included
            byj=by+sy #padding included
            n[sy,sx]=np.square(img[pyi,pxi] - img[byj,bxj])
            d[sy,sx]=np.power(img[byj,bxj],2*gamma)+epsilon
    pd=np.sum(n/d)
    # print(pd)
    return pd
@jit(nopython=True)
def calculate_distance(img,x,y,bx,by,sWr):
    d=0
    cnt=0

    for sx in range(0,2*sWr+1): #small window in x-dir
        for sy in range(0,2*sWr+1): #small window in y-dir
            pxi=x+sx #mapping window on image
            pyi=y+sy #mapping window on image
            bxj=bx+sx #padding included
            byj=by+sy #padding included
            d+=np.square(img[pyi,pxi] - img[byj,bxj])
            
            cnt+=1
    
    
    return np.divide(d,cnt)
@jit(nopython=True)
def get_average(img,bxj,byj,avBlock,w,r):
    
    
    for ax in range(avBlock.shape[1]): #small window in x-dir (top left corner point)
        for ay in range(avBlock.shape[0]): #small window in y-dir (top left corner point)
            pxj=bxj+ax #(top left corner point)
            pyj=byj+ay #(top left corner point)

            avBlock[ay,ax]+=w*(img[pyj,pxj])**2
            
            
            
@jit(nopython=True)
def get_intensity(img,pixel_update,px,py,avBlock,total_weights):
    
    
    
    for ax in range(avBlock.shape[1]): #small window in x-dir
        for ay in range(avBlock.shape[0]): #small window in y-dir
            pxi=px+ax
            pyi=py+ay
            
            v=np.divide(avBlock[ay,ax],total_weights)
            
            if v > 0:
                v=np.sqrt(v)
            else: v=0
            
            img[pyi,pxi]+=v
            pixel_update[pyi,pxi]+=1
            
    


@jit(nopython=True)
def bwNLM(original_img,patch_radius,block_radius,h,u1,var,voxel_selection=False,bayesian=False):
    ox=original_img.shape[1]
    oy=original_img.shape[0]
    img=createPaddedImg(original_img,patch_radius)
    img_mean=np.zeros_like(img)
    img_var=np.zeros_like(img)
    average_block=np.zeros((2*block_radius+1,2*block_radius+1),np.float64)
    restored_image=np.zeros_like(img)
    pixel_update=np.zeros_like(img)
    _mean=u1 #0.75
    _var=var+1e-7 #0.5+1e-7
    tu=0
    tv=0
    zepsilon=0.0001
    sWr=block_radius
    totalIterations = img.shape[1]*img.shape[0]*(patch_radius*2 - block_radius*2)**2
    iter=0

    # if optimized:
    for px in range(patch_radius,patch_radius+ox-(2*sWr+1)):
        for py in range(patch_radius,patch_radius+oy-(2*sWr+1)):
            img_mean[py,px]=_local_means(img,px,py,sWr)
            img_var[py,px]=_local_variance(img,img_mean[py,px],px,py,sWr)
    
    
    with objmode(start='f8'):
        start=time.time()
    for px in range(patch_radius,ox+patch_radius-(2*sWr+1),2): #image central x points
        for py in range(patch_radius,oy+patch_radius-(2*sWr+1),2): # image central y points
            wmax=0
            total_weight=0
            average_block[...]=0
            av_cnt=0
            if (img_mean[py,px] < zepsilon) or (img_var[py,px] < zepsilon):#mean or variance is zero e.g, Background patch
                w=1
                get_average(img,px,py,average_block,w,sWr)
                total_weight+=w
            else:
                wmax=0
                for bx in range(px-patch_radius,px+patch_radius+1-(2*sWr)): # Big window corner x points
                    for by in range(py-patch_radius,py+patch_radius+1-(2*sWr)): #Big window corner y points
                        
                        # if optimized:
                        if (bx == px) and (by == py): # condition 2
                            continue
                        if (img_mean[by,bx] < zepsilon) or (img_var[by,bx] < zepsilon): #rule out zero division
                            continue
                        if voxel_selection:
                            tu=img_mean[py,px] / (img_mean[by,bx])
                            tv=img_var[py,px] / (img_var[by,bx])
                        # if (_mean < tu < 1/_mean) or not optimized:
                        if not bayesian:
                            d=calculate_distance(img,px,py,bx,by,sWr)
                        else: 
                            d=calculate_pearson_distance(img,px,py,bx,by,sWr)
                        if (voxel_selection and (_mean < tu < 1/_mean) and (_var < tv < 1/_var)) or not voxel_selection:
                            w=np.exp(-d/h)
                            # if w > wmax: #exclude same blocks
                            wmax=w
                            get_average(img,bx,by,average_block,w,sWr)
                            total_weight+=w
                            av_cnt+=1

                        percentage=iter * 100 / totalIterations
                        if percentage % 5 ==0:
                            print(percentage)
                        iter+=1
                            
                
                
            if total_weight !=0:
                
                get_intensity(restored_image,pixel_update,px,py,average_block,total_weight)
            
                
        
    
    for px in range(patch_radius,ox+patch_radius):
        for py in range(patch_radius,patch_radius+oy):
            if pixel_update[py,px] != 0:
                restored_image[py,px]=restored_image[py,px]/pixel_update[py,px]
            else: 
                restored_image[py,px]=img[py,px]
    with objmode(ttime='f8'):
        ttime=time.time()-start
    
    return restored_image[patch_radius:patch_radius+oy,patch_radius:patch_radius+ox],ttime

def mean_std_normalize(arr):
    arr32=np.asarray(arr,np.float32)

    u=np.mean(arr32)
    std=np.std(arr32)

    t1=np.subtract(arr32,u) # arr - mean
    return t1/std

def min_max_normalize(arr):
    # return cv.normalize(arr,arr,0,255,cv.NORM_MINMAX)
    arr32=np.asarray(arr,np.float32)
    mappedMax = 255
    mappedMin = 0
    originalMax = np.amax(arr32)
    originalMin = np.amin(arr32)
    deltaOrigin = originalMax - originalMin
    scalar = (mappedMax - mappedMin)/deltaOrigin
    normalizedImg = (arr - originalMin) * scalar + mappedMin
    normalizedImg = normalizedImg.astype(np.uint8)
    
    return normalizedImg

def download_brainweb(p):
    if not path.exists(p):
        os.makedirs(p)

    LINKS = "04 05 06 18 20 38 41 42 43 44 45 46 47 48 49 50 51 52 53 54"
    LINKS = [
    'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1'
    '?do_download_alias=subject' + i + '_crisp&format_value=raw_short'
    '&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D'
    for i in LINKS.split()]
    RE_SUBJ = re.compile('.*(subject)([0-9]+).*')
    LINKS = dict((RE_SUBJ.sub(r'\1_\2.bin.gz', i), i) for i in LINKS)

    files=[]
    for f, origin in LINKS.items():
        
        fp=str(p)+"/"+f
        files.append(f)
        if not os.path.exists(str(p)+"/"+f):
            print("Downloading ",f,"\n")    
            d = requests.get(origin, stream=True)
            with open(fp, 'wb') as fo:
                for chunk in d.iter_content(chunk_size=None):
                    fo.write(chunk)
    return files
def load_file(path,files,no):
    ab=files[0].split("_")
    b=ab[1].split(".")
    file=ab[0]+"_"+str(no)+"."+b[-2]+"."+b[-1]
    print("Loading MRI file ",file)
    with gzip.open(str(path)+"/"+str(file)) as fi:
        data = np.frombuffer(fi.read(), dtype=np.uint16)
    return data.reshape((362, 434, 362))

def obnlm_voxel(gt,noisy_img,sigma,d,M,u1,var,fig=False):
    vals=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    
    _snr=np.zeros(len(vals))
    
    for j,u in enumerate(vals):
        h2=2*0.5 * (sigma)**2 * (2*d+1)**2
        print("Mean u1 = ",u)
        denoised,t=bwNLM(noisy_img,M,d,h2,u,var,voxel_selection=True,bayesian=True)#(arr,bigWindowRadius,smallWindowRadius,NF) # for slides 4
        imgs=[gt,noisy_img,denoised]
        _snr[j]=display(imgs,3,(10,10),t,fig)    

    plot_B_snr(vals,_snr)

    for j,v in enumerate(vals):
        h2=2*0.5 * (sigma)**2 * (2*d+1)**2
        print("Mean u1 = ",u)
        denoised,t=bwNLM(noisy_img,M,d,h2,u1,v,voxel_selection=True,bayesian=True)#(arr,bigWindowRadius,smallWindowRadius,NF) # for slides 4
        imgs=[gt,noisy_img,denoised]
        _snr[j]=display(imgs,3,(10,10),t,fig)    

    plot_B_snr(vals,_snr)

def obnlm_B(gt,noisy_img,sigma,d,M,u1,var,fig=False):
    B=[0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7,1]
    _snr=np.zeros(len(B))
    
    for j,b in enumerate(B):
        h2=2*b * (sigma)**2 * (2*d+1)**2
        print("Smoothing = ",h2)
        denoised,t=bwNLM(noisy_img,M,d,h2,u1,var,voxel_selection=True,bayesian=True)#(arr,bigWindowRadius,smallWindowRadius,NF) # for slides 4
        imgs=[gt,noisy_img,denoised]
        _snr[j]=display(imgs,3,(10,10),t,fig)    

    plot_B_snr(B,_snr)

def plot_B_snr(B,snr_mat):
    plt.plot(B,snr_mat[0],'r--')
    plt.plot(B,snr_mat[1],'g:')
    plt.plot(B,snr_mat[2],'b-')
    plt.xlabel(r'$ \beta$')
    plt.ylabel("SNR (in db)")
    plt.title(r'Influence of $\beta $')
    plt.grid(True)
    plt.legend([r'$\sigma = 0.2$',r'$\sigma = 0.4$',r'$\sigma = 0.8$'])

def preprocess_img(img,grayscale=0,scale=1,brainweb=False):
    if isinstance(img,Path):
        img=cv.imread(str(img),grayscale)#grayscale image    
    img_gray=np.float32(img)
    mmin=np.min(img_gray)
    mmax=np.max(img_gray)
    img_normalize=(img_gray-mmin) / (mmax - mmin)
    print(img_normalize.shape)    
    img_resize=cv.resize(img_normalize,(0,0),fx=scale,fy=scale)#grayscale resize image
    return img_resize




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='us data', help='Image Data Path')
    parser.add_argument('--brainweb', default=False, help='Perform Filtering on MRI brain web dataset')
    parser.add_argument('--brainweb-3d-file',type=int, default=54, help='FILES = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]')
    parser.add_argument('--brainweb-2d-file',type=int, default=181, help='Select a 2D frame from a 3D MRI file')
    parser.add_argument('--filter',type=int, default=3, help='[NLM, Optimized NLM, BNLM, OBNLM]')
    parser.add_argument('--plot-beta', default=False, help='Plot Beta vs SNR graph')
    parser.add_argument('--plot-voxel', default=False, help='Plot u1 vs SNR graph and var vs SNR graph')
    opt = parser.parse_args()

    
    images_path=Path(opt.data)
    image_files=list(images_path.glob("*"))
    print(opt.brainweb)
    data=[]
    if opt.brainweb:
        dataset_path=Path("brainweb")
        files=download_brainweb(dataset_path)

    # FILES = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
        data=load_file(dataset_path,files,opt.brainweb_3d_file)
        data=data[opt.brainweb_2d_file,:,:]
    
    grayscale=0
    scale=1
    uGauss=0
    threshold=0.025
    
    """
            Parameters for Non Local Means
    """
    M=5
    d=1
    ws=len(range(-2*d,2*d+1))
    """------------------------------"""
    
    gaussian_noises=[3,9,15,21]
    speckle_noises=[0.2,0.4,0.8]

    if len(image_files)>0:
        files=image_files
    elif opt.brainweb:
        files=[data]

    for f in files:
        if type(f) == str:
            print("Processing File -----------------> ",f)
        else: print("Processing File -----------------> ",opt.brainweb_3d_file)
        img=preprocess_img(f,brainweb=opt.brainweb)

        imgSpeckle=np.zeros_like(img.shape,dtype=np.float32)
        # imgGauss=np.ndarray((len(gaussian_noises),*img.shape))
        denoised=np.zeros_like(img.shape,dtype=np.float32)
        ttime=0

        for i,n in enumerate(speckle_noises):
            if  n == 0.2 or n == 0.4:
                    continue
            print("Noise ",n)
            sigma=n
            imgSpeckle=random_noise(img,mode='speckle',mean=0,var=(sigma)**2,clip=False)
    
            # imgGauss[i]=addGaussianNoise(img_grn,uGauss,sigma)
            # imgSnP[i]=addSnPNoise(img_grn,threshold)
            """
            Auto Tuning Parameter with B = 0.5
            """
            h2=2*0.5 * (sigma)**2 * (2*d+1)**2
            u1=(0.85 if sigma == 0.2 else 0.75 if sigma == 0.4 else 0.5)
            var=0.3
            
            """-------------------------"""
            if opt.plot_beta:
                obnlm_B(img,imgSpeckle,sigma,d,M,u1,var,False)
            if opt.plot_voxel:
                obnlm_voxel(img,imgSpeckle,sigma,d,M,0.95,0.5,False)
            else:
                match opt.filter:
                    case 0:
                        print("*************** NLM ***************")
                        denoised,ttime=nonLocalMeans(imgSpeckle,0,0,bW=2*M + 1,sW=2*d,h=h2,optimized=True,voxel_selection=False,bayesian=True)#(arr,(2M+1),2d,NF) #for slides 1
                    case 1:
                        print("*************** Optimized NLM ***************")
                        denoised,ttime=nonLocalMeans(imgSpeckle,u1,var,bW=2*M + 1,sW=2*d,h=h2,optimized=True,voxel_selection=True,bayesian=True)#(arr,(2M+1),2d,NF)
                    case 2:
                        print("*************** Block Wise NLM ***************")
                        denoised,ttime=bwNLM(imgSpeckle,M,d,h2,0,0,voxel_selection=False,bayesian=True)#(arr,bigWindowRadius,smallWindowRadius,NF) # for slides 3
                    case 3:
                        print("*************** Optimized Block Wise NLM ***************")
                        denoised,ttime=bwNLM(imgSpeckle,M,d,h2,u1,var,voxel_selection=True,bayesian=True)#(arr,bigWindowRadius,smallWindowRadius,NF) # for slides 4
                    case _:
                        print("Please Chose From [0,1,2,3]")
                
                imgs=[img,imgSpeckle,denoised]
                display(imgs,3,(10,10),ttime,True)    

        
        # break
    

            
    
    

    