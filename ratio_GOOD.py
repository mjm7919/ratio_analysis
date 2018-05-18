import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize



def click_and_crop(event,x,y,flags,param):
    global refPt #, cropping
    if event==cv2.EVENT_LBUTTONDOWN:
        refPt.append([(x,y)])
#        cropping=True
    elif event==cv2.EVENT_LBUTTONUP:
        refPt.append([(x,y)])
#        cropping=False
        

def plot_color_image(image,title,colorscheme,file_ext,colorbar,path):
#    sizes=np.shape(image)
    my_dpi=96
    pix=700
    plt.figure(figsize=(pix/my_dpi, pix/my_dpi), dpi=my_dpi)
    plt.imshow(image,cmap=colorscheme)
    if colorbar == 'yes':
        plt.colorbar(fraction=0.046, pad=0.04)
#        plt.colorbar(orientation='horizontal')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    print('saving %s.%s' % (title,file_ext,))
    plt.savefig('%s.%s' % (os.path.join(path,title),file_ext,),dpi=my_dpi, bbox_inches='tight')
    plt.show()


def plot_image(image,title,colorscheme,file_ext,colorbar,path):
#    sizes=np.shape(image)
    my_dpi=96
    pix=700
    plt.figure(figsize=(pix/my_dpi, pix/my_dpi), dpi=my_dpi)
    plt.imshow(image,cmap=colorscheme,vmin=0,vmax=255)
    if colorbar == 'yes':
        plt.colorbar(fraction=0.046, pad=0.04)
#        plt.colorbar(orientation='horizontal')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    print('saving %s.%s' % (title,file_ext,))
    plt.savefig('%s.%s' % (os.path.join(path,title),file_ext,),dpi=my_dpi, bbox_inches='tight')
    plt.show()

def plot_cv2(image,title,file_ext,path):
    cv2.imshow(title,image)
    print('saving %s.%s' % (title,file_ext,))
    cv2.imwrite('%s.%s'%(os.path.join(path,title),file_ext,),image)
    
def plot_histogram(image,title,xlabel,ylabel,file_ext):
    plt.figure()
    sizes=np.shape(image)
    plt.hist(image.ravel(),256,[0,256])
    plt.title('%s Histogram' % (title,))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    print('saving %s_Histogram.%s' % (title,file_ext,))
    plt.savefig('%s_Histogram.%s' % (title,file_ext,),dpi=sizes[0], bbox_inches='tight')
    plt.show()
    
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 	# 1 standard deviation above and below median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the canny detection edged image
	return edged

#
# read first frame of stabilized video

plt.close('all')
cv2.destroyAllWindows

# path = 'C:/Users/Matthew/Documents/Image_Analysis/RBC_flow'
#cv2.imwrite(os.path.join(path,'test.png'),test)
#cv2.waitKey(0)
plt.close('all')
cv2.destroyAllWindows()
path='C:/Users/Matthew/Documents/Image_Analysis/RBC_flow/Series/Series2/'

cap = cv2.VideoCapture(os.path.join(path,'Series002_Stablized.avi'))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

status, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

crop_size=25
crop=frame1[crop_size-1:-1-crop_size,crop_size-1:-1-crop_size]
hcrop,wcrop = crop.shape[:2]
crop_prev = crop.copy()
prev_thresh = np.zeros((hcrop,wcrop),np.uint8)
close_app = []
q=0
''' 
###### parameter search #######
for param1 in np.arange(11,24):
    for param2 in np.arange(1.74,2.04,0.04):
        for p3, param3 in enumerate(np.arange(1.74,2.04,0.04)):
'''
cap.release()
test_params=[]
for param1 in np.arange(1.78,2.04,0.04):
    start_time = time.time()
    print('param1=',str(param1))
    for param2 in np.arange(1.78,2.04,0.04):
        print('param2=',str(param2))
        for param3 in np.arange(7,18,1):
            print('param3=',str(param3))
            for kernel_k1 in np.arange(3,11,2):
                print('kernel_k1=',str(kernel_k1))
                for kernel_k2 in np.arange(3,11,2):
                    print('kernel_k2=',str(kernel_k2))
                    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_k1,kernel_k1))
                                
                    cap = cv2.VideoCapture(os.path.join(path,'Series002_Stablized.avi'))
                    status, frame1 = cap.read()
                    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                    
                    crop_size=25
                    crop=frame1[crop_size-1:-1-crop_size,crop_size-1:-1-crop_size]
                    hcrop,wcrop = crop.shape[:2]
                    crop_prev = crop.copy()
                    prev_thresh = np.zeros((hcrop,wcrop),np.uint8)
                    for k in np.arange(1,n_frames - 1,1):
                        status, frame = cap.read()
                        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        crop = frame[crop_size-1:-1-crop_size,crop_size-1:-1-crop_size]
                        ratio = np.arctan(crop/crop_prev) - np.pi/4 
                        
                        ht=1.85
                        ub=np.array(np.abs(np.mean(ratio)+param1*np.std(ratio)))
                        lb=np.array(np.abs(np.mean(ratio)-param2*np.std(ratio)))
                        ret1,thresh1 = cv2.threshold(ratio,ub,255,cv2.THRESH_BINARY)
                        ret2,thresh2 = cv2.threshold(ratio,-lb,255,cv2.THRESH_BINARY_INV) # cannot have negative
                        
                        thresh3 = thresh1 + thresh2
                        full_thresh = thresh3+prev_thresh
                        full_thresh = full_thresh.astype(np.uint8)
                        ret3, full_thresh =cv2.threshold(full_thresh,250,255,cv2.THRESH_BINARY)
                    
                        if k%param3 == 0:
                            q=q+1
                            close_app.append(k)
                            full_thresh = cv2.erode(full_thresh,kernel,iterations = 1)
                            full_thresh = cv2.dilate(full_thresh,kernel,iterations = 1)
                    
            #            cv2.imshow('thresh',full_thresh)
            #            if cv2.waitKey(1) & 0xFF == ord('q'):
            #                break 
                    #   
                        crop_prev = crop.copy()
                        prev_thresh = full_thresh.copy()
                    #    time.sleep(.0008)
                    
                    cap.release()
                    
                    #plt.close('all')
                    #cv2.destroyAllWindows()
                    ''' smoothing of capillaries'''
                    #plot_image(full_thresh,'raw ratio analysis','gray','png','no',path)
                    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_k2,kernel_k2))
                    
                    full_threshm = cv2.medianBlur(full_thresh,3)
                    #full_threshm = cv2.medianBlur(full_threshm,3)
                    full_threshe = cv2.erode(full_threshm,kernel,iterations = 1)
                    __, cnts, __ = cv2.findContours(full_threshe.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = sorted(cnts, key = cv2.contourArea, reverse = False)
                    area=[]
                    ones_mat = np.ones((hcrop,wcrop),np.uint8)
                    for i, cnt in enumerate(cnts):
                        area.append(cv2.contourArea(cnts[i][:,0,:]))
                        if area[i]<300:
                            roi=cv2.fillPoly(ones_mat.copy(),[cnts[i][:,0,:]],0)
                            full_threshe = full_threshe*roi
                    
                    full_threshD = cv2.dilate(full_threshe,kernel,iterations = 1)
                    #        plot_image(full_threshD,'post morphological ratio analysis','gray','png','no',path)
                    #
                    full_threshD_copy=full_threshD.copy()
                    full_threshG = cv2.GaussianBlur(full_threshD,(7,7),60.0)
                    ret1 ,full_threshG = cv2.threshold(full_threshG,80,255,cv2.THRESH_BINARY)
                    full_threshG = cv2.medianBlur(full_threshG,3)
                    
                    '''
                    ############# params for smoothing #########
                    
                     for p4, param4 in enumerate(np.arange(30,70,1)): # sigma
                    for p5, param5 in enumerate(np.arange(60,100,1)):  # threshold
                    
                    full_threshG = cv2.GaussianBlur(full_threshD,(7,7),60.0)
                    ret1 ,full_threshG = cv2.threshold(full_threshG,80,255,cv2.THRESH_BINARY)
                    full_threshG = cv2.medianBlur(full_threshG,3)
                    full_threshG = cv2.GaussianBlur(full_threshG,(5,5),50.0)
                    ret1, full_threshG = cv2.threshold(full_threshG,100,255,cv2.THRESH_BINARY)
                    full_threshG = cv2.GaussianBlur(full_threshG,(3,3),30.0)
                    ret1, full_threshG = cv2.threshold(full_threshG,120,255,cv2.THRESH_BINARY)
                    '''
                    
                    #plot_image(full_threshG,'smoothed ratio analysis','gray','png','no',path)
        #            plot_cv2(full_threshG,'smoothed ratio analysis','png',path)
                    roi1_coor=[[(133,51)],[(208,101)]]
                    roi2_coor=[[(391,166)],[(433,229)]]
                    roi1 = full_threshG[roi1_coor[0][0][1]:roi1_coor[1][0][1],roi1_coor[0][0][0]:roi1_coor[1][0][0] ]
                    roi2 = full_threshG[roi2_coor[0][0][1]:roi2_coor[1][0][1],roi2_coor[0][0][0]:roi2_coor[1][0][0] ]
                    #km=np.hstack((roi1,roi2))
                    #cv2.imshow('pp',roi2)
                    __, cnts1, __ = cv2.findContours(roi1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    __, cnts2, __ = cv2.findContours(roi2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    if len(cnts1)==2 and len(cnts2)==2:
                        test_params.append([kernel_k1,kernel_k2,param1,param2,param3])
                    
    end_time = time.time()
    print("total time taken this loop: ", (end_time - start_time)/60)