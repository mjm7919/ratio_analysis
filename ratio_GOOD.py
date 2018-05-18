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
#
''' 
######### make circle on video ######

empty = np.zeros((hcrop,wcrop),np.uint8)
make_circle=cv2.line(empty,(249,227),(260,245+20),255,15)
make_circle=cv2.circle(empty,(250,237),25,255,-1)
#
for i in range(0,hcrop):
    for j in range(0,wcrop):
        if make_circle[i,j] ==255:
            crop[i,j]=255.0
crop_circle = crop.copy()
plot_image(crop_circle,'insert circle','gray','png','no',path)
'''

crop_3channels = np.dstack((crop_prev,crop_prev,crop_prev))


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
start_time = time.time()
test_params=[]
#kernel_k=5
#for param1 in np.arange(11,24):
#
#for kernel_k1 in np.arange(3,7,2):
#    for kernel_k2 in np.arange(3,7,2):
#        print('kernel_k1=',str(kernel_k1))
#        print('kernel_k2=',str(kernel_k2))
kernel_k1=5
kernel_k2=5
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_k1,kernel_k1))
    #kernel = np.ones((3,3),np.uint8)
            
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

for k in np.arange(1,n_frames - 1,1):
    status, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    crop = frame[crop_size-1:-1-crop_size,crop_size-1:-1-crop_size]
    ratio = np.arctan(crop/crop_prev) - np.pi/4 
    
    ht=1.85
    ub=np.array(np.abs(np.mean(ratio)+ht*np.std(ratio)))
    lb=np.array(np.abs(np.mean(ratio)-ht*np.std(ratio)))
    ret1,thresh1 = cv2.threshold(ratio,ub,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(ratio,-lb,255,cv2.THRESH_BINARY_INV) # cannot have negative
    
    thresh3 = thresh1 + thresh2
    full_thresh = thresh3+prev_thresh
    full_thresh = full_thresh.astype(np.uint8)
    ret3, full_thresh =cv2.threshold(full_thresh,250,255,cv2.THRESH_BINARY)

    if k%13 == 0:
        q=q+1
        close_app.append(k)
        full_thresh = cv2.erode(full_thresh,kernel,iterations = 1)
        full_thresh = cv2.dilate(full_thresh,kernel,iterations = 1)
#        full_thresh = cv2.medianBlur(full_thresh,3)    

#    full_thresh_denoised = cv2.medianBlur(full_thresh,3)
##    full_threshm = cv2.medianBlur(full_threshm,3)
#    full_thresh_denoised = cv2.erode(full_thresh_denoised,kernel,iterations = 1)
#    full_thresh_denoised = cv2.dilate(full_thresh_denoised,kernel,iterations = 1)

    cv2.imshow('thresh',full_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
#   
    crop_prev = crop.copy()
    prev_thresh = full_thresh.copy()
#    time.sleep(.0008)

cap.release()
#        plot_image(full_thresh,'raw ratio analysis','gray','png','no',path)
#plot_image(full_thresh_denoised,'morphological ratio analysis','gray','png','no',path)


#
#plt.close('all')
#cv2.destroyAllWindows()
''' smoothing of capillaries'''
#plot_image(full_thresh,'raw ratio analysis','gray','png','no',path)
kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_k2,kernel_k2))
#kernel=np.ones((3,3),np.uint8)

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
plot_cv2(full_threshG,'smoothed ratio analysis','png',path)
roi1_coor=[[(133,51)],[(208,101)]]
roi2_coor=[[(391,166)],[(433,229)]]
roi1 = full_threshG[roi1_coor[0][0][1]:roi1_coor[1][0][1],roi1_coor[0][0][0]:roi1_coor[1][0][0] ]
roi2 = full_threshG[roi2_coor[0][0][1]:roi2_coor[1][0][1],roi2_coor[0][0][0]:roi2_coor[1][0][0] ]
#km=np.hstack((roi1,roi2))
cv2.imshow('pp',roi2)
__, cnts1, __ = cv2.findContours(roi1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
__, cnts2, __ = cv2.findContours(roi2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
if len(cnts1)==2 and len(cnts2)==2:
    test_params.append([kernel_k1,kernel_k2])
    
end_time = time.time()
print("total time taken this loop: ", end_time - start_time)

#%%
#cnts = sorted(cnts, key = cv2.contourArea, reverse = False)
#area=[]
#ones_mat = np.ones((hcrop,wcrop),np.uint8)
#for i, cnt in enumerate(cnts):
#    area.append(cv2.contourArea(cnts[i][:,0,:]))
#    if area[i]<300:
#        roi=cv2.fillPoly(ones_mat.copy(),[cnts[i][:,0,:]],0)
#        full_threshe = full_threshe*roi

'''
refPt=[]
cropping=False
clone = full_threshG.copy()
cv2.namedWindow("full_threshG")
cv2.setMouseCallback("full_threshG", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("full_threshG", full_threshG)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		full_threshG = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 
# if there are two reference points, then crop the region of interest
        
# from teh image and display it
full_threshG=np.dstack((full_threshG,full_threshG,full_threshG))
cv2.rectangle(full_threshG,refPt[0][0],refPt[1][0],(0,255,0),2)
cv2.imshow('full_threshG',full_threshG)
#time.sleep(5)
accept_roi='y'#input('accept roi? y/n: ')
if accept_roi=='y':
    if len(refPt) == 2:
    	roi = clone[refPt[0][0][1]:refPt[1][0][1],refPt[0][0][0]:refPt[1][0][0] ]
    	cv2.imshow("ROI", roi)
'''
#%%
skel_bones = []
bool_treshG=np.array(full_threshG,dtype=bool)
    
# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(bool_treshG, return_distance=True)
medial_axis_skel = np.zeros((hcrop,wcrop),np.float) # create float numbers of skeleton
diff_skel = np.zeros((hcrop,wcrop),np.float) # create float numbers of skeleton

for i in range(0,hcrop):
    for j in range(0,wcrop):
        if skel[i,j] == True:
            medial_axis_skel[i,j]=255.0
        else:
            medial_axis_skel[i,j]=0.0

dist_to_medial_axis = distance * skel#  makes floats
skel_bones.append([skel,distance,medial_axis_skel,dist_to_medial_axis])
#%%
skel_bones[0][2]=np.delete(skel_bones[0][2],(skel_bones[0][2].shape[0]-1),axis=0)
skel_bones[0][2]=np.delete(skel_bones[0][2],(0),axis=0)
skel_bones[0][2]=np.delete(skel_bones[0][2],(skel_bones[0][2].shape[1]-1),axis=1)
skel_bones[0][2]=np.delete(skel_bones[0][2],(0),axis=1)

skel_bones[0][3]=np.delete(skel_bones[0][3],(skel_bones[0][3].shape[0]-1),axis=0)
skel_bones[0][3]=np.delete(skel_bones[0][3],(0),axis=0)
skel_bones[0][3]=np.delete(skel_bones[0][3],(skel_bones[0][3].shape[1]-1),axis=1)
skel_bones[0][3]=np.delete(skel_bones[0][3],(0),axis=1)
#    
full_threshG=np.delete(full_threshG,(full_threshG.shape[0]-1),axis=0)
full_threshG=np.delete(full_threshG,(0),axis=0)
full_threshG=np.delete(full_threshG,(full_threshG.shape[1]-1),axis=1)
full_threshG=np.delete(full_threshG,(0),axis=1)


diameter_value=skel_bones[0][3].copy()
diameter_value=skel_bones[0][3].copy()
diameter_holder = skel_bones[0][2].copy()
end_time = time.time()
plot_image(skel_bones[0][2],'skeleton','gray','png','no',path)
plot_color_image(skel_bones[0][3],'ratio analysis capillary distance','jet','png','yes',path)
print("total time taken this loop: ", end_time - start_time)


full_threshG_3channel=np.dstack((full_threshG,full_threshG,full_threshG))

#%%
#a=np.array([[0, 1, 0], [0, 0, 0]],np.float64)*255
top=[]
bottom=[]
pad_diameter_holder=np.pad(diameter_holder, ((1,1),(1,1)), 'constant') # add rows and columns of zero to pad

for i in range(1,470,1):
    for j in range(1,645,1):
        bbox=pad_diameter_holder[i-1:i+1,j-1:j+2]
        if bbox[0,1]==255 and sum(sum(bbox))==255:
            bottom.append([i-1,j])
        if bbox[1,1]==255 and sum(sum(bbox))==255:
            top.append([i-1,j])

'''
full_threshGcp=full_threshG.copy()

for i in range(0,hcrop-2):
    for j in range(0,wcrop-2):
        if skel_bones[2][i,j] == 255:
            full_threshGcp[i,j,0]=255.0
            full_threshGcp[i,j,1]=0.0
            full_threshGcp[i,j,2]=0.0
plot_cv2(full_threshGcp,'viz_center','png')

kernel = np.ones((3,3),np.uint8)

diameter_holder_buff=cv2.dilate(diameter_holder,kernel,1)
plot_cv2(diameter_holder_buff,'diameter_holder_buff','png')
plot_cv2(diameter_holder,'diameter_holder','png')
'''
#%%
#kernel = np.ones((3,3),np.uint8)r
dilate_diameter_holder=cv2.dilate(diameter_holder,kernel,1)
dilate_diameter_holder=dilate_diameter_holder.astype('uint8')
diameter_holder=diameter_holder.astype('uint8')

zeros=np.zeros_like(dilate_diameter_holder)

__, dils, __ = cv2.findContours(diameter_holder.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
dils = sorted(dils, key = cv2.contourArea, reverse = False)
area=[]
sq=[]
#ones_mat = np.ones((hcrop,wcrop),np.uint8)
for i, sq_id in enumerate(dils):
    sq_id=dils[i].reshape(-1,2)#, cnt in enumerate(cnts):
    area.append(cv2.drawContours(zeros.copy(),dils[i],i,color=255,thickness=1))
    sq_id[:,[0, 1]] = sq_id[:,[1, 0]]
    sq.append([sq_id[:,[0]], sq_id[:,[1]]])

#plt.close('all')
zeros[sq[0][0],sq[0][1]]=255
