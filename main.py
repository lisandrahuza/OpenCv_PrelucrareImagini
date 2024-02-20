import pandas as pd
import numpy as np
from glob import glob
import cv2
from matplotlib import pyplot as plt


#img_cv2= plt.imread('C:\\Users\\huzal\\PycharmProjects\\opencv_project\\cat\\cat1.jpg')

#cv2.imshow("image",img_cv2)
#cv2.waitKey(0)

# cap=cv2.VideoCapture('MicaSirenă.mp4')
#
# if(cap.isOpened()==False):
#     print("Error openning video scream or file")
#
# while(cap.isOpened()):
#     ret, frame=cap.read()
#     if ret==True:
#         cv2.imshow('frame',frame)
#
#         if cv2.waitKey(25)==ord('q'):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()


# cap=cv2.VideoCapture(3)
# ret,frame=cap.read()
# print(ret)
# print(frame)
# plt.imshow(frame)
# plt.show()
# cap.release()
#
# def take_photo():
#     cap=cv2.VideoCapture(3)
#     ret,frame=cap.read()
#     cv2.imwrite('webCamePhoto.jpg',frame)
#     cap.release();
#take_photo()

#video webcamera

vid=cv2.VideoCapture(0)

#creat a background object
obj=cv2.createBackgroundSubtractorMOG2(history=2)
kernel=np.ones((3,3),np.uint8)
kernel2=None

while(True):
    ret,frame1=vid.read()
    if not ret:
        break
    frame=cv2.flip(frame1,1)
    # width=int(vid.get(3))
    # height=int(vid.get(4))
    #
    # image=np.zeros(frame.shape,np.uint8)
    # smaller_frame=cv2.resize(frame,(0,0),fx=1.5,fy=1.5)
    #frame=cv2.

    fgmask=obj.apply(frame)
    _,fgmask=cv2.threshold(fgmask,20,255,cv2.THRESH_BINARY)
    fgmask=cv2.erode(fgmask,kernel,iterations=1)
    fgmask=cv2.dilate(fgmask,kernel2,iterations=6)

    #detectarea contururilor
    countors,_=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    frameCopy=frame.copy()

    #bucla in interiorul contururilor si cautarea cel mari

    for cnt in countors:
        if cv2.contourArea(cnt)>20000:
            #coordonate arie
            x,y,width,height=cv2.boundingRect(cnt)
            #trasare dreptunghi in jurul ariei
            cv2.rectangle(frameCopy,(x,y),(x+width,x+height),(0,0,255),2)
            #scrierea unui text langa obiect
            cv2.putText(frameCopy,"Obiect detectat",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1,cv2.LINE_AA)

    forground=cv2.bitwise_and(frame,frame,mask=fgmask)

    stacked=np.hstack((frame,forground,frameCopy))
    cv2.imshow("stacked",cv2.resize(stacked,None,fx=0.5,fy=0.5))

    # cv2.imshow("forground",forground)
    # cv2.imshow("frameCopy",frameCopy)
    # cv2.imshow("fgmask",fgmask)
    #cv2.imshow("img",frame)
    # cv2.imshow('frame',smaller_frame)

    if cv2.waitKey(1)& 0xFF==ord('q'):
        break

vid.release()
cv2.destroyAllWindows()