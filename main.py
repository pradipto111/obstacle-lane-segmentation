import cv2
import numpy as np 
import math

cap = cv2.VideoCapture("./output.mov")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) #kernel for erosion/dilation operations

B = 0.005
G = 0.005
R = 0.990

while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        blue = frame[:,:,0]
        blue = cv2.GaussianBlur(blue,(7,7),0)
        green = frame[:,:,1]
        green = cv2.GaussianBlur(green,(7,7),0)
        red = frame[:,:,2]
        red = cv2.GaussianBlur(red,(7,7),0)
        transform = (B*blue)+(G*green)+(R*red)
        transform = transform.astype(np.uint8)
        #transform = cv2.GaussianBlur(transform,(11,11),0)
        ret, thresh = cv2.threshold(transform, 50,255,cv2.THRESH_BINARY)
        ret,red = cv2.threshold(red,200,255,cv2.THRESH_BINARY_INV)
        red = cv2.dilate(red,kernel,iterations=1)
        thresh[red == 0] = 0
        thresh = cv2.dilate(thresh,kernel,iterations=2)
        thresh = cv2.erode(thresh,kernel,iterations=2)
        thresh = thresh/255
        mask = np.full(frame.shape[:2],1, dtype = np.uint8)
        mask[np.logical_and(frame[:,:,1]<=frame[:,:,0],frame[:,:,1]<=frame[:,:,2])] = 0
        thresh[mask==0] = 0 #OBJECT MASK
        frame[:,:,0] = frame[:,:,0]*thresh # Masking
        frame[:,:,1] = frame[:,:,1]*thresh # the current frame
        frame[:,:,2] = frame[:,:,2]*thresh # of the video


        blue[thresh ==0] = 0
        ret,blue = cv2.threshold(blue,140,255,cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(blue,1,np.pi/180,200, minLineLength=30, maxLineGap=20)   #detect lines using Hough transform
        if lines is not None:   #drawing the lines
            for line in lines[:40]:
                x1,y1,x2,y2 = line[0]
                if y1>200 and y1<1020:
                    cv2.line(frame,(x1,y1), (x2,y2), (255,0,255),20)
                if y2>200 and y2<1020:
                    cv2.line(frame,(x1,y1), (x2,y2), (255,0,255),20)


        cv2.imshow("output",frame)
    else:
        cap.release()
        cv2.destroyAllWindows()

    cv2.waitKey(1)

