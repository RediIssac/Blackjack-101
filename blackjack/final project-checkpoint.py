#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2

cap = cv2.VideoCapture('south.mp4')

ret, frame = cap.read()

print('ret =', ret, 'W =', frame.shape[1], 'H =', frame.shape[0], 'channel =', frame.shape[2])


FPS= 20.0
FrameSize=(frame.shape[1], frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('Video_output.avi', fourcc, FPS, FrameSize, 0)


# initalizing an array for saving centroids
big_data = []
centroids = []

def track_frame(size_back):
   
    #dist = sqrt(pow((x2-x1),2) + pow((y2-y1),2))
    
    if len(centroids) < size_back:
        return
    print(len(centroids)-size_back)
    
    # we need to iterate starting from the last frame back till the size given
    back = len(centroids)-size_back
    #for i in range(back, len(centroids)):
   
    for c in centroids[:size_back]:
         # put text and highlight the center
    
        cv2.circle(frame, (c[0], c[1]), 5, (255, 255, 255), -1)
    
        cv2.putText(frame, "c", (c[0] - 25, c[1] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    print("hey",centroids[0][0])
    #print(centroids[1])
    
    # I was trying to calculate the slope here
    x2 = centroids[len(centroids)-1][0]
    y2 = centroids[len(centroids)-1][1]
    x1 = centroids[back][0]
    y1 = centroids[back][1]
    
    
    c_y = y2-y1
    c_x = x2 - x1
    slope = (y2-y1)/(x2-x1)
    # base on the sign of the slope predicte where the direction would be
    if (slope >= 0):
        if(c_y >=0 & c_x >=0):
            print("direction is right top")
        else:
            print("left down")
    else:
        if(c_y >=0 & c_x <= 0):
            print("top left")
        elif(c_y <=0 & c_x >= 0):
            print("right_down")
        
#trackedFrames = numpy.zeros((720,1280))



def process_frame(frame):
    # step 1 RGB to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = gray
    # step 2: filter video
    blur = cv2.GaussianBlur(frame, (15,15), 0) 
    # find delta 
    image_diff = frame - blur
    # find binary frames by thresholding
    retval, thresh = cv2.threshold(image_diff, 10, 255, cv2.THRESH_BINARY)
    frame = thresh
    big_data.append(frame)
    
     # calculate moments of binary image
    M = cv2.moments(thresh)
 
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = [cX, cY]
    centroids.append(centroid)
    
   
    return frame
    
    

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break 
    frame = process_frame(frame)
    # should be called once the card is detected and the hand is moving back
    track_frame(100)
    # Save the video
    out.write(frame)
    
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
out.release()
cv2.destroyAllWindows()


    # appending all the centroids for later tracking purpose

 #MJPG-encoded AVI as an input


# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:




