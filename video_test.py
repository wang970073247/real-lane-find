import cv2
import numpy as np

cap = cv2.VideoCapture(1)
print(cap)
while(1):
    # get a frame
    
    ret, frame = cap.read()
    #print(cap)
    if ret:
        #print(cap)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #new_frame = np.hstack([frame, frame])
    # show a frame
        cv2.imshow("new_frame", frame)
    #cv2.imshow("capture", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
