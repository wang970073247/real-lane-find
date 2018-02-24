cap = cv2.VideoCapture(1)
while(1):
    # get a frame

    ret, frame = cap.read()
    #print(cap)
    if ret:
        result_img = process_image(frame)
        cv2.imshow("new_frame", result_img)
    #cv2.imshow("capture", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
