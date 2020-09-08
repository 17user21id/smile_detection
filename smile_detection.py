import cv2

#load data on face-frontal
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#load data on smile
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#grab video from webcam
cam = cv2.VideoCapture(0)

#loop forever over frame
while True:
    #read the current frame
    check_sucessful_frame , frame = cam.read()

    #if error abort
    if not  check_sucessful_frame:
        break

    #convert to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

    

    #detecting faces
    face = face_detector.detectMultiScale(frame_grayscale)

    #draw rectangle around the faces
    for (x ,y ,w ,h) in face:
        #draw rectangle around the face
        cv2.rectangle(frame , (x,y) , (x+w,y+h) , (100,200,200) , 4)

        #get the subframe (using numpy array slicing)
        the_face = frame[y:y+h , x:x+w]
        
        #convert to grayscale
        face_gray_scale = cv2.cvtColor(the_face , cv2.COLOR_BGR2GRAY)
        
        #detecting smile
        smile = smile_detector.detectMultiScale(face_gray_scale, scaleFactor=1.7, minNeighbors=20)

        for (x_ , y_ , w_ , h_) in smile:
            cv2.rectangle(the_face , (x_ , y_) , (x_ + w_ , y_ + h_) , (50 , 50 ,130) , 4)

        #label the face smiling
        if len(smile)>0:
            cv2.putText(frame, 'smiling' , (x, y+h+40) , fontScale = 3, fontFace = cv2.FONT_HERSHEY_DUPLEX , color = (255, 255, 255))

    #show the current frame
    cv2.imshow('smile detector' , frame)
    key = cv2.waitKey(1)

    #exit screen when Q is pressed
    if key == 81 or key == 113:
        break 

#release video capture object
cam.release()
#destroy all window
cv2.destroyAllWindows()