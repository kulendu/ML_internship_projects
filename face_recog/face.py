import cv2

# --------- for detecting the image and importing it from the haarcascade in .xml format ----------- 
detect = cv2.CascadeClassifier(r'/home/kulendu/Desktop/ML internship/face_recog/haarcascade_file_face_detection/haarcascade_frontalface_default.xml')

# ------------ capturing the video from the webcam ---------- 
cap = cv2.VideoCapture(0)

while True :
    ret, image = cap.read()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = detect.detectMultiScale(grey, 1.3,5)

    for(x,y,w,h) in face:
        cv2.rectangle(grey, (x,y), (x+w, y+h), (0,0,225),3)
        
    cv2.imshow('Detect Me', grey)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ------------ for quitting the camera press'Q' ---------- 
cap.release()
cv2.destroyAllWindows()