import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("Resources/face_cascade.xml")
eyeCascade = cv2.CascadeClassifier("Resources/eye_cascade.xml")
# img = cv2.imread("Resources/friends.jpg")
# imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# faces = faceCascade.detectMultiScale(imgGrey,1.1,7)
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#
#
# cv2.imshow("image",img)
# cv2.waitKey(0)


vid = cv2.VideoCapture(0)

while True:
    flag, img_flip =  vid.read()
    img = cv2.flip(img_flip,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.3,7)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        #cv2.imshow("roi_grey", roi_gray)
        #cv2.imshow("roi_color",roi_color)

    eyes = eyeCascade.detectMultiScale(roi_gray)
    for (a,b,q,r) in eyes:
        cv2.rectangle(roi_color,(a,b),(a+q,b+r),(0,255,0),1)

    cv2.imshow("Face & Eye Detection",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

vid.release()
