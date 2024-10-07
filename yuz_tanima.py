import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier("yuz.xml")
cap = cv2.VideoCapture(0)
i = 0
img = ""
x = 0
y = 0
w = 0
h = 0

while True:
    
    ret, frame = cap.read() 
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
            
        for (x,y,w,h) in face_rect:
            print(i)
            i += 1
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x = x
            y = y
            w = w
            h = h
            cv2.putText(frame, "merhaba" ,(20,20),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),1)
        cv2.imshow("face detect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break
    
cv2.imwrite("gorsel.jpg", img)

cap.release()
cv2.destroyAllWindows()