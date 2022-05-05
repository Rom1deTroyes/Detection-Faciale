import cv2

cascade_path = "./cascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
color = (255, 255, 255) #La couleur du carré qui entoure le visage détecté
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
    if len(rect) > 0:
        for x, y, w, h in rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color)
    cv2.imshow('detected', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
