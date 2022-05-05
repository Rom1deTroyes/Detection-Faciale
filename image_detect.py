import cv2

cascade_path =  "./cascades/haarcascade_frontalface_default.xml"
img_path = "./images/001.jpg"
# color = (255, 255, 255)  # La couleur du carré qui entoure le visage détecté
color = (0, 0, 0)  # La couleur du carré qui entoure le visage détecté
hauteur_bandeau = 20

src = cv2.imread(img_path,0)
gray = cv2.cvtColor(src,cv2.cv2.COLOR_BAYER_BG2GRAY)
cascade = cv2.CascadeClassifier(cascade_path)
rect = cascade.detectMultiScale(gray)
if len(rect) > 0:
    for x, y, w, h in rect:
        cv2.rectangle(src, (x, y), (x+w, y+h), color)
        cv2.rectangle(src, (x, y - hauteur_bandeau), (x + w, y), color, -1)

        cv2.imshow('detected', src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
