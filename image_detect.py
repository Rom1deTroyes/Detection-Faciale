import cv2 

cascade_path =  "./cascades/haarcascade_frontalface_default.xml"
img_path = "./images/001.jpg"

color_cadre = (127, 127, 127)  # La couleur du carré qui entoure le visage détecté
color_back = (0, 0, 0)  # La couleur de fond du bandeau
color_text = (255, 255, 255)  # La couleur du texte
hauteur_bandeau = 20  # la hauteur du bandeau

src = cv2.imread(img_path)
cascade = cv2.CascadeClassifier(cascade_path)
#gray = cv2.cvtColor(src,cv2.cv2.COLOR_BAYER_BG2GRAY)
#rect = cascade.detectMultiScale(gray)
rect = cascade.detectMultiScale(src[:,:,0])

nb_personnes = 0

if len(rect) > 0:
    for x, y, w, h in rect:
        
        nb_personnes += 1
        
        cv2.rectangle(src, (x, y), (x+w, y+h), color_cadre)
        cv2.rectangle(src, (x, y - hauteur_bandeau), (x + w, y), color_back, -1)

        cv2.putText(src, f'Personne {nb_personnes}', (x, y - int(hauteur_bandeau/3)), cv2.FONT_HERSHEY_SIMPLEX, .5, color_text)

    cv2.imshow('detected', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
