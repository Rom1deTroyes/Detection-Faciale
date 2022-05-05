# Core Pkgs
import cv2
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


color_cadre = (0, 0, 255)  # La couleur du carré qui entoure le visage détecté
color_back = (0, 0, 0)  # La couleur de fond du bandeau
color_text = (255, 255, 255)  # La couleur du texte
hauteur_bandeau = 20  # la hauteur du bandeau

if "personnes" not in st.session_state :
    st.session_state.personnes = pd.DataFrame(columns= ['Personne', 'Date', 'Heure'])

cascade_path = "./cascades/"

face_cascade = cv2.CascadeClassifier(cascade_path+'haarcascade_frontalface_default.xml')


def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    nb_personnes = len(st.session_state.personnes)

    for (x, y, w, h) in faces:
        nb_personnes += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), color_cadre)
        cv2.rectangle(img, (x, y - hauteur_bandeau), (x + w, y), color_back, -1)
        label = 'Personne {}'.format(nb_personnes)
        cv2.putText(img, label, (x, y - int(hauteur_bandeau/3)), cv2.FONT_HERSHEY_SIMPLEX, .5, color_text)
        t = datetime.datetime.now()
        st.session_state.personnes.loc[t] = [label, t.strftime("%x"), t.strftime("%X")]
    return img,faces


@st.cache
def load_image(img):
    """Load an image from path

    :img: Path to the image file
    :returns: an Image object

    """
    image = Image.open(img)
    return image


def main():
    """Face Detection App"""

    st.title("Face Detection App")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Detection", "Report", "About"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    if choice == 'Detection':
        st.subheader("Face Detection")

        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        our_image = load_image('./images/001.jpg')

        if image_file is not None:
            our_image = load_image(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)

        else:
            st.image(our_image,width=300)


        # Face Detection
        if st.button("Process"):
            result_img,result_faces = detect_faces(our_image)
            st.image(result_img)

            st.success("Found {} faces".format(len(result_faces)))


    elif choice == 'Report':
        st.subheader('Report')
        st.write(st.session_state.personnes)

    elif choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown("Built with Streamlit by [Rom1deToyes](https://www.github.com/Rom1deTroyes/)")
        st.markdown("With code by [JCharisTech](https://www.jcharistech.com/)")
        st.text("Romain Heller (@Rom1deTroyes)")



if __name__ == '__main__':
    main()
