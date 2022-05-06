# Core Pkgs
import cv2
import datetime
import imutils
import numpy as np
import pandas as pd
import streamlit as st
from imutils.video import VideoStream
from PIL import Image
from streamlit.state import session_state
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# serialized face detector model from disk
modelPath = "./mask_detector.model"
prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"

cascade_path = "./cascades/"
face_cascade = cv2.CascadeClassifier(cascade_path+'haarcascade_frontalface_default.xml')

color_cadre = (0, 0, 255)  # La couleur du carré qui entoure le visage détecté
hauteur_bandeau = 20  # la hauteur du bandeau
color_back = (0, 0, 0)  # La couleur de fond du bandeau
color_text = (255, 255, 255)  # La couleur du texte


# load our serialized face detector model from disk
if 'faceNet' not in st.session_state:
    st.session_state.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
if 'maskNet' not in st.session_state:
    st.session_state.maskNet = load_model(modelPath)

# Initialise the report if not present
if "personnes" not in st.session_state :
    st.session_state.personnes = pd.DataFrame(columns= ['Personne', 'Date', 'Heure'])


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


def detect_and_predict_mask(frame, faceNet=st.session_state.faceNet, maskNet=st.session_state.maskNet):
    # convert frame to a numpy array
    frame = np.array(frame.convert('RGB'))
    # resize frame
    frame = imutils.resize(frame, width=400)
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    #return (locs, preds)
    return (frame, faces)


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

    activities = ["Face Detection", "Mask Detection", "Report", "About"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    if choice == 'Face Detection':
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

    if choice == 'Mask Detection':
        st.subheader("Mask Detection")

        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        our_image = load_image('./images/001.jpg')

        if image_file is not None:
            our_image = load_image(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)

        else:
            st.image(our_image,width=300)


        # Face and Mask Detection
        if st.button("Process"):
            result_img,result_faces = detect_and_predict_mask(our_image)
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
