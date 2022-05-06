"""
A Streamlit app making face and mask detection
"""
# Core Pkgs
import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# serialized face detector model from disk
MODEL_PATH = "./mask_detector.model"
PROTOTXT_PATH = "./face_detector/deploy.prototxt"
WEIGHTS_PATH = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"

CASCADE_PATH = "./cascades/"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH
        + 'haarcascade_frontalface_default.xml')

color_cadre = (0, 0, 255)  # La couleur du carré qui entoure le visage détecté
color_with = (0, 155, 0)  # La couleur du carré qui entoure le visage détecté
color_without = (255, 0, 0)  # La couleur du carré qui entoure le visage détecté
BOX_H = 20  # la hauteur du bandeau
color_back = (0, 0, 0)  # La couleur de fond du bandeau
color_text = (255, 255, 255)  # La couleur du texte


# load our serialized face detector model from disk
if 'faceNet' not in st.session_state:
    st.session_state.faceNet = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)

# load the face mask detector model from disk
if 'maskNet' not in st.session_state:
    st.session_state.maskNet = load_model(MODEL_PATH)

# Initialise the report if not present
if "personnes" not in st.session_state :
    st.session_state.personnes = pd.DataFrame(
            columns= ['Personne', 'Date', 'Heure', 'Masque', 'Précision'])

def add_count(label, state="?", accuracy=None):
    """Add a person label in the personnes counts

    :label: Label of the person
    :returns: None

    """
    now = datetime.datetime.now()
    st.session_state.personnes.loc[now] = [label, now.strftime("%x"),
            now.strftime("%X"), state, accuracy]

def detect_faces(our_image):
    """Detect faces and tag them

    :our_image: image to analyse
    :return: taged image and faces
    """
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (start_x, start_y, stop_x, stop_y) in faces:
        cv2.rectangle(img, (start_x, start_y),
                (start_x+stop_x, start_y+stop_y), color_cadre)
        cv2.rectangle(img, (start_x, start_y - BOX_H),
                (start_x + stop_x, start_y), color_cadre, -1)
        label = f"Personne {len(st.session_state.personnes)+1}"
        cv2.putText(img, label, (start_x, start_y - int(BOX_H/3)),
                cv2.FONT_HERSHEY_SIMPLEX, .5, color_text)
        add_count(label)
    return img,faces


def detect_and_predict_mask(frame, face_net=st.session_state.faceNet, mask_net=st.session_state.maskNet):
    """Detect faces and tag them with or without mask

    :frame: image to analyse
    :face_net: the prediction model to find faces
    :mask_net: the prediction model to find masks
    :return: taged image and predictions
    """
    # convert frame to a numpy array
    frame = np.array(frame.convert('RGB'))
    # resize frame
    #frame = imutils.resize(frame, width=400)
    # grab the dimensions of the frame and then construct a blob
    # from it
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

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
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(width - 1, end_x), min(height - 1, end_y))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (start_x, start_y, end_x, end_y) = box
        (with_mask, without_mask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        state = "Mask" if with_mask > without_mask else "No Mask"
        color = color_with if state == "Mask" else color_without

        # include the probability in the label
        name = f"Personne {len(st.session_state.personnes)+1}"
        accuracy = max(with_mask, without_mask) * 100
        label = f"{accuracy:.2f}%"

        # display the label and bounding box rectangle on the output
        # frame

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 1)
        cv2.rectangle(frame, (start_x, start_y - BOX_H),
                (end_x, start_y), color, -1)
        cv2.putText(frame, name, (start_x, start_y - int(BOX_H/3)),
                cv2.FONT_HERSHEY_SIMPLEX, .5, color_text, 1)
        cv2.rectangle(frame, (start_x, end_y - BOX_H),
                (end_x, end_y), color, -1)
        cv2.putText(frame, label, (start_x, end_y - int(BOX_H/3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_text, 1)

        # add personne to counts
        add_count(name, state, accuracy)

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

            st.success(f"Found {len(result_faces)} faces")

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

            st.success(f"Found {len(result_faces)} faces")



    elif choice == 'Report':
        st.subheader('Report')
        st.write(st.session_state.personnes)

    elif choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown("[Rom1deToyes](https://www.github.com/Rom1deTroyes/)")
        st.markdown("With code by [JCharisTech](https://www.jcharistech.com/)")
        st.text("Romain Heller (@Rom1deTroyes)")



if __name__ == '__main__':
    main()
