import cv2


# Classifier file contains a pre-modelled algorithm
classifier_file = "frontal_face.xml"

# camera index, test to check which of your cameras available you want to use
# Could be 0,1,2,etc
camera_index = 1
video = cv2.VideoCapture(camera_index)

face_classifier = cv2.CascadeClassifier(classifier_file)


while True:
    # Capture the frames of the video
    (success, frame) = video.read()

    if success == True:
        # convert frame to gray
        grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    face_coordinates = face_classifier.detectMultiScale(grayed_frame)

    for (x, y, w, h) in face_coordinates:
        # Drawing the rectangle on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # showing the face detector app
    cv2.imshow('Danny Face Detector', frame)

    # wait for Q key which has its ascii code as 81 or 113
    # stop the program
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

video.release()
print("Program finished successfully")
