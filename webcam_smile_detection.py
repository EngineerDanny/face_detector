import cv2


# Classifier file contains a pre-modelled algorithm
face_classifier_file = "frontal_face.xml"
smile_classifier_file = "frontal_smile.xml"

# camera index, test to check which of your cameras available you want to use
# Could be 0,1,2,etc
camera_index = 1
video = cv2.VideoCapture(camera_index)

face_classifier = cv2.CascadeClassifier(face_classifier_file)
smile_classifier = cv2.CascadeClassifier(smile_classifier_file)


while True:
    # Capture the frames of the video
    (success, frame) = video.read()

    if success != True:
        break

    # convert frame to gray
    gray_scaled_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_classifier.detectMultiScale(gray_scaled_face)

    for (x, y, w, h) in face_coordinates:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        # This takes a frame, so we gonna slice the frame from the face mainframe
        the_face = frame[y:y+h, x:x+w]

        # Detect the smile from the subframe
        smile_coordinates = smile_classifier.detectMultiScale(
            cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY), scaleFactor=1.7, minNeighbors=20)

        for (x_, y_, w_, h_) in smile_coordinates:
            # Drawing the rectangle on the smile
            cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), (255, 0, 0), 5)

    # showing the face detector app
    cv2.imshow('Danny Smile Detector', frame)

    # wait for Q key which has its ascii code as 81 or 113
    # stop the program
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

video.release()
print("Program finished successfully")
