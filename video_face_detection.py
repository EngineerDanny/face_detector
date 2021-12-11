import cv2


# Import a Video
video_str = "video.mkv"

# Classifier file contains a pre-modelled algorithm
classifier_file = "frontal_face.xml"

video = cv2.VideoCapture(video_str)

face_classifier = cv2.CascadeClassifier(classifier_file)


while True:
    # Capture the frames of the video
    (success, frame) = video.read()

    # convert frame to gray
    grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
