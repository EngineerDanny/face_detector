import cv2


# Import an Image
img_str = "./dummy_faces/face1.jpeg"

# Classifier file contains a pre-modelled algorithm
classifier_file = "frontal_face.xml"

img = cv2.imread(img_str)
face_classifier = cv2.CascadeClassifier(classifier_file)


# convert image to gray-scaled image for easier identification
grayed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates = face_classifier.detectMultiScale(grayed_img)

for (x, y, w, h) in face_coordinates:
    # Drawing the rectangle on the image or images if they are many
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 5)

# showing the face detector app
cv2.imshow('Danny Face Detector', img)

cv2.waitKey()

print("Program finished successfully")
