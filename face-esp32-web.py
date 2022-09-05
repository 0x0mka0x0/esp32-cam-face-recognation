import cv2
import urllib.request
import numpy as np
import face_recognition as fr
def nothing(x):
    pass

# loading the selected face file
image = fr.load_image_file("image.jpg")
image_face_encoding = fr.face_encodings(image)[0]
# giving a name to face
known_face_encodings = [image_face_encoding]
known_face_names = ["image"]
#the url of esp32 web cam
url='http://192.168.1.102/cam-lo.jpg'

cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)

    #recognation
    rgb_frame= frame[:,:,::-1]
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)
    cv2.imshow("live transmission", frame)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
    
cv2.destroyAllWindows()