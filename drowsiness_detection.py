# importing required libraries
import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer

# video capture object
cap = cv2.VideoCapture(0)

mixer.init()
mixer.music.load("sound.mp3")

# detect Returns the default face detector and predict gets info about the face from landmarks file
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# get info about left and right eye from shape predictor
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

threshold = 0.30
frame_check = 20
flag = 0

# calculate eye open ration
def eye_aspect_ratio(eye):
    vert_distance1 = distance.euclidean(eye[1], eye[5])
    vert_distance2 = distance.euclidean(eye[2], eye[4])
    horizon_distance = distance.euclidean(eye[0], eye[3])
    e_a_r = (vert_distance1 + vert_distance2) / (2.0 * horizon_distance)
    return e_a_r

# creating a while loop that captures each frame
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    cv2.imshow("Frame", frame)
    # convert frame to grayscale and detect region of interest
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = detect(gray, 0)

    # iterate and calculate within roi
    for i in roi:
        shape = predict(gray, i)
        shape = face_utils.shape_to_np(shape)

        lefteye = shape[lStart:lEnd]
        righteye = shape[rStart:rEnd]

        leftE_A_R = eye_aspect_ratio(lefteye)
        rightE_A_R = eye_aspect_ratio(righteye)

        total_e_a_r = (leftE_A_R + rightE_A_R) / 2.0

        cv2.drawContours(frame, [lefteye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [righteye], -1, (0, 255, 0), 1)

        if total_e_a_r < threshold:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "---------ALERT!---------", (10, 30), cv2.FONT_ITALIC , 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Eyes CLOSED", (10, 325), cv2.FONT_ITALIC , 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
            cv2.putText(frame, "Eyes OPEN", (10, 30), cv2.FONT_ITALIC , 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
