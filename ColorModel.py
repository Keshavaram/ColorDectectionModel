import cv2
import numpy as np

img = cv2.imread("Resources/rubik.jpg")


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 400, 260)
cv2.createTrackbar("Hue_min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue_max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat_min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat_max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val_min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val_max", "TrackBars", 255, 255, empty)

while True:
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue_min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue_max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat_min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat_max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val_min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val_max", "TrackBars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(HSV_img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Result", result)
    cv2.waitKey(1)
