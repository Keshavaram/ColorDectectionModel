import cv2
import numpy as np

video = cv2.VideoCapture(0)
while True:
    _, imageFrame = video.read()
    imageHSV = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)  # BGR to HSV Hue , saturation and value

    # red mask
    red_lower = np.array([0, 193, 164], np.uint8)
    red_upper = np.array([10, 255, 255], np.uint8)
    red_mask = cv2.inRange(imageHSV, red_lower, red_upper)

    # green mask
    green_lower = np.array([69, 133, 46], np.uint8)
    green_upper = np.array([80, 255, 255], np.uint8)
    green_mask = cv2.inRange(imageHSV, green_lower, green_upper)

    # blue mask
    blue_lower = np.array([95, 158, 132], np.uint8)
    blue_upper = np.array([127, 255, 255], np.uint8)
    blue_mask = cv2.inRange(imageHSV, blue_lower, blue_upper)

    # yellow mask
    yellow_lower = np.array([24, 88, 156], np.uint8)
    yellow_upper = np.array([63, 169, 255], np.uint8)
    yellow_mask = cv2.inRange(imageHSV, yellow_lower, yellow_upper)

    # orange mask
    orange_lower = np.array([12, 183, 194], np.uint8)
    orange_upper = np.array([29, 255, 255], np.uint8)
    orange_mask = cv2.inRange(imageHSV, orange_lower, orange_upper)

    # white mask
    white_lower = np.array([0,0,167], np.uint8)
    white_upper = np.array([95,18,255], np.uint8)
    white_mask = cv2.inRange(imageHSV, white_lower, white_upper)

    # contour for red
    contourRed, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourRed:
        area = cv2.contourArea(ctr)
        if area > 500:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Red", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for green
    contourGreen, b = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourGreen:
        area = cv2.contourArea(ctr)
        if area > 300:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for blue
    contourBlue, c = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourBlue:
        area = cv2.contourArea(ctr)
        if area > 100:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for Yellow
    contourYellow, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourYellow:
        area = cv2.contourArea(ctr)
        if area > 300:
            x, y, h, w = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Yellow", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for orange
    contourOrange, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourOrange:
        area = cv2.contourArea(ctr)
        if area > 300:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Orange", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for white
    contourWhite, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourWhite:
        area = cv2.contourArea(ctr)
        if area > 300 and area < 1000:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "White", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    cv2.imshow("Colour detection", imageFrame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
