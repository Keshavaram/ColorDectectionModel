import cv2
from tkinter import *
from cv2 import *
import numpy as np

root = Tk()  # base window
root.geometry("350x100")  # dimension of root window
root.title("Color Detection")

"""functions to be called when buttons in root is hit"""


# for image button
def imgCommand():
    imgDialog = Tk()  # dialog box to accept directory of an image
    imgDialog.geometry("200x100")  # size of dialog box

    l1 = Label(imgDialog, text="Enter complete directory")  # text to help user

    txt = StringVar()  # to use the entry of entry box
    ent = Entry(imgDialog, textvariable=txt)  # entry box

    def callImageColour():  # to call imageColour() function and pass the content of the entry box as the argument
        imageColour(ent.get())  # .get() returns the current text in the entry box

    # Basic buttons to navigate
    backButton = Button(imgDialog, text="Back", command=imgDialog.destroy)
    enterButton = Button(imgDialog, text="Enter", command=callImageColour)

    # aligning every UI entity
    l1.pack()
    ent.pack()
    enterButton.pack()
    backButton.pack()


# for video button
def startVideo():
    EnterDir = Tk()  # dialog box
    EnterDir.geometry("200x100")  # size of dialog box

    l1 = Label(EnterDir, text="Enter complete directory")  # text to instruct the user

    txt = StringVar()  # to get the content of entry box
    ent = Entry(EnterDir, textvariable=txt)  # entry box

    def callVideoColour():  # to call videoColour() with contents of the entry box as the argument
        videoColour(ent.get())  # .get() returns the current text in the entry box

    # basic buttons to navigate
    enter = Button(EnterDir, text="Enter", command=callVideoColour)
    back = Button(EnterDir, text="Back", command=EnterDir.destroy)

    # aligning every UI entity
    l1.pack()
    ent.pack()
    enter.pack()
    back.pack()


# for live video
def callLiveColour():
    """ function to access webcam and process frame by frame
    videoColour() accepts a parameter which is the id of the camera
     id is 0 if only one camera is used"""
    videoColour(0)


# function to process an image
def imageColour(directory):
    """basic logic:
            image is to be converted from BGR format to HSV
                BGR->Blue,Green,Red
                HSV->Hue,Saturation,Value
            using HSV, a mask for each colour is to be created
            Mask is created using ColorModel.py
            Calculate the area of contour of each colour,because very small areas have to be ignored
            A rectangle is drawn around the contour
            A text is included to notify the colour of each contour
            """
    imageFrame = cv2.imread(directory)  # Reading image from 'directory'(path of image)
    imageHSV = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)  # Converting each frame from BGR to HSV

    # red mask
    """lower and upper limits of HUE,SATURATION and VALUE are stored as an array using numpy.array
       HSV values are unique to a particular colour"""
    red_lower = np.array([0, 193, 164], np.uint8)  # format of array -> [Hue,Saturation,Value] min value
    red_upper = np.array([10, 255, 255], np.uint8)  # format of array -> [Hue,Saturation,Value] max value
    """inRange() function will create mask of a colour w.r.to the values of HSV"""
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
    white_lower = np.array([0, 0, 167], np.uint8)
    white_upper = np.array([95, 18, 255], np.uint8)
    white_mask = cv2.inRange(imageHSV, white_lower, white_upper)

    # contour for red
    """contourRed -> will have the info about the contour that is currently found.
       hierarchy -> will have info about image topology, numbers fo contours found(Optional output).
       RetrievalMode -> RETR.LIST: finds contours and doesn't bother in calculating hierarchy.
       RetrievalMethod -> CHAIN_APPROX_SIMPLE:gives only the end points of the contour in a
                          particular direction i.e vertical, horizontal and diagonal.
                          CHAIN_APPROX_NONE:gives all the points on the boundary of a contour in a particular direction
                          i.e vertical, horizontal and diagonal."""
    contourRed, hierarchy = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourRed:
        area = cv2.contourArea(ctr)
        if area > 500:  # to remove unnecessary detection due to noise in the images
            x, y, w, h = cv2.boundingRect(ctr)  # coordinate of starting point and diagonally opposite point
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])  # drawing rectangle around contour
            cv2.putText(imageFrame, "Red", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])  # Text to indicate colour of a particular contour

    # contour for green
    contourGreen, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourGreen:
        area = cv2.contourArea(ctr)
        if area > 300:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for blue
    contourBlue, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourBlue:
        area = cv2.contourArea(ctr)
        if area > 100:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for Yellow
    contourYellow, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourYellow:
        area = cv2.contourArea(ctr)
        if area > 300:
            x, y, h, w = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Yellow", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for orange
    contourOrange, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourOrange:
        area = cv2.contourArea(ctr)
        if area > 300:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "Orange", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    # contour for white
    contourWhite, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ctr in contourWhite:
        area = cv2.contourArea(ctr)
        if 300 < area < 1000:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
            cv2.putText(imageFrame, "White", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

    cv2.imshow("Colour detection", imageFrame)

    cv2.waitKey(0)


# function to process video or live footage
def videoColour(camIdOrDir):
    vid = cv2.VideoCapture(camIdOrDir)  # captures video from directory or directly from the camera of particular id
    while True:
        """ => reads each frame of the video aa an image.
            => imageFrame will have each frame of the video and
            => '_' will have the boolean returned by the read() to ensure 
                each frame is being red properly(as we are not using it,it has named by underscore)"""
        _, imageFrame = vid.read()
        imageHSV = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)  # BGR to HSV

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
        white_lower = np.array([0, 0, 167], np.uint8)
        white_upper = np.array([95, 18, 255], np.uint8)
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
        contourGreen, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourGreen:
            area = cv2.contourArea(ctr)
            if area > 300:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for blue
        contourBlue, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourBlue:
            area = cv2.contourArea(ctr)
            if area > 100:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for Yellow
        contourYellow, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourYellow:
            area = cv2.contourArea(ctr)
            if area > 300:
                x, y, h, w = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Yellow", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for orange
        contourOrange, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourOrange:
            area = cv2.contourArea(ctr)
            if area > 300:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Orange", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for white
        contourWhite, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourWhite:
            area = cv2.contourArea(ctr)
            if 300 < area < 1000:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "White", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        cv2.imshow("Colour detection", imageFrame)

        """=> each frame is displayed for 1ms.
           => when user hits 'q',code stops"""
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# basic buttons to navigate
image = Button(root, text="Add Image", command=imgCommand)
video = Button(root, text="Add a video clip", command=startVideo)
live = Button(root, text="Use webcam", command=callLiveColour)
end = Button(root, text="Exit", command=root.destroy)

image.pack()
video.pack()
live.pack()
end.pack(side="bottom")

root.mainloop()
