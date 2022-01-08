from tkinter import *
from cv2 import *
import numpy as np

root = Tk()  # base window
root.geometry("350x75")  # dimension of root window
root.title("Color Detection")

"""functions to be called when buttons in root is hit"""


# for image button
def onButtonClicked():
    dialog = Tk()  # dialog box to accept directory of an image
    dialog.geometry("200x100")  # size of dialog box

    Label(dialog, text="Enter complete directory").pack()  # text to help user

    txt = StringVar()  # to use the entry of entry box
    ent = Entry(dialog, textvariable=txt)
    ent.pack()  # entry box

    def callVideoColour():  # to call imageColour() function and pass the content of the entry box as the argument
        """.get() returns the current text in the entry box when textvariable parameter in Entry is set"""
        detectColour(ent.get())

    # Basic buttons to navigate
    buttons = {"Enter": callVideoColour, "Back": dialog.destroy}

    for nameOfbutton, actionOnClick in buttons.items():
        Button(dialog, text=nameOfbutton, command=actionOnClick).pack()


# for live video
def onUseWebcamClicked():
    """ function to access webcam and process frame by frame
    detectColour() accepts a parameter which is the id of the camera
     id is 0 if only one camera is used"""
    detectColour(0)


def detectColour(camIdOrDir):
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
    vid = cv2.VideoCapture(camIdOrDir)  # captures video from directory or directly from the camera of particular id
    while True:
        """ => reads each frame of the video as an image.
            => imageFrame will have each frame of the video and
            => '_' will have the boolean returned by the read() to ensure 
                each frame is being red properly(as we are not using it,it has named by underscore)"""
        _, imageFrame = vid.read()  # _ -> boolean value, it signifies if the read function was able to read a frame of video
        imageHSV = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)  # BGR to HSV

        # red mask
        red_lower = np.array([0, 156, 137], np.uint8)  # uint8 -> Unsigned integer -> value range -> 0 to 255
        red_upper = np.array([7, 238, 243], np.uint8)
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
        white_lower = np.array([101, 37, 209], np.uint8)
        white_upper = np.array([127, 82, 255], np.uint8)
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
            if area > 500:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Red", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for green
        contourGreen, hierarchy = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourGreen:
            area = cv2.contourArea(ctr)
            if area > 300:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for blue
        contourBlue, hierarchy = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourBlue:
            area = cv2.contourArea(ctr)
            if area > 100:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for Yellow
        contourYellow, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourYellow:
            area = cv2.contourArea(ctr)
            if area > 300:
                x, y, h, w = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Yellow", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for orange
        contourOrange, hierarchy = cv2.findContours(orange_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in contourOrange:
            area = cv2.contourArea(ctr)
            if area > 300:
                x, y, w, h = cv2.boundingRect(ctr)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), [0, 0, 0])
                cv2.putText(imageFrame, "Orange", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])

        # contour for white
        contourWhite, hierarchy = cv2.findContours(white_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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


# basic buttons to navigate on root
rootKeys = {"Add content": onButtonClicked,
            "Use webcam": onUseWebcamClicked,
            "Exit": root.destroy}

for text, command in rootKeys.items():
    Button(root, text=text, command=command).pack()

root.mainloop()
