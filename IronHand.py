import numpy as np
import cv2
import pyautogui as gui
import time
import math
from threading import Thread
from audiotranscripter import AudioTranscripter
import queue
import winsound
import pyaudio

gui.FAILSAFE = False;

# Switches for testing
click = True

# Important globals
initialCalibrate = 0
bgModel = None
backgroundVersion = True
rightHanded = True
bgCaptured = False
boxX = .5
boxY = 1
isRecording = False
if not rightHanded:
    boxX = .5
scrollMode = False
scrollBaseY = 0
numFing = 0
pos = tuple([0, 0])
detect = False
listenClick = False;
maxQueueSize = 5
inputQueue = queue.Queue(maxQueueSize)
numFours = 0
timeSinceFive = time.time() - 5
timeSinceThree = time.time()
at = None
recording = False;
# constant
threshold = 20
bgThreshold = 50
stillTime = 3
clickHold = .3
mouseSensitivity = 60
sizeThreshold = 8000
screenWidth = int(960 * 2)
screenHeight = int(540 * 2)
if backgroundVersion:
    screenWidth = int(720)
    screenHeight = int(480)
bound = 4
xBoundLow = int(screenWidth / bound)
yBoundLow = int(screenHeight / bound)
xBoundHigh = int((bound - 1) * screenWidth / bound)
yBoundHigh = int((bound - 1) * .85
                 * screenHeight / bound)
# movementSmooth < 1
movementSmooth = 0.0
# prior bounds
lowerBound = np.array([80, 50, 50])  # before 60,100,100
upperBound = np.array([120, 255, 255])  # before 180, 255, 255
blurValue = 41
font = cv2.FONT_HERSHEY_SIMPLEX
webcam = cv2.VideoCapture(0)
kernelOpen = np.ones((2, 2))

clickTimer = time.time()
clickReset = None;
prevx = 0
prevy = 0
newX = gui.position()[0]
newY = gui.position()[1]
oldX = 0
oldY = 0
numStill = 0

def set_right(b):
    global rightHanded
    rightHanded = b
def set_background(b):
    global backgroundVersion
    backgroundVersion = b
def set_at(a):
    global at
    at = a
def calculateHighestPoint(maxCont):
    extTop = tuple(maxCont[maxCont[:, :, 1].argmin()][0])
    return extTop


def calculateOneZero(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                if rightHanded:
                    if ((end[0] - calculateHighestPoint(res)[0]) ** 2 + (
                            end[1] - calculateHighestPoint(res)[1]) ** 2) ** (
                            1 / 2) > 50:
                        continue
                else:
                    if ((start[0] - calculateHighestPoint(res)[0]) ** 2 + (
                            start[1] - calculateHighestPoint(res)[1]) ** 2) ** (
                            1 / 2) > 50:
                        continue
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                if a < 50:
                    continue
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                # Angle debug
                # print(angle)

                if 15 / 18 * math.pi > angle >= 4 / 9 * math.pi:
                    cnt = 1
                    cv2.circle(drawing, start, 5, (0, 0, 255), -1)
                    cv2.circle(drawing, end, 5, (0, 255, 255), -1)
                    cv2.circle(drawing, far, 5, (0, 0, 255), -1)
                    # cv2.imshow('angles', drawing)
                    cv2.waitKey(1)
            return cnt
    return 0


def calculateFingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


def run():
    global numStill, clickTimer, listenClick, bgCaptured, bgModel
    while (webcam.isOpened()):
        moveThread = Thread(target=gui.moveTo, args=(newX, newY, movementSmooth))
        if -mouseSensitivity < newX - oldX < mouseSensitivity and -mouseSensitivity < newY - oldY < mouseSensitivity:
            numStill += 1
        else:
            numStill = 0
        #    loop()
        # else:
        if numStill < stillTime and listenClick:
            moveThread.start()
            loop()
            moveThread.join()
            clickTimer = time.time()
        else:
            if time.time() > clickTimer + clickHold and click and listenClick:
                gui.click(newX, newY)
                clickTimer += 1
            moveThread.start()
            loop()
            moveThread.join()
        # print(newX-oldX, newY-oldY)
        # Checking for keyboard inputs
        k = cv2.waitKey(1)  # TODO May cause lag
        if k == 27:  # press ESC to exit
            break
        elif k == ord('b') and backgroundVersion:  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgThreshold)
            bgCaptured = True
        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0


def loop():
    global newX, newY, oldX, oldY, listenClick, scrollMode, scrollBaseY, dragging, timeSinceFive, bgModel, bgCaptured, initialCalibrate, at, recording, timeSinceThree, numFours, isRecording
    _, frame = webcam.read()
    if not backgroundVersion:
        # flip image
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (screenWidth, screenHeight))
        img = cv2.GaussianBlur(frame, (blurValue, blurValue), 0)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # show frame
        # cv2.imshow('Blur', imgHSV)
        mask = cv2.inRange(imgHSV, lowerBound, upperBound)
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelOpen)
        # show frame
        cv2.imshow('maskClose', maskClose)
        # show frame
        cv2.imshow('mask', mask)
    else:  # TODO Working here
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # use smoothing filter
        frame = cv2.flip(frame, 1)
        # TODO may mess up calculations
        frame = cv2.resize(frame, (screenWidth, screenHeight))
        if rightHanded:
            cv2.rectangle(frame, (int(boxX * frame.shape[1]), 0),(frame.shape[1], int(boxY * frame.shape[0])), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), (int(boxX * frame.shape[1]), int(boxY * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)
        cv2.waitKey(1)
        # print(bgCaptured)
        if bgCaptured:
            img = removeBG(frame)
            if rightHanded:
                img = img[0:int(boxY * screenHeight),
                      int(boxX * screenWidth):screenWidth]  # clip the ROI
            else:
                img = img[0:int(boxY * screenHeight),
                      0:screenWidth - int(boxX * screenWidth)]  # clip the ROI
            cv2.imshow('mask', img)
            # Binarize
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('ori', thresh)
            # maskOpen = thresh[0:0+int(screenHeight*boxY), int(screenWidth*boxX):int(screenWidth*boxX)+screenWidth]
            maskOpen = thresh
            maskOpen = cv2.resize(maskOpen, (screenWidth, screenHeight))  # TODO Here
            cv2.imshow('IMPORTANT', maskOpen)
        # TODO End of BG version
    if not backgroundVersion or (backgroundVersion and bgCaptured):
        _, conts, _ = cv2.findContours(cv2.resize(maskOpen, (screenWidth, screenHeight)), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        maxx, maxy, maxh, maxw = [0, 0, 0, 0];
        ci = 0
        for i in range(len(conts)):
            x, y, w, h = cv2.boundingRect(conts[i])
            if h * w > maxh * maxw:
                maxx = x;
                maxy = y;
                maxw = w;
                maxh = h;
                ci = i
        # If worthy size
        if maxw * maxh > sizeThreshold:
            detect = True
            # Finger processing
            maxCont = conts[ci]
            hull = cv2.convexHull(maxCont)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [maxCont], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            # cv2.imshow("test", drawing)
            isFinishCal, cnt = calculateFingers(maxCont, drawing)
            numFing = cnt + 1
            if numFing == 1:
                numFing = calculateOneZero(maxCont, frame)
            pos = calculateHighestPoint(maxCont)
            # Check for one finger
            if inputQueue.qsize() == maxQueueSize:
                if inputQueue.get() == 4:
                    numFours -= 1
            inputQueue.put(numFing)
            if numFing == 4:
                numFours += 1
            if numFours > maxQueueSize - 2 and not isRecording:
                at.start()
                isRecording = True
                winsound.PlaySound('beep-01a.wav',winsound.SND_FILENAME)
            if numFing == 0 and isRecording:
                transcript=at.stop()
                isRecording = False
                winsound.PlaySound('beep-02.wav',winsound.SND_FILENAME)
                if transcript is not None:
                    gui.typewrite(transcript)
            print(numFing, pos, numFours)
            # Debug
            # print(pos)
            cv2.circle(frame, pos, 5, (0, 0, 255), -1)
            cv2.imshow('Point', frame)
            # Recalibrate
            # # print(cv2.contourArea(maxCont)/screenWidth/screenHeight)
            # if cv2.contourArea(maxCont)/screenWidth/screenHeight > .5 and backgroundVersion and initialCalibrate != 0:
            #     bgCaptured = False
            # initialCalibrate = 1
        else:
            detect = False
        if detect and numFing == 1:
            scaleX = gui.size()[0] / (xBoundHigh - xBoundLow)
            scaleY = gui.size()[1] / (yBoundHigh - yBoundLow)
            oldX = newX
            oldY = newY
            newX = (pos[0] - xBoundLow) * scaleX
            newY = (pos[1] - yBoundLow) * scaleY
            listenClick = True
            scrollMode = False
        elif detect and numFing == 2:
            # if not in scroll mode, store the base y and set scroll mode to true
            if not scrollMode:
                scrollMode = True
                scrollBaseY = pos[1]
            else:
                clicks = int((scrollBaseY - pos[1]))
                gui.scroll(clicks)
        elif detect and numFing >= 5:
            scrollMode = False
            listenClick = False
            timeSinceFive = time.time()
        elif detect and numFing == 0:
            scrollMode = False
            listenClick = False
            # if recording and time.time()-timeSinceThree > 0.5:
            #     recording = False
            #     transcript = at.stop()
            #     if transcript is not None:
            #         gui.typewrite(transcript)
            if time.time() - timeSinceFive < 0.5:
                gui.keyDown('ctrl')
                gui.press('w')
                gui.keyUp('ctrl')
        # elif detect and numFing == 4:
        #     scrollMode = False
        #     listenClick = False
        #     timeSinceThree = time.time()
        #     if not recording:
        #         recording = True
        #         at.start(
        else:
            scrollMode = False
            listenClick = False


def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


run()
