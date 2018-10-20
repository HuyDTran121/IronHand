import numpy as np
import cv2
import pyautogui as gui
import time
import math

gui.FAILSAFE = False;

# Important globals
numFing = 0
pos = tuple([0,0])
detect = False
# constant
sizeThreshold = 8000
screenWidth = int(960/2)
screenHeight = int(540/2)
bound = 4
xBoundLow = int(screenWidth/bound)
yBoundLow = int(screenHeight/bound)
xBoundHigh = int((bound-1)*screenWidth/bound)
yBoundHigh = int((bound-1)*screenHeight/bound)
# movemenSmooth < 1
movementSmooth = .1
# prior bounds
lowerBound = np.array([80, 50, 50])  # before 60,100,100
upperBound = np.array([120, 255, 255])  # before 180, 255, 255
# new bounds
# lowerBound = np.array([200, 7, 209])
# upperBound = np.array([230, 21, 235])
blurValue = 41
font = cv2.FONT_HERSHEY_SIMPLEX
webcam = cv2.VideoCapture(0)
kernelOpen = np.ones((2, 2))

clickTimer = time.time()
clickReset = None;
prevx = 0
prevy = 0


def calculateHighestPoint(maxCont):
    extTop = tuple(maxCont[maxCont[:, :, 1].argmin()][0])
    return extTop


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


while webcam.isOpened():
    _, frame = webcam.read()
    # flip image
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (screenWidth, screenHeight))
    img = cv2.GaussianBlur(frame, (blurValue, blurValue), 0)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # show frame
    cv2.imshow('Blur', imgHSV)
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelOpen)
    # show frame
    cv2.imshow('maskClose', maskClose)
    # show frame
    cv2.imshow('mask', mask)
    cv2.waitKey(1)
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    _, conts, _ = cv2.findContours(maskOpen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    #Area debug
    print(maxw * maxh)
    #If worthy size
    if maxw * maxh > sizeThreshold:
        # w, h = gui.size()
        # wr, hr = w / (xBoundHigh - xBoundLow), h / (yBoundHigh - yBoundLow)
        # x1 = maxx + maxw / 2
        # y1 = maxy + maxh / 2
        # x = int(w - (x1 - xBoundLow) * wr)
        # y = int((y1 - yBoundLow) * hr)
        detect = True

    # Finger processing
        res = conts[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        isFinishCal, cnt = calculateFingers(res, drawing)
        numFing = cnt + 1
        pos = calculateHighestPoint(res)
        # Debug
        # print(pos)
        cv2.circle(frame, pos, 5, (0, 0, 255), -1)
        cv2.imshow('Point', frame)
        cv2.waitKey(1)
    else: detect = False
    if detect:
        scaleX = gui.size()[0]/(xBoundHigh-xBoundLow)
        scaleY = gui.size()[1]/ (yBoundHigh - yBoundLow)
        x = (pos[0]-xBoundLow)*scaleX
        y = (pos[1]-yBoundLow)*scaleY
        gui.moveTo(x, y, movementSmooth, gui.easeInQuad)
    #     if abs(x-prevx) < 10 and y-prevy < 10:
    #         if time.time() > clickTimer+0.5:
    #             gui.click(x, y)
    #             clickTimer +=1
    #     else:
    #         prevx = x
    #         prevy = y
    #         clickTimer = time.time()
    # else:
    #     clickTimer = time.time()
    #  Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
