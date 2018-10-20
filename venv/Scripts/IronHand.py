import numpy as np
import cv2
import pyautogui as gui
import time
import math
gui.FAILSAFE = False;


#Important globals
numFing = 0
#constant
xleft = 330
xright = 0
ytop = 220
ybot = 0
#prior bounds
lowerBound = np.array([80, 50, 50]) #before 60,100,100
upperBound = np.array([120, 255, 255]) #before 180, 255, 255
#new bounds
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

def calculateHighestPoint(res):
    

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
    img = cv2.resize(frame, (330, 220))
    #flip image
    img = cv2.flip(img,1)
    img = cv2.GaussianBlur(img,(blurValue,blurValue),0)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # show frame
    cv2.imshow('Blur', imgHSV)
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelOpen)
    #show frame
    cv2.imshow('maskClose',maskClose)
    # show frame
    cv2.imshow('mask', mask)
    # show frame
    cv2.imshow('maskOpen', maskOpen)
    # show frame
    cv2.imshow('maskClose', maskClose)
    cv2.waitKey(1)
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    _,conts,_ = cv2.findContours(maskOpen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxx, maxy, maxh, maxw = [0,0,0,0];
    ci=0
    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        if h * w > maxh * maxw:
            maxx = x; maxy = y; maxw = w; maxh = h;
            ci = i
    if maxw * maxh > 25:
        w,h = gui.size()
        wr, hr = w/(xright-xleft), h/(ybot-ytop)
        x1 = maxx+maxw/2
        y1 = maxy+maxh/2
        x = int(w-(x1-xleft)*wr)
        y = int((y1-ytop)*hr)

    #Finger processing
    try:
        res = conts[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        isFinishCal, cnt = calculateFingers(res, drawing)
        numFing = cnt + 1
        # Debug
        print(numFing)
    except:
        fdsa =1


    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    #     gui.moveTo(x, y)
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

