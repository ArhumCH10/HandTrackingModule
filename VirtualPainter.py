import cv2
import numpy as np
import os
import HandTrackingModule as htm

##################################
brushThickness = 15
eraserThickness = 100
##################################

# Load header images
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlay = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlay.append(image)
print(f"{len(overlay)} header images loaded")
header = overlay[0]
drawColor = (49, 49, 255)  # default color

# Set up camera
cap = cv2.VideoCapture(0)
cap.set(3, 1000)   # Width
cap.set(4, 720)    # Height

# Initialize hand detector
detector = htm.HandDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1000, 3), np.uint8)

while True:
    # 1. Capture image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1000, 720))  # resize immediately to match canvas

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
    fingers = detector.fingersUp()

    # 4. Selection Mode - Two fingers up
    if fingers[1] and fingers[2]:
        xp, yp = 0, 0
        # Check for header click
        if y1 < 110:
            if 100 < x1 < 250:
                header = overlay[0]
                drawColor = (49, 49, 255)
            elif 280 < x1 < 380:
                header = overlay[1]
                drawColor = (99, 191, 0)
            elif 400 < x1 < 460:
                header = overlay[2]
                drawColor = (255, 182, 56)
            elif 560 < x1 < 700:
                header = overlay[3]
                drawColor = (0, 0, 0)  # Eraser
        cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)

    # 5. Drawing Mode - Index finger up
    if fingers[1] and not fingers[2]:
        cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
        if xp == 0 and yp == 0:
            xp, yp = x1, y1

        if drawColor == (0, 0, 0):  # Eraser
            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
        else:
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

        xp, yp = x1, y1

    # 6. Merge canvas and frame
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # 7. Place header
    header_resized = cv2.resize(header, (1000, 125))
    img[0:125, 0:1000] = header_resized

    # 8. Show images
    cv2.imshow('Virtual Painter', img)
    cv2.imshow('Canvas', imgCanvas)
    cv2.waitKey(1)
