import cv2 as cv
import numpy as np
import glob
import os

def main():
    cam = cv.VideoCapture(1)

    if not cam.isOpened():
        print('Could not open camera!')
        exit()

    while cam.isOpened():
        status, frame = cam.read()
        key = cv.waitKey(1)
        if status:
            cv.imshow('test', frame)
        if key == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()

def colibration():
    key = cv.waitKey(0)
    CheckerBoard = (6,6) # checkerboard's row / col
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objPoint = [] # image's 3D point vector
    imgPoint = [] # image's 2D point vector

    # 3D coordinate
    objp = np.zeros((1, CheckerBoard[0] * CheckerBoard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CheckerBoard[0], 0:CheckerBoard[1]].T.reshape(-1,2)
    prev_img_shape = None

    capture = cv.VideoCapture(1)
    capNum = 1
    if not capture.isOpened():
        print('Error!')
        exit()
    while True:
        ret, img = capture.read()
        if not ret:
            print('Error!')
            break
        cv.imshow('capture', img)
        if cv.waitKey(1) == ord('c'):
            img_capture = cv.imwrite('C:\Team_Proj\Camera_Colibration\checkerboard\capture_file%03d.jpg' %capNum, img)
            capNum += 1

        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()

    images = glob.glob('./checkerboard/*.jpg')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardConers(gray,
                                               CheckerBoard,
                                               cv.CALIB_CB_ADAPTIVE_THRESH +
                                               cv.CALIB_CB_FAST_CHECK +
                                               cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objPoint.append(objp)
            corners2 = cv.conerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgPoint.append(corners2)
            img = cv.drawChessboardCorners(img, CheckerBoard, corners2, ret)

        cv.imshow('img', img)
        cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    colibration()
    main()