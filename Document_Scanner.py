import cv2 as cv
import numpy as np
import utlis

ImgWidth, ImgHeight = 450, 600


#############################################
#   PREPROCESSING THE IMAGE
#############################################
image = cv.imread('Document 1.jpg')
image = cv.resize(image, (ImgWidth, ImgHeight))
utils.create_trackbar()

while True:
    threshold = utlis.val_trackbar()
    imgCanny, imgErode = utlis.preprocessing(image)
    ################################################################################################
    ################################################################################################
    #                                   CONTOURS
    ################################################################################################
    try:
        contours, img_cont_1 = utlis.get_contour(image, imgCanny, minArea=5000, filters=4, draw=True)
        Biggest_contour = contours[0][3]
    except:
        contours, img_cont_1 = utlis.get_contour(image, imgErode, minArea=5000, filters=4, draw=True)
        Biggest_contour = contours[0][3]
    # contours contain (contours, area, peri, approx, len(approx), center, bbox)
    ##########################################################################################
    ##########################################################################################
    #                 Arranging in order of points for Warp Pers.
    ##########################################################################################
    Biggest_contour = utlis.reorder(Biggest_contour)

    imgWarp = utlis.get_warp(image, Biggest_contour, ImgWidth, ImgHeight, ImgWidth, ImgHeight)
    imgWarp = imgWarp[10:imgWarp.shape[0]-10, 10:imgWarp.shape[1]-10]
    imgWarp = cv.resize(imgWarp, (ImgWidth, ImgHeight))

    imgWarpGray = cv.cvtColor(imgWarp, cv.COLOR_BGR2GRAY)
    imgAdaptiveThre = cv.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    imgAdaptiveThre = cv.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre = cv.medianBlur(imgAdaptiveThre, 3)


    imgResult = utlis.concat(0.5, [[image, imgCanny, imgErode], [img_cont_1, imgWarp, imgAdaptiveThre]])
    cv.imshow('imgResult', imgResult)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
