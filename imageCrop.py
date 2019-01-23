import cv2 as cv
import numpy as np
import os

path_dir = 'dataset'
output_path = "output/"
image_list = os.listdir(path_dir)

for img_name in image_list:
    img_color = cv.imread(path_dir+"/"+img_name)
    final_img = img_color.copy()
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img01 = img02 = img03 = img_color.copy()

    #컬러 이미지 --> canny --> 컨투어
    img_canny = cv.Canny(img_color, 50, 150)
    ret, img_binary = cv.threshold(img_canny, 127, 255, 0)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_result = contours[0]
    max = 0

    for cnt in contours:

        x, y, w, h = cv.boundingRect(cnt)
        #cv.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour_area = cv.contourArea(cnt)
        #contour_area = w * h
        if contour_area > max :
            max = contour_area
            max_result = cnt

    x, y, w, h = cv.boundingRect(max_result)
    cv.rectangle(final_img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv.imwrite(output_path+img_name+"_01.jpg", img01[y:y+h, x:x+w])

    # 그레이 --> 컨투어
    ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_result = contours[0]
    max = 0

    for cnt in contours:

        x, y, w, h = cv.boundingRect(cnt)
        #cv.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour_area = cv.contourArea(cnt)
        #contour_area = w*h
        if contour_area > max :
            max = contour_area
            max_result = cnt

    x, y, w, h = cv.boundingRect(max_result)
    cv.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv.imwrite(output_path+img_name+"_02.jpg", img02[y:y+h, x:x+w])

    # 그레이 --> 캐니 --> 컨투어
    img_canny = cv.Canny(img_gray, 50, 150)
    ret, img_binary = cv.threshold(img_canny, 127, 255, 0)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_result = contours[0]
    max = 0

    for cnt in contours:

        x, y, w, h = cv.boundingRect(cnt)
        #cv.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour_area = cv.contourArea(cnt)
        #contour_area = w * h
        if contour_area > max :
            max = contour_area
            max_result = cnt

    x, y, w, h = cv.boundingRect(max_result)
    cv.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    cv.imwrite(output_path + img_name + "_03.jpg", img03[y:y+h, x:x+w])
    cv.imwrite(output_path + img_name + ".jpg", img_color)

    print("finish")

    cv.imshow("imageCrop_result", final_img)
    cv.waitKey(0)