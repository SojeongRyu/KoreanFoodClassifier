import cv2 as cv
import numpy as np
import os

path_dir = 'dataset'
output_path = "output/"
image_list = os.listdir(path_dir)

for img_name in image_list:
    img_color = cv.imread(path_dir+"/"+img_name)
    contour_final_img = img_color.copy()
    rectangle_final_img = img_color.copy()
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img01_1 = img_color.copy()
    img02_1 = img_color.copy()
    img03_1 = img_color.copy()
    img01_2 = img_color.copy()
    img02_2 = img_color.copy()
    img03_2 = img_color.copy()

    #컬러 이미지 --> canny --> 컨투어
    img_canny = cv.Canny(img_color, 50, 150)
    ret, img_binary = cv.threshold(img_canny, 127, 255, 0)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    ContourArea_max_result = contours[0]
    ContourArea_max = 0
    RectangleArea_max_result = contours[0]
    RectangleArea_max = 0

    for cnt in contours:

        x, y, w, h = cv.boundingRect(cnt)
        #cv.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour_area = cv.contourArea(cnt)
        rectangle_area = w * h
        if contour_area > ContourArea_max :
            ContourArea_max = contour_area
            ContourArea_max_result = cnt

        if rectangle_area > RectangleArea_max :
            RectangleArea_max = contour_area
            RectangleArea_max_result = cnt

    x, y, w, h = cv.boundingRect(ContourArea_max_result)
    cv.rectangle(contour_final_img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv.imwrite(output_path+img_name+"_01_1.jpg", img01_1[y:y+h, x:x+w])

    x, y, w, h = cv.boundingRect(RectangleArea_max_result)
    cv.rectangle(rectangle_final_img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv.imwrite(output_path + img_name + "_01_2.jpg", img01_2[y:y + h, x:x + w])

    # 그레이 --> 컨투어
    ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    ContourArea_max_result = contours[0]
    ContourArea_max = 0
    RectangleArea_max_result = contours[0]
    RectangleArea_max = 0

    for cnt in contours:

        x, y, w, h = cv.boundingRect(cnt)
        #cv.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour_area = cv.contourArea(cnt)
        rectangle_area = w * h
        if contour_area > ContourArea_max:
            ContourArea_max = contour_area
            ContourArea_max_result = cnt

        if rectangle_area > RectangleArea_max:
            RectangleArea_max = contour_area
            RectangleArea_max_result = cnt

    x, y, w, h = cv.boundingRect(ContourArea_max_result)
    cv.rectangle(contour_final_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv.imwrite(output_path + img_name + "_02_1.jpg", img02_1[y:y + h, x:x + w])

    x, y, w, h = cv.boundingRect(RectangleArea_max_result)
    cv.rectangle(rectangle_final_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv.imwrite(output_path + img_name + "_02_2.jpg", img02_2[y:y + h, x:x + w])

    # 그레이 --> 캐니 --> 컨투어
    img_canny = cv.Canny(img_gray, 50, 150)
    ret, img_binary = cv.threshold(img_canny, 127, 255, 0)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    ContourArea_max_result = contours[0]
    ContourArea_max = 0
    RectangleArea_max_result = contours[0]
    RectangleArea_max = 0

    for cnt in contours:

        x, y, w, h = cv.boundingRect(cnt)
        #cv.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour_area = cv.contourArea(cnt)
        rectangle_area = w * h
        if contour_area > ContourArea_max:
            ContourArea_max = contour_area
            ContourArea_max_result = cnt

        if rectangle_area > RectangleArea_max:
            RectangleArea_max = contour_area
            RectangleArea_max_result = cnt

    x, y, w, h = cv.boundingRect(ContourArea_max_result)
    cv.rectangle(contour_final_img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    cv.imwrite(output_path + img_name + "_03_1.jpg", img03_1[y:y + h, x:x + w])

    x, y, w, h = cv.boundingRect(RectangleArea_max_result)
    cv.rectangle(rectangle_final_img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    cv.imwrite(output_path + img_name + "_03_2.jpg", img03_2[y:y + h, x:x + w])

    cv.imwrite(output_path + img_name + ".jpg", img_color)
    print("finish")

    cv.imshow("contourArea_imageCrop_result", contour_final_img)
    cv.imshow("rectangleArea_imageCrop_result", rectangle_final_img)
    cv.waitKey(0)