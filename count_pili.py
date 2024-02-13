import cv2
import numpy as np
import prep_images
import overlap
import newprep

def get_num_of_white_pix(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 55, 90])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])
    first_mask = cv2.inRange(hsv, lower_green, upper_green)

    first_mask = cv2.cvtColor(first_mask, cv2.COLOR_GRAY2BGR)
    first_mask = cv2.cvtColor(first_mask, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(first_mask, lower_white, upper_white)
    mask_white_pix = np.sum(white_mask == 255)
    return mask_white_pix

def get_num_of_white_pix_notBase(image, count_which):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    averageGreenPixelVal = prep_images.findAverageRGBValues(image, count_which)

    vValForLowerGreen = 0
    ratio = (24-17) / (210-175)
    if (210 > averageGreenPixelVal > 175):
        vValForLowerGreen = 90
        vValForLowerGreen = int(vValForLowerGreen)
    elif (averageGreenPixelVal <= 130):
        vValForLowerGreen = 90
    elif (averageGreenPixelVal <= 145):
        vValForLowerGreen = 90
    elif (averageGreenPixelVal <= 175):
        vValForLowerGreen = 90
    elif (averageGreenPixelVal >= 210):
        vValForLowerGreen = 90


    lower_green = np.array([35, 55, vValForLowerGreen])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])
    first_mask = cv2.inRange(hsv, lower_green, upper_green)

    first_mask = cv2.cvtColor(first_mask, cv2.COLOR_GRAY2BGR)
    first_mask = cv2.cvtColor(first_mask, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(first_mask, lower_white, upper_white)
    mask_white_pix = np.sum(white_mask == 255)
    return mask_white_pix

#
# lower_green = np.array([35, 55, 90])  # Adjust the thresholds as needed
# upper_green = np.array([80, 255, 255])


def countPili(base_frame, count, x, y, w, hei, count_which):
    #
    # converted_amount_range = [0.070]
    # threshold_range = [0.030]
    i = 0
    total_num_of_pili = 0
    random = None
    mask3 = None
    subtracted_mask3 = None
    white_mask = None
    const = 0
    store_contours = []


    x = int(x)
    y = int(y)
    w = int(w)
    hei = int(hei)
    const = int(const)

    print("count", count)

    hasPili = False
    angleAdjust = 0
    adjustX = 0
    adjustY = 0



    while i < count:

        input_base = base_frame.copy()
        image3 = cv2.imread("vid_images/frame%d.jpg" % i)

        x = int(x)
        y = int(y)
        w = int(w)
        hei = int(hei)
        const = int(const)

        image3 = image3[y - const:y + hei + const, x - const:x + w + const]
        another_cropped = image3.copy()
        img3_copy = image3.copy()


        base_mask_white_pix = get_num_of_white_pix(base_frame)
        image3_white_pix = get_num_of_white_pix_notBase(image3, count_which)

        converted_amount = image3_white_pix * (1.0 - 0.020)

        if (converted_amount <= base_mask_white_pix):

            # print(i, converted_amount, base_mask_white_pix, "HUH????")
            hasPili = False

            store_contours = prep_images.takeout(base_frame, image3, angleAdjust, adjustX, adjustY, i, store_contours)
            # store_contours.clear()

            cv2.imwrite("output_images/BASE_FRAME%d.jpg" % i, base_frame)  # saves frame as JPEG file
            cv2.imwrite("output_images/CASE_FRAME%d.jpg" % i, image3)  # saves frame as JPEG file

            i += 1
            continue


        if (hasPili):
            piliStillThere = newprep.checkForNewFrameIsPiliIsStillThere(input_base, image3, adjustX, adjustY, angleAdjust, i, store_contours)

            if (piliStillThere == False):
                hasPili = False


        mask3, angleAdjust, adjustX, adjustY = newprep.perform_op(input_base, image3, i, hasPili, angleAdjust, adjustX, adjustY)

        mask3 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)

        hsv_image = cv2.cvtColor(mask3, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
        upper_white = np.array([255, 30, 255])

        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        result = cv2.bitwise_and(another_cropped, another_cropped, mask=white_mask)
        cv2.imwrite("output_images/result%d.jpg" % i, result)  # saves frame as JPEG file
        result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        cv2.imwrite("output_images/Gmask%d.jpg" % i, white_mask)  # saves frame as JPEG file

        pili_count = 0
        graycnts, h = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_pili = []

        if (len(graycnts) == 0):
            store_contours.clear()
            hasPili = False


        newPili = []

        for white_cont in range(len(graycnts)):
            cont_area = cv2.contourArea(graycnts[white_cont])
            contThreshold = 0.1
            contX, contY, contW, contH = cv2.boundingRect(graycnts[white_cont])

            if (total_num_of_pili > 10):
                continue

            threshold = 0.1
            pixel_theshold = 0
            if (i == 0):  pixel_theshold = 4
            else:  pixel_theshold = 4

            countWhite = np.zeros_like(white_mask)
            cv2.drawContours(countWhite, graycnts, white_cont, (255, 255, 255), cv2.FILLED)

            pixel_count2 = np.sum(countWhite == 255)

            passesPixelCount = False
            if (pixel_count2 >= pixel_theshold and contW > 1 and contH > 1):
                passesPixelCount = True
            elif (pixel_count2 >= 5):
                passesPixelCount = True

            if (cont_area >= contThreshold or passesPixelCount):
                if (i == 0):
                    (x3, y3, w3, h3) = cv2.boundingRect(graycnts[white_cont])
                    cv2.rectangle(img3_copy, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 255), 1)
                    newPili.append(graycnts[white_cont])
                    total_num_of_pili += 1
                    if (cont_area > 8 or pixel_count2 >= 6):
                        hasPili = True

                else:
                    hasPili = True
                    pili_count += 1
                    intersects_check = False
                    print(i, "VALLLLL", len(store_contours), store_contours)
                    for e in range(len(store_contours)):
                        intersects = overlap.overlap_percentage2(store_contours[e], graycnts[white_cont], i)
                        intersects2 = overlap.overlap_percentage2(graycnts[white_cont], store_contours[e], i)
                        if (intersects == True or intersects2 == True):
                            intersects_check = True
                            break

                    if (intersects_check == False):
                        (x3, y3, w3, h3) = cv2.boundingRect(graycnts[white_cont])
                        cv2.rectangle(img3_copy, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 1)
                        newPili.append(graycnts[white_cont])
                        total_num_of_pili += 1


        beggining_l = len(store_contours)
        for e in range(beggining_l - 1, -1, -1):
            intersects_check = False
            largestContour = -1
            index = -1
            for a in range(len(graycnts)):
                intersects = overlap.overlap_percentage2(store_contours[e], graycnts[a], i)
                intersects2 = overlap.overlap_percentage2(graycnts[a], store_contours[e], i)
                if (intersects == True or intersects2 == True):
                    if (cv2.contourArea(graycnts[a]) > largestContour):
                        largestContour = cv2.contourArea(graycnts[a])
                        index = a
                    intersects_check = True

            if (intersects_check):
                store_contours[e] = graycnts[index]
            else:
                store_contours.pop(e)

        store_contours += newPili



#
#
#         for e in range(len(store_contours)):
#             intersects = overlap.overlap_percentage2(store_contours[e], graycnts[white_cont], i)
#             intersects2 = overlap.overlap_percentage2(graycnts[white_cont], store_contours[e], i)
#             if (intersects == True or intersects2 == True):
#                 intersects_check = True
#                 break
#
#
#
#
#
#         for white_cont in range(len(graycnts)):
#             cont_area = cv2.contourArea(graycnts[white_cont])
#             contThreshold = 0.1
#
#             contX, contY, contW, contH = cv2.boundingRect(graycnts[white_cont])
#
#
#             if (total_num_of_pili > 10):
#                 continue
#
#             threshold = 0.1
#             pixel_theshold = 0
#             if (i == 0):
#                 pixel_theshold = 4
#             else :
#                 pixel_theshold = 4
#
#             countWhite = np.zeros_like(white_mask)
#             cv2.drawContours(countWhite, graycnts, white_cont, (255, 255, 255), cv2.FILLED)
#
#             pixel_count2 = np.sum(countWhite == 255)
#
#             print(cont_area, pixel_count2, i, "COUNTSS")
#
#             passesPixelCount = False
#             if (pixel_count2 >= pixel_theshold and contW > 1 and contH > 1):
#                 passesPixelCount = True
#             elif (pixel_count2 >= 5):
#                 passesPixelCount = True
#             #
#             #
#
#             # if (cont_area > threshold):
#             #     print("areaY", pixel_count, i)
#
#             if (cont_area >= contThreshold or passesPixelCount):
#
#                 if (i == 0):
#                     (x3, y3, w3, h3) = cv2.boundingRect(graycnts[white_cont])
#                     cv2.rectangle(img3_copy, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 255), 1)
#
#                     store_contours.append(graycnts[white_cont])
#                     # hasPili = True
#                     total_num_of_pili += 1
#                     if (cont_area > 8 or pixel_count2 >= 6):
#                         hasPili = True
#                 else:
#                     hasPili = True
#                     pili_count += 1
#                     intersects_check = False
#                     for e in range(len(store_contours)):
#
#                         intersects = overlap.overlap_percentage2(store_contours[e], graycnts[white_cont], i)
#                         intersects2 = overlap.overlap_percentage2(graycnts[white_cont], store_contours[e], i)
#
#                         if (intersects == True or intersects2 == True):
#                             intersects_check = True
#                             store_contours[e] = graycnts[white_cont]
#                             break
#
#                     # if (i == 12):
#                     #     print()
#                     #     print("Passed here?", intersects_check)
#
#                     if (intersects_check == False):
#                         (x3, y3, w3, h3) = cv2.boundingRect(graycnts[white_cont])
#                         cv2.rectangle(img3_copy, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 1)
#
#                         new_pili.append(graycnts[white_cont])
#                         # hasPili = True
#                         total_num_of_pili += 1
#
#
#                 rect = cv2.minAreaRect(graycnts[white_cont])
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#
#                 cv2.drawContours(random, [box], 0, (0, 255, 0), 2)
#
# ## seperate counting the pili and adding the new pili
#         len_of_store_cont = len(store_contours)
#
# ## only let it pass as inter[sects once if the area of the contour is not greater than zero
#         for e2 in range(len_of_store_cont - 1, 0 - 1, -1):
#             checkIfStillThere = False
#             for white_cont2 in range(len(graycnts)):
#                 cont_area = cv2.contourArea(graycnts[white_cont2])
#                 # if (cont_area > 0):
#                 intersects = overlap.overlap_percentage2(store_contours[e2], graycnts[white_cont2], i)
#
#                 if (intersects):
#                     checkIfStillThere = True
#                     break
#
#             # print(checkIfStillThere, i, "IS STILL THERE?")
#             if (checkIfStillThere == False):
#                 store_contours.pop(e2)
#                 len_of_store_cont -= 1
#
#         if (len(new_pili) > 0):
#             for ll in new_pili:
#                 store_contours.append(ll)
#                 rect = cv2.minAreaRect(ll)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#
#
#         if (len(store_contours) == 0):
#             hasPili = False

        cv2.imwrite("vid_images/img3_copy%d.jpg" % i, img3_copy)  # saves frame as JPEG file
        cv2.imwrite("vid_images/mask3%d.jpg" % i, mask3)  # saves frame as JPEG file

        i += 1

    # if (random == null):
    #
    # else:
    #
    # print(random)
    return total_num_of_pili, random, mask3, image3, subtracted_mask3, white_mask, img3_copy


## 35 37 33 11 16 10