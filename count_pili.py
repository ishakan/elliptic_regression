import cv2
import numpy as np
import prep_images
import overlap
import newprep

def get_num_of_white_pix(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 55, 110])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])
    first_mask = cv2.inRange(hsv, lower_green, upper_green)

    first_mask = cv2.cvtColor(first_mask, cv2.COLOR_GRAY2BGR)
    first_mask = cv2.cvtColor(first_mask, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(first_mask, lower_white, upper_white)
    mask_white_pix = np.sum(white_mask == 255)
    return mask_white_pix


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

    # print(count)
    # print("COUNT")

    image3 = cv2.imread("vid_images/frame%d.jpg" % 1)
    x = int(x)
    y = int(y)
    w = int(w)
    hei = int(hei)
    const = int(const)

    image3 = image3[y - const:y + hei + const, x - const:x + w + const]


    while i < count:
        # print(i)
        # print(len(store_contours))
        input_base = base_frame.copy()
        image3 = cv2.imread("vid_images/frame%d.jpg" % i)
        # height = image3.shape[0]
        # width = image3.shape[1]
        # print("SPACEr")
        # print(x, y, w, hei)
        # print("NEXT PHA")
        x = int(x)
        y = int(y)
        w = int(w)
        hei = int(hei)
        const = int(const)

        image3 = image3[y - const:y + hei + const, x - const:x + w + const]
        img3_copy = image3.copy()

        # img3_copy = image3.copy()


        base_mask_white_pix = get_num_of_white_pix(base_frame)
        image3_white_pix = get_num_of_white_pix(image3)


        converted_amount = image3_white_pix * (1.0 - 0.070)
        # print("area", converted_amount, base_mask_white_pix)

        if (converted_amount <= base_mask_white_pix):
            # print("conntinue")
            # print(i, count, "WHAa")

            i += 1
            continue

        # print("basemask_white_pix", base_mask_white_pix, "image3_white_pix", image3_white_pix, "pili #", count_which)


        # print(i)
        if (i == 4):
            print("")
            # print("$")
        if (i == 6):
            # print("PILIIII2")
            hsv = cv2.cvtColor(input_base, cv2.COLOR_BGR2HSV)
            hsv3 = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)

            lower_green = np.array([35, 55, 110])  # Adjust the thresholds as needed
            upper_green = np.array([80, 255, 255])

            base_mask = cv2.inRange(hsv, lower_green, upper_green)
            mask3 = cv2.inRange(hsv3, lower_green, upper_green)

            subtracted_mask3 = cv2.subtract(mask3, base_mask)
            white_pix = np.sum(subtracted_mask3 == 255)

            # print("PILIIII", white_pix)

        random = image3.copy()
        random, mask3, image3, subtracted_mask3, idrllyk = newprep.perform_op(input_base, image3, i)

        # random, mask3, image3, subtracted_mask3, idrllyk = prep_images.perform_op(input_base, image3, i)
        # cv2.imwrite("vid_images/idrllyk%d.jpg" % i, idrllyk)  # saves frame as JPEG file

        mask3 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)

        hsv_image = cv2.cvtColor(mask3, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
        upper_white = np.array([255, 30, 255])

        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        pili_count = 0
        graycnts, h = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_pili = []


        for white_cont in range(len(graycnts)):
            cont_area = cv2.contourArea(graycnts[white_cont])
            # print(cont_area, white_cont)
            # if (cont_area > 100):
# 132,  135 , 136, 139, 144, 146, 147, 148 , 149
            if (total_num_of_pili > 10):
                continue

            # print("pili_area", cont_area)

            threshold = base_mask_white_pix * (0.015)

                # print("aREA")
            if (cont_area > threshold):
                if (i == 0):
                    # print(i)
                    (x3, y3, w3, h3) = cv2.boundingRect(graycnts[white_cont])
                    cv2.rectangle(img3_copy, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 255), 1)
                    # print(x3, y3, w3, h3)
                    # print("COORDINATES")
                    store_contours.append(graycnts[white_cont])
                    total_num_of_pili += 1
                    # print(len(store_contours), "LEN")
                else:
                    pili_count += 1
                    intersects_check = False
                    for e in range(len(store_contours)):
                        # intersects = overlap.contour_overlap(store_contours[e], graycnts[white_cont])
                        # overlap.rectangle_overlap(store_contours[e], graycnts[white_cont])
                        intersects = overlap.overlap_percentage2(store_contours[e], graycnts[white_cont])
                        intersects2 = overlap.overlap_percentage2(graycnts[white_cont], store_contours[e])

                        if (intersects == True or intersects2 == True):
                            intersects_check = True
                            store_contours[e] = graycnts[white_cont]
                            break
                    if (intersects_check == False):
                        # print(i, cont_area)
                        (x3, y3, w3, h3) = cv2.boundingRect(graycnts[white_cont])
                        # print(x3, y3, w3, h3, "INTERSECTS")
                        cv2.rectangle(img3_copy, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 1)
                        # print(x3, y3, w3, h3)
                        # print("COORDINATES")

                        new_pili.append(graycnts[white_cont])
                        total_num_of_pili += 1

                rect = cv2.minAreaRect(graycnts[white_cont])
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                cv2.drawContours(random, [box], 0, (0, 255, 0), 2)

        len_of_store_cont = len(store_contours)
        # print(i, white_cont, cont_area, "KEl")

        for e2 in range(len_of_store_cont - 1, 0 - 1, -1):
            checkIfStillThere = False
            for white_cont2 in range(len(graycnts)):
                cont_area = cv2.contourArea(graycnts[white_cont2])
                if (cont_area > 0):
                    # print(len(store_contours), white_cont2, "FIRST")
                    # intersects = overlap.contour_overlap(store_contours[e2], graycnts[white_cont2])
                    # overlap.rectangle_overlap(store_contours[e2], graycnts[white_cont2])
                    intersects = overlap.overlap_percentage2(store_contours[e2], graycnts[white_cont2])

                    if (intersects):
                        checkIfStillThere = True
                        break
            if (checkIfStillThere == False):
                store_contours.pop(e2)
                len_of_store_cont -= 1
                # print(len(store_contours), white_cont2, "SECOND")

        if (len(new_pili) > 0):
            # print(i, "FRA ME NUM")
            for ll in new_pili:
                store_contours.append(ll)
                rect = cv2.minAreaRect(ll)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # randd = cv2.imread("vid_images/frame%d.jpg" % i)

                # cv2.drawContours(randd, [box], 0, (255, 255, 0), 2)
                # cv2.imwrite("vid_images/frame%d.jpg" % i, randd)  # saves frame as JPEG file

            # print(len(store_contours), "THIRD")
        # print(i, count, "WHAa")
        cv2.imwrite("vid_images/img3_copy%d.jpg" % i, img3_copy)  # saves frame as JPEG file
        cv2.imwrite("vid_images/mask3%d.jpg" % i, mask3)  # saves frame as JPEG file
        cv2.imwrite("vid_images/random%d.jpg" % i, random)  # saves frame as JPEG file
        cv2.imwrite("vid_images/subtracted_mask3%d.jpg" % i, subtracted_mask3)  # saves frame as JPEG file

        i += 1

    # if (random == null):
    #
    # else:
    #
    # print(random)
    return total_num_of_pili, random, mask3, image3, subtracted_mask3, white_mask, img3_copy


## 35 37 33 11 16 10