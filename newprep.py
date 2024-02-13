import cv2
import numpy as np
import prep_images
import overlap


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour2(cnt, angle):
    if cnt is None:
        print("NOOOONNEEEE")
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        # Handle the case where M['m00'] is zero (e.g., the contour is empty)
        cx = cy = 0
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def draw_elipse(input_mask, target_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(input_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 0  # Minimum contour area to consider
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    angle = None
    width = None
    height = None
    x = None
    y = None
    for contour in filtered_contours:
        ellipse = cv2.fitEllipse(contour)

        (x, y), (width, height), angle = ellipse
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        # print(x, y)

        center, axes, angle = ellipse

    return angle, width, height, x, y


## create if statement saying if roatted covers more area versus non rotated

def perform_op(image, image3, count_which, hasPili, angleAdjust, adjustX, adjustY):
    # print("Running???????")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)
    cv2.imwrite("output_images/CASE_FRAME%d.jpg" % count_which, image3)  # saves frame as JPEG file

    ## get average value of green pixel and base masks off of that value
    ##110, 157, 135, 140, 124, 104,

    averageGreenPixelVal = prep_images.findAverageRGBValues(image3, count_which)

    # print(averageGreenPixelVal, count_which, "GREEN AVERGAGE")


    vValForLowerGreen = 0
    ratio = (24-17) / (210-175)
    if (210 > averageGreenPixelVal > 175):
        vValForLowerGreen = 117 + (ratio * (averageGreenPixelVal - 175))
        vValForLowerGreen = int(vValForLowerGreen)
    elif (averageGreenPixelVal <= 130):
        vValForLowerGreen = 95
    elif (averageGreenPixelVal <= 145):
        vValForLowerGreen = 108
    elif (averageGreenPixelVal <= 175):
        vValForLowerGreen = 117
    elif (averageGreenPixelVal >= 210):
        vValForLowerGreen = 124


    lower_green = np.array([35, 55, vValForLowerGreen])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])

    maskGreen = 0

    ratio2 = (95-88) / (210-175)
    if (averageGreenPixelVal <= 150):
        maskGreen = 85
    elif (210 > averageGreenPixelVal > 175):
        maskGreen = 88 + (ratio2 * (averageGreenPixelVal - 175))
        maskGreen = int(maskGreen)
    elif (averageGreenPixelVal <= 175):
        maskGreen = 88
    elif (averageGreenPixelVal >= 210):
        maskGreen = 95


    LowerGreen = np.array([35, 55, maskGreen])  # Adjust the thresholds as needed

    base_mask = cv2.inRange(hsv, LowerGreen, upper_green)

    cv2.imwrite("output_images/BASEMASKK%d.jpg" % count_which, base_mask)  # saves frame as JPEG file

    mask3 = cv2.inRange(hsv3, lower_green, upper_green)

    if (count_which == 3):
        cv2.imwrite("output_images/cimage3.jpg", image3)  # saves frame as JPEG file
        cv2.imwrite("output_images/checkItOut.jpg", mask3)  # saves frame as JPEG file

    mask3_without_extra = np.zeros_like(mask3)

    graycnts, h = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image3 = np.zeros_like(image3)

    for i in range(len(graycnts)):
        cv2.drawContours(image3, [graycnts[i]], 0, (255, 255, 255), cv2.FILLED)

    cv2.imwrite("output_images/CASEMASKK%d.jpg" % count_which, image3)  # saves frame as JPEG file

    # print("LENGTH OF CONTOURRRR:", len(graycnts))

    maskcnts, h = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    for mcnt in range(len(maskcnts)):
        mask_cnt_area = cv2.contourArea(maskcnts[mcnt])
        largest_area = max(largest_area, mask_cnt_area)

    for mcnt in range(len(maskcnts)):
        mask_cnt_area = cv2.contourArea(maskcnts[mcnt])
        if (mask_cnt_area == largest_area):
            cv2.drawContours(mask3_without_extra, [maskcnts[mcnt]], 0, (255, 255, 255), cv2.FILLED)

    subtracted_mask3 = cv2.subtract(mask3_without_extra, base_mask)

    cv2.imwrite("vid_images/subtracted%d.jpg" % count_which, subtracted_mask3)  # saves frame as JPEG file

    subtracted_mask3 = cv2.subtract(mask3_without_extra, subtracted_mask3)

    white_pix = np.sum(subtracted_mask3 == 255)
    adjust_x = 0
    adjust_y = 0
    angle_adjust = 0
    increaseSize = 1.00
    shift_x = 0
    shift_y = 0

    if (hasPili):
        adjust_x = adjustX
        adjust_y = adjustY
        angle_adjust = angleAdjust + 0


    ## FOR HAS PILI CHECK AGAIN, IF THERE WAS PILI BEFORE, CHECK IF THE PILI IS STILL THERE
    # #IN THE NEW FRAME USING THE OLD ADJUST VALUES, IF THERE IS NO PILI THEN SET hasPILI to False
    elif (white_pix != 0):
        base_angle, width, height, x, y = draw_elipse(base_mask, image)
        angle_turned3, width3, height3, x3, y3 = draw_elipse(subtracted_mask3, image3)

        lowestamount = 1000000

        count = 0
#range(60)
        for e in range(30):
            # print("ee", e)
            angle_turned = (e - 15) * 1

            before = np.zeros_like(mask3_without_extra)
            after = np.zeros_like(mask3_without_extra)
            after2 = np.zeros_like(mask3_without_extra)

            cv2.drawContours(before, [graycnts[i]], 0, (80, 80, 0), cv2.FILLED)

            # prep_images.rotate_contour(before, graycnts[i], angle_turned, e)

            # cnt_rotated = rotate_contour2(graycnts[i], angle_turned)
            cv2.imwrite("output_images/BASSEEE%d.jpg" % e, base_mask)  # saves frame as JPEG file

            rotatedImage = prep_images.rotate_image(base_mask, angle_turned, e)

            rotatedcnts, h = cv2.findContours(rotatedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #

            # cnt_rotated = rotatedcnts[0]
            #
            # # cnt_rotated = np.round(rotatedcnts[0] * [increaseSize, increaseSize]).astype(int)
            #
            # cv2.drawContours(after, [cnt_rotated], 0, (255, 255, 255), cv2.FILLED)

            hsvOfBase = cv2.cvtColor(rotatedImage, cv2.COLOR_GRAY2BGR)
            hsvOfBase = cv2.cvtColor(hsvOfBase, cv2.COLOR_BGR2HSV)

            lower_white = np.array([0, 0, 150])
            upper_white = np.array([19, 240, 255])

            # lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
            # upper_white = np.array([255, 30, 255])

            whiteMaskOfBase = cv2.inRange(hsvOfBase, lower_white, upper_white)
            cv2.imwrite("output_images/WHITEMASKOFBASE%d.jpg" % e, whiteMaskOfBase)  # saves frame as JPEG file

            baseCnts, h = cv2.findContours(whiteMaskOfBase, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largestContourKept = np.zeros_like(rotatedImage)
            largestArea = 0
            for rcnts in baseCnts:
                rcont_area = cv2.contourArea(rcnts)
                if (rcont_area > largestArea):
                    largestArea = rcont_area

            for rcnts in baseCnts:
                rcont_area = cv2.contourArea(rcnts)
                if (rcont_area == largestArea):
                    largestArea = rcont_area
                    cv2.drawContours(largestContourKept, [rcnts], 0, (255, 255, 255), cv2.FILLED)

            # rotatedImage = cv2.cvtColor(rotatedImage, cv2.COLOR_BGR2GRAY)
            baseCnts, h = cv2.findContours(largestContourKept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(after2, [baseCnts[0]], 0, (80, 80, 0), cv2.FILLED)

            cv2.imwrite("output_images/BEFORE%d.jpg" % e, before)  # saves frame as JPEG file
            cv2.imwrite("output_images/AFTER%d.jpg" % e, after2)  # saves frame as JPEG file

            # x, y, w, h = cv2.boundingRect(cnt_rotated)
            # x = x + int(w // 2)
            # y = y + int(h // 2)
            # print(x, y)

            for x_val in range(-5, 5):
                for y_val in range(-5, 5):
                    new_xval = shift_x + x_val
                    new_yval = shift_y + y_val
                    countingimg = mask3_without_extra.copy()
                    cnt_sizedUp = np.round(baseCnts[0] * [increaseSize, increaseSize]).astype(int)

                    cnt_rotated2 = cnt_sizedUp + [new_xval, new_yval]
                    cv2.drawContours(countingimg, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)
                    white_pixel_count = np.sum(countingimg == 255)

                    # if (white_pixel_count <= lowestamount):
                    #     if (count_which == 3):
                            # print(adjust_x, adjust_y, white_pixel_count, "COUNTS")

                    count += 1
                    # print(new_xval,new_yval, angle_turned, white_pixel_count)

                    if (white_pixel_count <= lowestamount):
                        # print(white_pixel_count)
                        adjust_x = x_val
                        adjust_y = y_val
                        angle_adjust = angle_turned + 0
                        lowestamount = white_pixel_count

    else:
        return mask3_without_extra, angle_adjust, adjust_x, adjust_y
    # #
    # if (count_which == 59):
    #     adjust_y -= 1
        # adjust_x -= 1

    rotatedImage = prep_images.rotate_image(base_mask, angle_adjust, count_which)

    hsvOfBase = cv2.cvtColor(rotatedImage, cv2.COLOR_GRAY2BGR)
    hsvOfBase = cv2.cvtColor(hsvOfBase, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([19, 240, 255])
    whiteMaskOfBase = cv2.inRange(hsvOfBase, lower_white, upper_white)

    baseCnts, h = cv2.findContours(whiteMaskOfBase, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContourKept = np.zeros_like(rotatedImage)
    largestArea = 0
    for rcnts in baseCnts:
        rcont_area = cv2.contourArea(rcnts)
        if (rcont_area > largestArea):
            largestArea = rcont_area

    for rcnts in baseCnts:
        rcont_area = cv2.contourArea(rcnts)
        if (rcont_area == largestArea):
            largestArea = rcont_area
            cv2.drawContours(largestContourKept, [rcnts], 0, (255, 255, 255), cv2.FILLED)

    # rotatedImage = cv2.cvtColor(rotatedImage, cv2.COLOR_BGR2GRAY)
    baseCnts, h = cv2.findContours(largestContourKept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # baseCnts, h = cv2.findContours(whiteMaskOfBase, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_sizedUp = np.round(baseCnts[0] * [increaseSize, increaseSize]).astype(int)

    # if (count_which == 2):
    #     shift_x += 1
    cnt_rotated2 = cnt_sizedUp + [shift_x + adjust_x, shift_y + adjust_y]
    cv2.drawContours(image3, [cnt_rotated2], 0, (255, 255, 0), cv2.FILLED)

    cv2.drawContours(mask3_without_extra, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)

    # print("ERM WHY ISN THIS RUNNING?", count_which)
    cv2.imwrite("output_images/erm%d.jpg" % count_which, mask3_without_extra)  # saves frame as JPEG file

    ##image33, mask33
    epsilon = 0.02 * cv2.arcLength(graycnts[i], True)  # Adjust the epsilon value as needed
    smoothed_contour = cv2.approxPolyDP(graycnts[i], epsilon, True)

    # scaled_contour = np.round(graycnts[i] * [0 + 0.075, 0 + 0.075]).astype(int)
    #
    # cv2.drawContours(image, [graycnts[i]], 0, (255, 0, 255), 2)
    # cv2.imwrite("output_images/2image%d.jpg" % count_which, image)  # saves frame as JPEG file

    trying_to_undestand = np.zeros_like(image)
    # cv2.drawContours(trying_to_undestand, [scaled_contour], 0, (255, 255, 255), 2)
    #
    # cv2.drawContours(image, [scaled_contour], 0, (255, 255, 255), 2)

    cv2.imwrite("output_images/image3%d.jpg" % count_which, image3)  # saves frame as JPEG file
    cv2.imwrite("output_images/mask3%d.jpg" % count_which, mask3_without_extra)  # saves frame as JPEG file
    cv2.imwrite("output_images/image%d.jpg" % count_which, image)  # saves frame as JPEG file
    cv2.imwrite("output_images/tryingto%d.jpg" % count_which, trying_to_undestand)  # saves frame as JPEG file

    return mask3_without_extra, angle_adjust, adjust_x, adjust_y


def find_base_frame(count, x, y, w, h, count_which):
    i = 0
    base_frame_num = 0
    min_amount_of_pix = 10000000
    const = 0
    while i < count:
        img = cv2.imread("vid_images/frame%d.jpg" % i)
        # print("RUN?")
        img = img[y - const:y + h + const, x - const:x + w + const]
        # print("OKAY?")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 55, 110])  # Adjust the thresholds as needed
        upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        white_pixel_count = np.sum(mask == 255)

        if (white_pixel_count < min_amount_of_pix):
            min_amount_of_pix = white_pixel_count
            base_frame_num = i
        i += 1

    return base_frame_num


## dealing with trasslation about cells
## when measuring the pili factor into account that the base is englarged by 1.1 time so length may be taken off
## polar cordinates, specifiy poitns with r and thetha


def checkForNewFrameIsPiliIsStillThere(image, image3, adjust_x, adjust_y, angle_adjust, count_which, storedPili):
    # print("Running???????")
    cv2.imwrite("output_images/CASE_FRAME%d.jpg" % count_which, image3)  # saves frame as JPEG file

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)

    ##110, 157, 135, 140, 124,
    averageGreenPixelVal = prep_images.findAverageRGBValues(image3, count_which)

    vValForLowerGreen = 0
    ratio = (24-17) / (210-175)
    if (210 > averageGreenPixelVal > 175):
        vValForLowerGreen = 117 + (ratio * (averageGreenPixelVal - 175))
        vValForLowerGreen = int(vValForLowerGreen)
    elif (averageGreenPixelVal <= 136):
        vValForLowerGreen = 100
    elif (averageGreenPixelVal <= 145):
        vValForLowerGreen = 108
    elif (averageGreenPixelVal <= 175):
        vValForLowerGreen = 117
    elif (averageGreenPixelVal >= 210):
        vValForLowerGreen = 124


    lower_green = np.array([35, 55, vValForLowerGreen])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])

    maskGreen = 0

    ratio2 = (95-90) / (210-175)
    if (averageGreenPixelVal <= 150):
        maskGreen = 85
    elif (210 > averageGreenPixelVal > 175):
        maskGreen = 90 + (ratio2 * (averageGreenPixelVal - 175))
        maskGreen = int(maskGreen)
    elif (averageGreenPixelVal <= 175):
        maskGreen = 90
    elif (averageGreenPixelVal >= 210):
        maskGreen = 95


    LowerGreen = np.array([35, 55, maskGreen])  # Adjust the thresholds as needed

    base_mask = cv2.inRange(hsv, LowerGreen, upper_green)

    cv2.imwrite("output_images/BASEMASKK%d.jpg" % count_which, base_mask)  # saves frame as JPEG file

    mask3 = cv2.inRange(hsv3, lower_green, upper_green)

    mask3_without_extra = np.zeros_like(mask3)

    graycnts, h = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image3 = np.zeros_like(image3)

    for i in range(len(graycnts)):
        cv2.drawContours(image3, [graycnts[i]], 0, (255, 255, 255), cv2.FILLED)

    cv2.imwrite("output_images/CASEMASKK%d.jpg" % count_which, image3)  # saves frame as JPEG file

    # print("LENGTH OF CONTOURRRR:", len(graycnts))

    maskcnts, h = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    for mcnt in range(len(maskcnts)):
        mask_cnt_area = cv2.contourArea(maskcnts[mcnt])
        largest_area = max(largest_area, mask_cnt_area)

    for mcnt in range(len(maskcnts)):
        mask_cnt_area = cv2.contourArea(maskcnts[mcnt])
        if (mask_cnt_area == largest_area):
            cv2.drawContours(mask3_without_extra, [maskcnts[mcnt]], 0, (255, 255, 255), cv2.FILLED)

    increaseSize = 1.00
    shift_x = 0
    shift_y = 0

    rotatedImage = prep_images.rotate_image(base_mask, angle_adjust, count_which)
    hsvOfBase = cv2.cvtColor(rotatedImage, cv2.COLOR_GRAY2BGR)
    hsvOfBase = cv2.cvtColor(hsvOfBase, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([19, 240, 255])
    whiteMaskOfBase = cv2.inRange(hsvOfBase, lower_white, upper_white)
    baseCnts, h = cv2.findContours(whiteMaskOfBase, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_sizedUp = np.round(baseCnts[0] * [increaseSize, increaseSize]).astype(int)

    cnt_rotated2 = cnt_sizedUp + [shift_x + adjust_x, shift_y + adjust_y]

    cv2.drawContours(mask3_without_extra, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)

    mask3 = cv2.cvtColor(mask3_without_extra, cv2.COLOR_GRAY2BGR)

    hsv_image = cv2.cvtColor(mask3, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
    upper_white = np.array([255, 30, 255])

    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    graycnts, h = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkIfAtAll = False

    for e in range(len(storedPili)):
        for white_cont in range(len(graycnts)):
            intersects = overlap.overlap_percentage2(storedPili[e], graycnts[white_cont], count_which)
            intersects2 = overlap.overlap_percentage2(graycnts[white_cont], storedPili[e], count_which)

            if (intersects == True or intersects2 == True):
                checkIfAtAll = True
                break

    # print(count_which, checkIfAtAll, "IS THE PILI STILL THERE?")
    return checkIfAtAll