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
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

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

def takeout(base_frame, image3, angleAdjust, adjustX, adjustY, count_which, store_contours):
    hsv = cv2.cvtColor(base_frame, cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 55, 90])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])

    base_mask = cv2.inRange(hsv, lower_green, upper_green)


    maskGreen = 0

    averageGreenPixelVal = prep_images.findAverageRGBValues(image3, count_which)

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


    lower_green = np.array([35, 55, maskGreen])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])

    image3_mask = cv2.inRange(hsv3, lower_green, upper_green)


    rotatedImage = prep_images.rotate_image(base_mask, angleAdjust, count_which)
    rotatedImage_image3 = prep_images.rotate_image(image3_mask, angleAdjust, count_which)

    base_mask = cv2.cvtColor(rotatedImage, cv2.COLOR_GRAY2BGR)
    base_mask = cv2.cvtColor(base_mask, cv2.COLOR_BGR2HSV)

    image3_mask = cv2.cvtColor(rotatedImage_image3, cv2.COLOR_GRAY2BGR)
    image3_mask = cv2.cvtColor(image3_mask, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(base_mask, lower_white, upper_white)
    white_image3mask = cv2.inRange(image3_mask, lower_white, upper_white)


    baseCnts, h = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_sizedUp = np.round(baseCnts[0] * [1.0, 1.0]).astype(int)

    cnt_rotated2 = cnt_sizedUp + [adjustX, adjustY]
    # cv2.drawContours(image3, [cnt_rotated2], 0, (255, 255, 0), cv2.FILLED)

    cv2.drawContours(white_image3mask, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)

    lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
    upper_white = np.array([255, 30, 255])

    white_image3mask = cv2.cvtColor(white_image3mask, cv2.COLOR_GRAY2BGR)
    white_image3mask = cv2.cvtColor(white_image3mask, cv2.COLOR_BGR2HSV)

    white_image3mask = cv2.inRange(white_image3mask, lower_white, upper_white)

    graycnts, h = cv2.findContours(white_image3mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    beggining_l = len(store_contours)
    for e in range(beggining_l - 1, -1, -1):
        intersects_check = False
        largestContour = -1
        index = -1
        for a in range(len(graycnts)):
            intersects = overlap.overlap_percentage2(store_contours[e], graycnts[a], count_which)
            intersects2 = overlap.overlap_percentage2(graycnts[a], store_contours[e], count_which)
            if (intersects == True or intersects2 == True):
                if (cv2.contourArea(graycnts[a]) > largestContour):
                    largestContour = cv2.contourArea(graycnts[a])
                    index = a
                intersects_check = True

        if (intersects_check):
            store_contours[e] = graycnts[index]
        else:
            store_contours.pop(e)



    cv2.imwrite("output_images/WHITEIMAGE3%d.jpg" % count_which, white_image3mask)  # saves frame as JPEG file

    # print(len(store_contours), count_which, "StORED CONTOURRRS")
    return store_contours


def findAverageRGBValues(image3, count_which):

    lower_green = np.array([35, 55, 90])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])
    hsv = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    result = cv2.bitwise_and(image3, image3, mask=mask)

    non_black_mask = (result[:, :, 0] != 0) & (result[:, :, 1] != 0) & (result[:, :, 2] != 0)

    # Apply the mask to get non-black pixels in the green channel
    green_pixels = result[non_black_mask, 1]  # Index 1 corresponds to the green channel

    average_green = np.average(green_pixels)

    # print(np.average(result[:, :, 1], where=(result != 0)))  # replace 0 with 1 for green channel and with 2 for blue channel

    # average_rgb_values = np.mean(result, axis=(0, 1), where=(result != 0))

    print(average_green, "AVERAGE PIXEL VALUES", count_which)
    return average_green


def rotate_contour(image, contour, angle, count_which):

    angle = 180

    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        # Avoid division by zero if contour is a line
        cx, cy = 0, 0

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    cv2.imwrite("output_images/rotated_image%d.jpg" % count_which, rotated_image)  # saves frame as JPEG file

    return rotated_image


def rotate_image(image, angle, count_which):
    height, width = image.shape[:2]
    centerX, centerY = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    cv2.imwrite("output_images/ARotatedImage%d.jpg" % count_which, rotated)  # saves frame as JPEG file


    return rotated



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
        # print(cv2.contourArea(contour))
        # print("AREA  aA")

        (x, y), (width, height), angle = ellipse
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        # print(x, y)
        center, axes, angle = ellipse

    return angle, width, height, x, y

## create if statement saying if roatted covers more area versus non rotated

def perform_op(image, image3, count_which):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 55, 110])  # Adjust the thresholds as needed
    upper_green = np.array([80, 255, 255])

    base_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask3 = cv2.inRange(hsv3, lower_green, upper_green)


    graycnts, h = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(graycnts)):
        cv2.drawContours(image3, [graycnts[i]], 0, (255, 255, 255), cv2.FILLED)

    isGood = True
    subtracted_mask3 = cv2.subtract(mask3, base_mask)

    cv2.imwrite("vid_images/subtracted%d.jpg" % count_which, subtracted_mask3)  # saves frame as JPEG file

    subtracted_mask3 = cv2.subtract(mask3, subtracted_mask3)

    white_pix = np.sum(subtracted_mask3 == 255)
    if (white_pix != 0):

        angle_turned3, width3, height3, x3, y3 = draw_elipse(subtracted_mask3, image3)
        base_angle, width, height, x, y = draw_elipse(base_mask, image)

        # print("ANGEL")

        ## try with different angle combos and different movement combos, if the area is alrger than the base size
        graycnts, h = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        subtracted_mask3 = cv2.cvtColor(subtracted_mask3, cv2.COLOR_GRAY2BGR)

        for i in range(len(graycnts)):
            area = cv2.contourArea(graycnts[i])
            # cont_area = cv2.contourArea(graycnts[white_cont])
            # print(cont_area, white_cont)
            # if (cont_area > 100):
            # print("area", area)
            if (area > 50):

        # Print the area
        #         print("Contour Area:", area)
                # print(graycnts[i])
                if (angle_turned3 is None or base_angle is None):
                    return base_mask, mask3, image3, subtracted_mask3

                else:
                    cnt_rotated = rotate_contour2(graycnts[i], angle_turned3 - base_angle)
                    # cnt_rotated = graycnts[i]

                    scale_x = height3 / height
                    scale_y = width3 / width

                    cnt_rotated = np.round(cnt_rotated * [1.1, 1.1]).astype(int)
                    x, y, w, h = cv2.boundingRect(cnt_rotated)
                    x = x + int(w // 2)
                    y = y + int(h // 2)
                    # print(x, y)
                    shift_x = x3 - x
                    shift_y = y3 - y

                    lowestamount = 1000000
                    adjust_x = 0
                    adjust_y = 0
                    for x_val in range(-20, 20):
                        for y_val in range(-20, 20):
                            new_xval = shift_x + x_val
                            new_yval = shift_y + y_val
                            countingimg = subtracted_mask3.copy()
                            cnt_rotated2 = cnt_rotated + [new_xval, new_yval]
                            cv2.drawContours(countingimg, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)
                            white_pixel_count = np.sum(countingimg == 255)

                            if (white_pixel_count < lowestamount):
                                adjust_x = x_val
                                adjust_y = y_val
                                lowestamount = white_pixel_count


                    # print(adjust_x, adjust_y)

                    cnt_rotated2 = cnt_rotated + [shift_x + adjust_x, shift_y + adjust_y]
                    cv2.drawContours(image3, [cnt_rotated2], 0, (255, 255, 0), cv2.FILLED)
                    cv2.drawContours(subtracted_mask3, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)


                    epsilon = 0.02 * cv2.arcLength(graycnts[i], True)  # Adjust the epsilon value as needed
                    smoothed_contour = cv2.approxPolyDP(graycnts[i], epsilon, True)

                    scaled_contour = np.round(graycnts[i] * [scale_x + 0.075, scale_y + 0.075]).astype(int)

                    cv2.drawContours(image, [graycnts[i]], 0, (255, 0, 255), 2)
                    cv2.drawContours(image, [scaled_contour], 0, (0, 0, 255), 2)

    return base_mask, mask3, image3, subtracted_mask3, image


def find_base_frame(count, x, y, w, h, count_which):
    i = 0
    base_frame_num = 0
    min_amount_of_pix = 10000000
    const = 0

    allPairs = []

    while i < count:
        img = cv2.imread("vid_images/frame%d.jpg" % i)
        # print("RUN?")
        img = img[y - const:y + h + const, x - const:x + w + const]
        # print("OKAY?")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower_range = np.array([35, 55, 90])  # Adjust the thresholds as needed
        # upper_range = np.array([80, 255, 255])

        upper_green = np.array([80, 255, 255])
        averageGreenPixelVal = prep_images.findAverageRGBValues(img, count_which)

        maskGreen = 0

        ratio2 = (95 - 88) / (210 - 175)
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
        #
        # base_mask = cv2.inRange(hsv, LowerGreen, upper_green)
        #
        #
        #
        #
        #
        # lower_green = np.array([35, 55, 90])  # Adjust the thresholds as needed
        # upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, LowerGreen, upper_green)
        white_pixel_count = np.sum(mask == 255)

        pair = (white_pixel_count, i)
        allPairs.append(pair)

        if (white_pixel_count < min_amount_of_pix):
            min_amount_of_pix = white_pixel_count
            base_frame_num = i
        i += 1

    sorted_pairs = sorted(allPairs, key=lambda x: x[0])
    print(sorted_pairs)
    print("PAIRSSSSSS")
    base_frame_num = sorted_pairs[1][1]

    return base_frame_num
## dealing with trasslation about cells
## when measuring the pili factor into account that the base is englarged by 1.1 time so length may be taken off
## polar cordinates, specifiy poitns with r and thetha