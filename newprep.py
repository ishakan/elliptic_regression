import cv2
import numpy as np

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

    mask3_without_extra = np.zeros_like(mask3)

    graycnts, h = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(graycnts)):
        cv2.drawContours(image3, [graycnts[i]], 0, (255, 255, 255), cv2.FILLED)


    maskcnts, h = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    for mcnt in range(len(maskcnts)):
        mask_cnt_area = cv2.contourArea(maskcnts[mcnt])
        largest_area = max(largest_area, mask_cnt_area)

    for mcnt in range(len(maskcnts)):
        mask_cnt_area = cv2.contourArea(maskcnts[mcnt])
        if (mask_cnt_area == largest_area):
            cv2.drawContours(mask3_without_extra, [maskcnts[mcnt]], 0, (255, 255, 255), cv2.FILLED)


    isGood = True
    subtracted_mask3 = cv2.subtract(mask3_without_extra, base_mask)

    cv2.imwrite("vid_images/subtracted%d.jpg" % count_which, subtracted_mask3)  # saves frame as JPEG file

    subtracted_mask3 = cv2.subtract(mask3_without_extra, subtracted_mask3)

    white_pix = np.sum(subtracted_mask3 == 255)
    if (white_pix != 0):

        base_angle, width, height, x, y = draw_elipse(base_mask, image)
        angle_turned3, width3, height3, x3, y3 = draw_elipse(subtracted_mask3, image3)

        lowestamount = 1000000
        adjust_x = 0
        adjust_y = 0
        angle_adjust = 0

        for e in range(20):
            angle_turned = e - 10
            cnt_rotated = rotate_contour2(graycnts[i], angle_turned)

            cnt_rotated = np.round(cnt_rotated * [1.1, 1.1]).astype(int)
            x, y, w, h = cv2.boundingRect(cnt_rotated)
            x = x + int(w // 2)
            y = y + int(h // 2)
            # print(x, y)
            shift_x = 0
            shift_y = 0

            for x_val in range(-10, 10):
                for y_val in range(-10, 10):
                    new_xval = shift_x + x_val
                    new_yval = shift_y + y_val
                    countingimg = subtracted_mask3.copy()
                    cnt_rotated2 = cnt_rotated + [new_xval, new_yval]
                    cv2.drawContours(countingimg, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)
                    white_pixel_count = np.sum(countingimg == 255)

                    if (white_pixel_count < lowestamount):
                        adjust_x = x_val
                        adjust_y = y_val
                        angle_adjust = e - 10
                        lowestamount = white_pixel_count

        # print(angle_adjust, lowestamount, "AL")
        cnt_rotated = rotate_contour2(graycnts[i], angle_adjust)
        cnt_rotated = np.round(cnt_rotated * [1.1, 1.1]).astype(int)

        cnt_rotated2 = cnt_rotated + [shift_x + adjust_x, shift_y + adjust_y]
        cv2.drawContours(image3, [cnt_rotated2], 0, (255, 255, 0), cv2.FILLED)

        cv2.drawContours(mask3_without_extra, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)

        epsilon = 0.02 * cv2.arcLength(graycnts[i], True)  # Adjust the epsilon value as needed
        smoothed_contour = cv2.approxPolyDP(graycnts[i], epsilon, True)

        scaled_contour = np.round(graycnts[i] * [0 + 0.075, 0 + 0.075]).astype(int)

        cv2.drawContours(image, [graycnts[i]], 0, (255, 0, 255), 2)
        cv2.imwrite("output_images/2image%d.jpg" % count_which, image)  # saves frame as JPEG file

        trying_to_undestand = np.zeros_like(image)
        cv2.drawContours(trying_to_undestand, [scaled_contour], 0, (255, 255, 255), 2)

        cv2.drawContours(image, [scaled_contour], 0, (255, 255, 255), 2)

        cv2.imwrite("output_images/image3%d.jpg" % count_which, image3)  # saves frame as JPEG file
        cv2.imwrite("output_images/mask3%d.jpg" % count_which, mask3_without_extra)  # saves frame as JPEG file
        cv2.imwrite("output_images/image%d.jpg" % count_which, image)  # saves frame as JPEG file
        cv2.imwrite("output_images/tryingto%d.jpg" % count_which, trying_to_undestand)  # saves frame as JPEG file

    return base_mask, mask3_without_extra, image3, subtracted_mask3, image


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