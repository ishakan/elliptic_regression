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

def draw_elipse(input_mask, target_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(input_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area or any other criteria as needed
    min_contour_area = 100  # Minimum contour area to consider
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    # Iterate over the filtered contours and detect pili sticking out of oval cells
    for contour in filtered_contours:
        # Fit an ellipse to the contour
        ellipse = cv2.fitEllipse(contour)
        # cv2.ellipse(target_img, ellipse, (0, 255, 0), 2)

        (x, y), (width, height), angle = ellipse
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        print(x, y)
        # cv2.rectangle(target_img, (x - int(width / 2), y - int(height / 2)), (x + int(width / 2), y + int(height / 2)), (0, 255, 0), 2)


        center, axes, angle = ellipse

        print(angle)
        # Calculate the major and minor axis lengths
        major_axis = max(axes)
        minor_axis = min(axes)
        ellipse_centroid = ellipse[0]

        # Calculate the elongation ratio
        elongation_ratio = major_axis / minor_axis

        # Set the criteria for detecting pili
        max_elongation_ratio = 1.5  # Adjust the maximum elongation ratio as needed

        # If the contour satisfies the criteria for pili detection, draw it on the image
        # if elongation_ratio > max_elongation_ratio:
        #     cv2.drawContours(target_img, [contour], -1, (0, 255, 0), 2)

    return angle, width, height, x, y

# Read the image
image = cv2.imread('non_pili_img.jpg')
image3 = cv2.imread('pili_img.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
hsv3 = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)

# Define the lower and upper green color thresholds
lower_green = np.array([35, 55, 110])  # Adjust the thresholds as needed
upper_green = np.array([80, 255, 255])

# Create a mask for green pixels using the thresholds
base_mask = cv2.inRange(hsv, lower_green, upper_green)
# mask2 = cv2.inRange(hsv2, lower_green, upper_green)
mask3 = cv2.inRange(hsv3, lower_green, upper_green)

# graycnts, h = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for i in range(len(graycnts)):
#     cv2.drawContours(image2, [graycnts[i]], 0, (255, 255, 255), cv2.FILLED)

graycnts, h = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(graycnts)):
    cv2.drawContours(image3, [graycnts[i]], 0, (255, 255, 255), cv2.FILLED)


# subtracted_mask2 = cv2.subtract(mask2, base_mask)
subtracted_mask3 = cv2.subtract(mask3, base_mask)

# subtracted_mask2 = cv2.subtract(mask2, subtracted_mask2)
subtracted_mask3 = cv2.subtract(mask3, subtracted_mask3)

# angle_turned2, width2, height2, x2, y2 = draw_elipse(subtracted_mask2, image2)
angle_turned3, width3, height3, x3, y3 = draw_elipse(subtracted_mask3, image3)
base_angle, width, height, x, y = draw_elipse(base_mask, image)

# print(angle_turned2, base_angle)
print("ANGEL")
graycnts, h = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

subtracted_mask3 = cv2.cvtColor(subtracted_mask3, cv2.COLOR_GRAY2BGR)

# Rotate the input contours to match the rotation of the target contours
rotated_contours = []
for i in range(len(graycnts)):
    cnt_rotated = rotate_contour2(graycnts[i], angle_turned3 - base_angle)
    im_copy = image.copy()

    scale_x = height3 / height
    scale_y = width3 / width

    cnt_rotated = np.round(cnt_rotated * [1.15, 1.15]).astype(int)
    x, y, w, h = cv2.boundingRect(cnt_rotated)
    x = x + int(w // 2)
    y = y + int(h // 2)
    print(x, y)
    shift_x = x3 - x
    shift_y = y3 - y

    cnt_rotated2 = cnt_rotated + [shift_x, shift_y]

    ## shift according to mask without pili
    lowestamount = 1000000
    adjust_x = 0
    adjust_y = 0
    for x_val in range(-20, 20):
        for y_val in range(-20, 20):
            # if (x_val != 0 and y_val != 0):
            new_xval = shift_x + x_val
            new_yval = shift_y + y_val
            countingimg = subtracted_mask3.copy()
            cnt_rotated2 = cnt_rotated + [new_xval, new_yval]
            cv2.drawContours(countingimg, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)
            # _, binary = cv2.threshold(countingimg, 127, 255, cv2.THRESH_BINARY)
            white_pixel_count = np.sum(countingimg == 255)
            # print(white_pixel_count)

            if (white_pixel_count < lowestamount):
                adjust_x = x_val
                adjust_y = y_val
                lowestamount = white_pixel_count


    print(adjust_x, adjust_y)

    cnt_rotated2 = cnt_rotated + [shift_x + adjust_x, shift_y + adjust_y]
    cv2.drawContours(image3, [cnt_rotated2], 0, (255, 255, 0), cv2.FILLED)
    cv2.drawContours(subtracted_mask3, [cnt_rotated2], 0, (80, 80, 0), cv2.FILLED)


    # cnt_rotated = rotate_contour2(graycnts[i], angle_turned2 - base_angle)
    #
    # cnt_rotated = np.round(cnt_rotated * [1.1, 1.1]).astype(int)
    # x, y, w, h = cv2.boundingRect(cnt_rotated)
    # x = x + int(w // 2)
    # y = y + int(h // 2)
    # print(x, y)
    # shift_x = x2 - x
    # shift_y = y2 - y
    #
    # cnt_rotated = cnt_rotated + [shift_x, shift_y]
    #
    # cv2.drawContours(image2, [cnt_rotated], 0, (255, 255, 0), cv2.FILLED)


    cv2.imshow('cnt_rotated Image', im_copy)
    epsilon = 0.02 * cv2.arcLength(graycnts[i], True)  # Adjust the epsilon value as needed
    smoothed_contour = cv2.approxPolyDP(graycnts[i], epsilon, True)

    # scale_x = height2 / height
    # scale_y = width2 / width
    # print(scale_x, scale_y)
    # print(height, width, "Base")
    # print(height2, width3, "2")
    # print(height2, width3, "3")
    #

    scaled_contour = np.round(graycnts[i] * [scale_x + 0.075, scale_y + 0.075]).astype(int)

    # Display the original and resized contours
    # image = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.drawContours(image, [graycnts[i]], 0, (255, 0, 255), 2)
    cv2.drawContours(image, [scaled_contour], 0, (0, 0, 255), 2)
    print("Run?")

# cv2.imshow('mask2', subtracted_mask2)
cv2.imshow('mask', base_mask)
# cv2.imshow('subtracted_mask', subtracted_mask2)
cv2.imshow('subtracted_mask3', subtracted_mask3)
cv2.imshow('mask3', mask3)
# cv2.imshow('image2', image2)
cv2.imshow('image', image)
cv2.imshow('image3', image3)
cv2.imshow('countingimg', countingimg)

cv2.waitKey(0)
cv2.destroyAllWindows()



## dealing with trasslation about cells
## when measuring the pili factor into account that the base is englarged by 1.1 time so length may be taken off
## polar cordinates, specifiy poitns with r and thetha