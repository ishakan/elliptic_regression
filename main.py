import cv2
import numpy as np
import prep_images
import count_pili
import filter_out


## 373 - 0, program said 1

vidcap = cv2.VideoCapture('videos/TND0905.001.mov')

success, image = vidcap.read()
#hi
count = 0

height = image.shape[0]
width = image.shape[1]
x1 = int(width / 2)
x2 = int(width)
y1 = int(height / 2)
y2 = int(height)

while success:
    cv2.imwrite("vid_images/frame%d.jpg" % count, image)  # saves frame as JPEG file
    success, image = vidcap.read()
    count += 1

vidcap.release()  # Release the video capture object when done

print(count)
print("COUNT1")
lower_white = np.array([0, 0, 150])
upper_white = np.array([19, 240, 255])

imgmin = cv2.imread("vid_images/frame1.jpg")

lower_range = np.array([35, 55, 90])  # Adjust the thresholds as needed
upper_range = np.array([80, 255, 255])

hsvmin = cv2.cvtColor(imgmin, cv2.COLOR_BGR2HSV)
maskmin = cv2.inRange(hsvmin, lower_range, upper_range)

pili = np.zeros_like(imgmin)
i = 0
print("here??")

while i < count:
    img = cv2.imread("vid_images/frame%d.jpg" % i)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    sub = cv2.subtract(mask, maskmin)

    contours, h = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(pili, [c], -1, [255, 255, 255], cv2.FILLED)
    sub = cv2.subtract(maskmin, mask)
    contours, h = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if (area > -1):
            cv2.drawContours(pili, [c], -1, [255, 255, 255], cv2.FILLED)

    i += 1

maskmin = cv2.cvtColor(maskmin, cv2.COLOR_GRAY2BGR)
added_mask = cv2.addWeighted(pili, 1, maskmin, 1, 0)

filtered_mask = filter_out.filter_out_mutli_cells(added_mask, imgmin, pili)
filtered_mask2 = np.zeros_like(filtered_mask.copy())

take_out_large_pili = cv2.cvtColor(filtered_mask, cv2.COLOR_BGR2GRAY)
pilicnts, h = cv2.findContours(take_out_large_pili, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

all_counted_cells = np.zeros_like(filtered_mask)

for pilicnt in pilicnts:
    total_num_of_pili = 0
    (x, y, w, h) = cv2.boundingRect(pilicnt)
    const = 0
    cropped_pili = pili[y - const:y + h + const, x - const:x + w + const]
    # pili_bgr = cv2.cvtColor(cropped_pili, cv2.COLOR_GRAY2BGR)
    pili_hsv = cv2.cvtColor(cropped_pili, cv2.COLOR_BGR2HSV)
    area12 = cv2.contourArea(pilicnt)


    lower_white = np.array([0, 0, 200])  # Adjust the thresholds as needed
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(pili_hsv, lower_white, upper_white)
    num_of_white_pix = np.sum(white_mask == 255)

    # num_of_white_pix = count_pili.get_num_of_white_pix(base_frame)
    # print(, "ratio")
    ratio = num_of_white_pix / area12
    if (ratio > 0.7):
        # print(area12, "area")
        cv2.putText(pili, "filtered out",
                    (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 0), 2)
        # print(ratio, "ratio")
        #
        # print(num_of_white_pix, "Num of White Pix")

    else:
        cv2.drawContours(filtered_mask2, [pilicnt], -1, [255, 255, 255], cv2.FILLED)

cv2.imwrite("output_images/filtered_mask.jpg", filtered_mask)  # saves frame as JPEG file

grayfinalimg = cv2.cvtColor(filtered_mask2, cv2.COLOR_BGR2GRAY)
graycnts, h = cv2.findContours(grayfinalimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count_which = 0
random = None
blank_set = []
for grays in graycnts:
    total_num_of_pili = 0
    area12 = cv2.contourArea(grays)

    if (area12 > 15):
        count_which += 1
#280 ] ,
        if (count_which < 0 or count_which > 100):
            continue

        const = 0
        (x, y, w, h) = cv2.boundingRect(grays)

        base_frame_num = prep_images.find_base_frame(count, x, y, w, h, count_which)

        # base_frame_num = 53
        base_frame = cv2.imread("vid_images/frame%d.jpg" % base_frame_num)
        base_frame = base_frame[y - const:y + h + const, x - const:x + w + const]

        cv2.imwrite("vid_images/basegrame%d.jpg" % count_which, base_frame)  # saves frame as JPEG file

        total_num_of_pili, random, mask3, image3, subtracted_mask3, white_mask, img3_copy = count_pili.countPili(base_frame, count, x, y, w, h, count_which)


        if (total_num_of_pili >= 8):
            count_which -= 1
            continue

        cv2.drawContours(all_counted_cells, [grays], -1, [255, 255, 255], cv2.FILLED)

        blank_set.append(total_num_of_pili)
        print(count_which, total_num_of_pili)
        # print("TOTAL AMOUNT", total_num_of_pili)
        cv2.putText(filtered_mask, str(count_which),
                    (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 0), 2)


i = 0
min_amount_of_pix = 10000000
print(blank_set)
image = cv2.imread("vid_images/frame%d.jpg" % 0)
print(total_num_of_pili, "total amount")
# cv2.imshow('subtracted_mask3', subtracted_mask3)
# cv2.imshow('mask3', mask3)
cv2.imshow('image', image)
# cv2.imshow('white_mask', white_mask)
# cv2.imshow('image3', image3)
# cv2.imshow('img3_copy', img3_copy)
cv2.imshow('pili', pili)
cv2.imshow('all_counted_cells', all_counted_cells)
cv2.imshow('maskmin', maskmin)
cv2.imshow('mask', mask)
cv2.imshow('hold_all_single_cells', filtered_mask)
# cv2.imshow('random', random)
# cv2.imshow('base', base_frame)


cv2.waitKey(0)
cv2.destroyAllWindows()
