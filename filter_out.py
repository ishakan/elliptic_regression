import cv2
from keras.models import load_model
from PIL import ImageOps
import numpy as np
from PIL import Image as im

def resize_with_padding(img, expected_size):
    img = im.fromarray(img)
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def filter_out_mutli_cells(image_to_filter, imgmin, pili):
    # model = load_model('ai_model/keras_model.h5')
    imgmincopy = imgmin.copy()
    model = load_model('ai_model/keras_model(4).h5', compile=False)

    class_names = ["Single Cell", "Multi Cell", "Other Cell"]

    hold_all_single_cells = np.zeros_like(image_to_filter)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    pili2 = cv2.cvtColor(pili, cv2.COLOR_BGR2GRAY)

    grayfinalimg = cv2.cvtColor(image_to_filter, cv2.COLOR_BGR2GRAY)
    ## img already converte to GRAY
    graycnts, h = cv2.findContours(grayfinalimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    cbreak = 1
    counter = 0
    # subtract = cv2.cvtColor(subtract, cv2.COLOR_BGR2GRAY)
    for icontour in graycnts:
        area = cv2.contourArea(icontour)
        # print(area, "sreaa")
        (x, y, w, h) = cv2.boundingRect(icontour)
        # print("first stage?")
        if (area > 50):
            print(x, y, w, h)
            try:
                # print("okayyyy")
                area2 = cv2.contourArea(icontour)
                counter += 1
                const = 0
                # print(pili2.shape[1], pili2.shape[0], "before")
                # print(grayfinalimg.shape[1], grayfinalimg.shape[0], "before2")

                croppedsub = grayfinalimg[y - const:y + h + const, x - const:x + w + const]
                pili3 = pili2[y - const:y + h + const, x - const:x + w + const]
                croppedfram1 = imgmin[y - const:y + h + const, x - const:x + w + const]

                # croppedsub = cv2.cvtColor(croppedsub, cv2.COLOR_BGR2GRAY)
                # pili2 = cv2.cvtColor(pili2, cv2.COLOR_BGR2GRAY)
                # print(pili3.shape[1], pili3.shape[0], "Shape")
                # print(croppedsub.shape[1], croppedsub.shape[0], "Shapee")
                # print(croppedfram1.shape[1], croppedfram1.shape[0], "Shapeee")

                croppedsub = cv2.resize(croppedsub, (pili3.shape[1], pili3.shape[0]))
                subb = cv2.subtract(croppedsub, pili3)
                croppedsub2 = cv2.cvtColor(subb, cv2.COLOR_GRAY2BGR)

        # Perform bitwise AND operation
        #         together = cv2.bitwise_and(croppedfram1, croppedsub2, mask=subb)
        #
        #         print("befff")
        #
        #         croppedsub2 = cv2.cvtColor(subb, cv2.COLOR_GRAY2BGR)
        #         print("First")

                together = cv2.bitwise_and(croppedfram1, croppedsub2, mask=subb)

                # cv2.imwrite("cropped_img/cropped_img%.jpg" % counter, together)  # saves frame as JPEG file
                # cv2.imwrite("cropped_img/cropped_img%d.jpg" % counter, together)

                resize = (0, 0)
                if (together.shape[0] > together.shape[1]):
                    ratio = int(244 / together.shape[0])
                    resize = (together.shape[0] * ratio, together.shape[1] * ratio)
                else:
                    ratio = int(244 / together.shape[1])
                    resize = (together.shape[0] * ratio, together.shape[1] * ratio)

                immg = cv2.resize(together, (resize[1], resize[0]), interpolation=cv2.INTER_CUBIC)

                immg = resize_with_padding(immg, (244, 244))



                immg = np.asarray(immg)
                cv2.imwrite("cropped_img/cropped_img%d.jpg" % counter, immg)

                # cv2.imwrite("together%d.jpg" % counter, immg)  # saves frame as JPEG file
                print("second")

                image = together.copy()
                size = (224, 224)
                image = cv2.resize(immg, size, interpolation=cv2.INTER_LINEAR)

                # cv2.imwrite("cropped_img/cropped_img%d.jpg" % counter, image)  # saves frame as JPEG file

                image_array = np.asarray(image)

                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array

                prediction = model.predict(data)
                # print(prediction)

                index = np.argmax(prediction)
                class_name = class_names[index]
                print("third")
                #
                message = str(counter)
                message += " "
                message += str(class_name)
                cv2.putText(imgmincopy, str(message),
                            (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 167, 255), 1)

                cv2.imwrite("output_images/classifiedImg.jpg", imgmincopy)  # saves frame as JPEG file

                print("Class: ", class_name, " ", counter)
                if (class_name == "Single Cell"):
                    cv2.drawContours(hold_all_single_cells, [icontour], -1, [255, 255, 255], cv2.FILLED)

            except:
                print("This one didn't work", str(counter))

    return hold_all_single_cells

# cv2.imshow("added_mask2", added_mask2)

# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
