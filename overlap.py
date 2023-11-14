import cv2
#
# def rectangle_overlap(contour1, contour2, threshold=1.2):
#
#     x1, y1, w1, h1 = cv2.boundingRect(contour1)
#     x2, y2, w2, h2 = cv2.boundingRect(contour2)
#
#     # Create a Rect variable
#     # opencv_rect1 = cv2.Rect(x, y, width, height)
#     # opencv_rect1 = (x, y, width, height)
#     # opencv_rect2 = (x2, y2, width2, height2)
#
#     x_intersection = max(x1, x2)
#     y_intersection = max(y1, y2)
#
#     w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
#     h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
#
#     if w_intersection > 0 and h_intersection > 0:
#         intersection_area = w_intersection * h_intersection
#         area_rect1 = w1 * h1
#         area_rect2 = w2 * h2
#
#         # Calculate the percentage of area covered by the intersection
#         percentage_covered = (intersection_area / (area_rect1 + area_rect2 - intersection_area)) * 100
#         print("AYO???", percentage_covered)
#
#         if (percentage_covered < 35):
#             return False
#         else:
#             return True
#         # return (x_intersection, y_intersection, w_intersection, h_intersection)
#     else:
#         return False  #


import cv2

def overlap_percentage2(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    rect1 = (x1, y1, w1, h1)  # Rect1 is completely within Rect2
    rect2 = (x2, y2, w2, h2)
    # Create a Rect variable


    # Calculate the intersection rectangle
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

    # Check for non-overlapping rectangles
    if x1 >= x2 or y1 >= y2:
        return False

    # Calculate areas of the rectangles
    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]

    # Calculate the area of the intersection
    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate the overlap percentage
    overlap_percentage = (intersection_area / min(area1, area2)) * 100
    if (overlap_percentage < 35):
        return False
    else:
        return True
    # print(overlap_percentage, "OVERLAP")
    # return overlap_percentage


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def contour_intersect(cnt_ref, cnt_query):
    ## Contour is a list of points
    ## Connect each point to the following point to get a line
    ## If any of the lines intersect, then break

    for ref_idx in range(len(cnt_ref) - 1):
        ## Create reference line_ref with point AB
        A = cnt_ref[ref_idx][0]
        B = cnt_ref[ref_idx + 1][0]

        for query_idx in range(len(cnt_query) - 1):
            ## Create query line_query with point CD
            C = cnt_query[query_idx][0]
            D = cnt_query[query_idx + 1][0]

            ## Check if line intersect
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                ## If true, break loop earlier
                return True

    return False

def higher_res_model(base_frame):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = 'model/EDSR_x4.pb'  # Replace with the actual path to the model file
    sr.readModel(model_path)
    sr.setModel('edsr', 4)
    upscaled_image = sr.upsample(base_frame)
    cv2.imwrite('upscaled_image.jpg', upscaled_image)
#
# def contour_overlap(cnt_ref, cnt_query, threshold=1.2):
#     total_ref_length = len(cnt_ref)
#     total_query_length = len(cnt_query)
#
#     for ref_idx in range(total_ref_length - 1):
#         A = cnt_ref[ref_idx][0]
#         B = cnt_ref[ref_idx + 1][0]
#
#         for query_idx in range(total_query_length - 1):
#             C = cnt_query[query_idx][0]
#             D = cnt_query[query_idx + 1][0]
#
#             if do_lines_overlap(A, B, C, D, threshold):
#                 return True
#
#     return False
#
#
# def do_lines_overlap(A, B, C, D, threshold):
#     # Calculate the intersection length between line segments AB and CD
#     intersection_length = calculate_intersection_length(A, B, C, D)
#
#     # Calculate the length of line segment AB
#     length_AB = distance(A, B)
#
#     # Calculate the length of line segment CD
#     length_CD = distance(C, D)
#
#     # Calculate the ratio of intersection length to the shorter line segment length
#     ratio = intersection_length / min(length_AB, length_CD)
#
#     # Check if the overlap ratio is greater than or equal to the threshold
#     return ratio >= threshold
#
#
# def calculate_intersection_length(A, B, C, D):
#     # Calculate the intersection point
#     intersection = intersection_point(A, B, C, D)
#
#     if intersection is None:
#         return 0  # No intersection
#
#     # Calculate the distance between the intersection point and both ends of line AB
#     distance_to_A = distance(intersection, A)
#     distance_to_B = distance(intersection, B)
#
#     # Check if the intersection point is within line segment AB
#     if 0 <= distance_to_A <= distance(A, B) and 0 <= distance_to_B <= distance(A, B):
#         return distance_to_A + distance_to_B
#
#     return 0  # Intersection point is not within line segment AB
#
#
# def intersection_point(A, B, C, D):
#     # Calculate the intersection point of two lines AB and CD
#     x1, y1 = A
#     x2, y2 = B
#     x3, y3 = C
#     x4, y4 = D
#
#     denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#
#     if denominator == 0:
#         return None  # Lines are parallel or coincident
#
#     t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
#     u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
#
#     if 0 <= t <= 1 and 0 <= u <= 1:
#         intersection_x = x1 + t * (x2 - x1)
#         intersection_y = y1 + t * (y2 - y1)
#         return (intersection_x, intersection_y)
#
#     return None  # Intersection point is not within line segments AB and CD
#
#
# def distance(point1, point2):
#     # Calculate the Euclidean distance between two points
#     print("EKEJEJEJE")
#     x1, y1 = point1
#     x2, y2 = point2
#     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
