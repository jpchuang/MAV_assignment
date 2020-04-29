# Gate detection method using SURF

# modules
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pandas as pd
from shapely.geometry import Polygon


# SURF function

def func_SURF(template, test):

    # Initiate timer
    start_time = time.time()

    # Local variables
    template_size = 360  # template size
    test_size = 360  # test image size
    invariance_size = 300  # invariance image resizing
    min_matching = 15  # minimum number of keypoint areas

    # Resize images
    template = cv2.resize(template, (template_size, template_size))
    test = cv2.resize(test, (test_size, test_size))

    # Addition of test images to check scale and rotational invariance - also done for time testing
    img_test_invariance = cv2.pyrDown(test)
    nrows, ncols = img_test_invariance.shape[:2]

    # rotation matrix 45 degrees and scale size of 1
    rotation_matrix = cv2.getRotationMatrix2D((ncols/2, nrows/2), 45, 1)
    # transformation of original image using rotation matrix
    img_test_invariance = cv2.warpAffine(
        img_test_invariance, rotation_matrix, (ncols, nrows))

    # resize
    img_test_invariance = cv2.resize(
        img_test_invariance, (invariance_size, invariance_size))

    # display transformed image for testing purposes
    # plt.imshow(img_test_invariance), plt.show()

    # usage of the manipulated images - uncomment to test variety of manipulations
    # test = img_test_invariance

    # Initiate SURF detection method
    # larger value filters more extensively (300-800 value)
    surf = cv2.xfeatures2d.SURF_create(500)

    # detection and extraction of feature keypoints + feature description (step 1 and 2 in the method explanation)
    kp_template, des_template = surf.detectAndCompute(template, None)
    kp_test, des_test = surf.detectAndCompute(test, None)

    # Next step is matching the detected keypoint regions using the FLANN Matcher method in OpenCV
    # usage of flann index tree algorithm which indexes the features in a tree like structure and iterates through it
    FLANN_INDEX_KDTREE = 0
    # Pass of two dictionaries that specifies the algorithm that is going to be used and its related parameters
    index_dict = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # this specifies the number of times the trees in the index dict should be recursively traversed
    search_dict = dict(checks=50)

    # initiate matching using the extracted descriptor vectors
    flann = cv2. FlannBasedMatcher(index_dict, search_dict)
    matches = flann.knnMatch(des_template, des_test, k=2)

    # Filter the matches from outliers using the Lowe's ratio test
    inliers = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # threshold of less than 0.75
            inliers.append(m)

    # Enables for a minimum amount of matches otherwise the detection method will not work properly
    if len(inliers) > min_matching:
        # extract the locations of matched keypoints in both images
        img_template_kp = np.float32(
            [kp_template[m.queryIdx].pt for m in inliers]).reshape(-1, 1, 2)
        img_test_kp = np.float32(
            [kp_test[m.trainIdx].pt for m in inliers]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(
            img_template_kp, img_test_kp, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # obtain points from the reference image - use of data excel given
        pts = np.float32([[88, 140], [237, 150], [233, 279], [68, 292]]
                         ).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # draw polylines based on the perspective transform on the test image to draw an estimated box
        img_test = cv2.polylines(
            test, [np.int32(dst)], True, 0, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found")
        matchesMask = None

    # Draw the inlier matches that were found previously
    draw_matches = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=matchesMask,  # draw only inliers
                        flags=2)

    # Resulting timer
    print("--- %s seconds ---" % (time.time() - start_time))

    # Print total number of matching points between the training and query images
    print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
    # print("\nNumber of Keypoints - template detected ", len(kp_template))
    # print("\nNumber of Keypoints - test detected ", len(kp_test))

    # Draw the figure that includes the template image and the testing image
    img_result = cv2.drawMatches(
        template, kp_template, test, kp_test, inliers, None, **draw_matches)
    plt.imshow(img_result, "gray"), plt.show()

    return dst


# Function to output the False Positives and True Positives

def func_ROC(dst, x1, y1, x2, y2, x3, y3, x4, y4):
    # obtain the estimated coordinates of the box in the test image
    coordinates_test = dst.flatten()

    # Draw the polygon of the box - predicted
    p = Polygon([(coordinates_test[0], coordinates_test[1]),
                 (coordinates_test[2], coordinates_test[3]),
                 (coordinates_test[4], coordinates_test[5]),
                 (coordinates_test[6], coordinates_test[7])])

    # draw polygon with image coordinates from given excel data - Ground truth
    # (110, 147), (233, 156), (226, 264), (96, 269) - img 166 data
    q = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    # Calculate the intersection and the resulting IoU
    threshold = 0.9  # set at 0.9
    global_thresh = threshold
    bool_intersect = p.intersects(q)  # True
    ratio = p.intersection(q).area / p.union(q).area  # 1.0
    output_bool = False
    if ratio == threshold or ratio > threshold:
        output_bool = True
    else:
        output_bool = False

    return(output_bool)


# Init - Main

# get template and test image
# fixed template image - pic #65
img_template = cv2.imread(
    "figures/matching_template.png", cv2.IMREAD_GRAYSCALE)
img_test = cv2.imread(
    "figures/test_data.png", cv2.IMREAD_GRAYSCALE)  # test normal #166

# call the SURF method
func_SURF(img_template, img_test)


# Part for the ROC curve - loop through the training dataset to find TPR and FPR
# comment these part below out to get the TPR and FPR

# excel_file = "ROC_excel.xlsx"
# data = pd.read_excel(excel_file)

# # extracted values from excel
# positive_list = []
# negative_list = []

# for i in data.index:
#     image_test_loop = data["image"][i]

#     # retrieval of coordinate points from excel
#     x1 = data["x1"][i]
#     y1 = data["y1"][i]
#     x2 = data["x2"][i]
#     y2 = data["y2"][i]
#     x3 = data["x3"][i]
#     y3 = data["y3"][i]
#     x4 = data["x4"][i]
#     y4 = data["y4"][i]

#     # Loop through the training images
#     img_template = cv2.imread(
#         "figures/matching_template.png", cv2.IMREAD_GRAYSCALE)  # fixed template image - pic #65
#     img_test = cv2.imread("figures/images/{image}".format(image=image_test_loop),
#                           cv2.IMREAD_GRAYSCALE)  # test normal #166

#     # call the functions
#     dst = func_SURF(img_template, img_test)
#     bool_ROC = func_ROC(dst, x1, y1, x2, y2, x3, y3, x4, y4)

#     if bool_ROC == True:
#         positive_list.append(bool_ROC)
#         # print("\nDetected Gate: ", bool_ROC)
#     else:
#         negative_list.append(bool_ROC)
#         # print("\nDetected Gate: ", bool_ROC)

# Calculations
# TPR - Ratio of gates was successfully detected given a specific IoU threshold
# TPR = len(positive_list) / (len(positive_list) + len(negative_list))
# FPR = len(negative_list) / (len(positive_list) + len(negative_list))

# FPR_list = [0, 0.15, 0.236, 0.243, 0.257,
#             0.279, 0.293, 0.314, 0.421, 0.471, 0.993]
# TPR_list = [0.007, 0.529, 0.579, 0.686, 0.707,
#             0.721, 0.743, 0.757, 0.764, 0.85, 1]

# plt.figure()
# plt.plot(FPR_list, TPR_list, 'b')
# plt.title("ROC Curve for low to high IoU thresholds")
# plt.xlabel("FPR")
# plt.ylabel("TPR")
# plt.show()
