# Gate detection method using SURF

# modules
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image

# Initiate timer
start_time = time.time()

# Global variables
template_size = 300  # template resizing
invariance_size = 300  # invariance image resizing
min_matching = 15  # minimum number of keypoint areas

# Read the matching template and the testing image in grayscale
img_template = cv2.imread(
    "figures/matching_template_3.png", cv2.IMREAD_GRAYSCALE)
# resize the template image
img_template = cv2.resize(img_template, (template_size, template_size))
img_test = cv2.imread("figures/test_data.png", cv2.IMREAD_GRAYSCALE)

# Addition of test images to check scale and rotational invariance - also done for time testing
img_test_invariance = cv2.pyrDown(img_test)
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
# img_test = img_test_invariance

# Initiate SURF detection method
# larger value filters more extensively (500-1000 value)
surf = cv2.xfeatures2d.SURF_create(800)

# detection and extraction of feature keypoints + feature description (step 1 and 2 in the method explanation)
kp_template, des_template = surf.detectAndCompute(img_template, None)
kp_test, des_test = surf.detectAndCompute(img_test, None)

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
    if m.distance < 0.75 * n.distance:  # threshold of less than 0.75
        inliers.append(m)

# Enables for a minimum amount of matches otherwise the detection method will not work properly
if len(inliers) > min_matching:
    # extract the locations of matched keypoints in both images
    img_template_kp = np.float32(
        [kp_template[m.queryIdx].pt for m in inliers]).reshape(-1, 1, 2)
    img_test_kp = np.float32(
        [kp_test[m.trainIdx].pt for m in inliers]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(img_template_kp, img_test_kp, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img_template.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # draw the lines around the detected object
    img_test = cv2.polylines(
        img_test, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
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
print("\nNumber of Keypoints - template detected ", len(kp_template))
print("\nNumber of Keypoints - test detected ", len(kp_test))

# Draw the figure that includes the template image and the testing image
img_result = cv2.drawMatches(
    img_template, kp_template, img_test, kp_test, inliers, None, **draw_matches)
plt.imshow(img_result, "gray"), plt.show()
