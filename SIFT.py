# Gate detection method using SIFT - matching to a gate template image

import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the matching template and the testing image
SIZE = 180
img_template = cv2.imread("matching_template.png", cv2.IMREAD_GRAYSCALE)
# img_template = cv2.resize(img_template, (SIZE, SIZE))
img_test = cv2.imread("test_data.png", cv2.IMREAD_GRAYSCALE)


# ORB detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img_template, None)
kp2, des2 = orb.detectAndCompute(img_test, None)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matching_result = cv2.drawMatches(
    img_template, kp1, img_test, kp2, matches[:50], None, flags=2)

# Store all the good mathces as per Lowe's rxatio test
good = []
for m in matches[:25]:
    good.append(m)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

h, w = img_template.shape
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

img_test = cv2.polylines(img_test, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img_result = cv2.drawMatches(
    img_template, kp1, img_test, kp2, good, None, **draw_params)
plt.imshow(img_result, "gray"), plt.show()

# plt.imshow(img3, 'gray'),plt.show()
# cv2.imshow("Matching Result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
