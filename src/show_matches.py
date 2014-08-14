#!/usr/bin/env python
from sys import argv
import features_ as feat
import cv2

if len(argv) < 6:
    print "Usage: show_matches img_1 img_2 keypoints_1 keypoints_2 matches"
    exit(1)
img1 = cv2.imread(argv[1], 0)
img2 = cv2.imread(argv[2], 0)

kp1 = feat.read_features(argv[3])[0]
kp2 = feat.read_features(argv[4])[0]

matches = feat.read_matches(argv[5])


matched_img = feat.draw_matches(img1, img2, kp1, kp2, matches)

cv2.imshow("Matches", matched_img)
cv2.waitKey()
