#!/usr/bin/env python

import argparse
from features_ import *

def main():
    parser = argparse.ArgumentParser(description="Match keypoints");
    parser.add_argument('in_fname_1', type=str, help='input keypoint file')
    parser.add_argument('in_fname_2', type=str, help='input keypoint file')
    parser.add_argument('out_fname', type=str, help='output matches filename')
    parser.add_argument('--lowe_threshold', nargs='?', type=float, default=0.6, help='Use David Lowe\'s ratio criterion for pruning bad matches')
    parser.add_argument('--homography_threshold', nargs='?', type=float, default=0, help='fit a homography to matches and prune by this reprojection error threshold')

    args = parser.parse_args()

    (kp1, desc1) = read_features(args.in_fname_1)
    (kp2, desc2) = read_features(args.in_fname_2)

    matches = match_features(desc1, desc2, args.lowe_threshold)

    if args.homography_threshold > 0:
        [matches, H] = filter_matches_by_homography(kp1, kp2, matches, args.homography_threshold)

    write_matches(args.out_fname, matches)
    read_matches(args.out_fname)

if __name__ == "__main__":
    main();

