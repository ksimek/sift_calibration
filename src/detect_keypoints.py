#!/usr/bin/env python

import argparse

from features_ import *




def main():
    parser = argparse.ArgumentParser(description="Extract Sift/surf keypoints and features");
    parser.add_argument('in_fname', type=str, help='input image file')
    parser.add_argument('out_fname', type=str, help='output filename')
    parser.add_argument('--use_sift', action='store_true', help='Use sift (default surf)');


    args = parser.parse_args()

    (kp, desc) = extract_features(args.in_fname, args.use_sift)

    write_features(args.out_fname, kp, desc)


if __name__ == "__main__":
    main();

