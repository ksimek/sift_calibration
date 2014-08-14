#!/usr/bin/env python

import argparse
from features_ import *

def main():
    parser = argparse.ArgumentParser(description="Calibrate using keypoint matches");
    parser.add_argument('num_cameras', type=int, help='Number of views')
    parser.add_argument('num_frames', type=int, help='Number of frames')
    parser.add_argument('cam_fmt', type=int, help='Printf-formatted string representing intrinsic camera calibration files (see calibrate_intrinsic)')
    parser.add_argument('pattern_fname', type=str, help='reference pattern keypoint file (from extract_keypoints)')
    parser.add_argument('key_fmt', type=str, help='printf-formatted string for keypoint files (from extract_keypoints)')
    parser.add_argument('match_fmt', type=str, help='printf-formatted string for match files (from match_keypoints)')
    parser.add_argument('camera_order', type=str, help='sequential order of cameras for initial pairwise estimation of extrinsics')

    args = parser.parse_args()

    ref_keys = read_features(arg.pattern_fname);

    keys = read_keys_(arg.num_cameras, arg.num_frames, arg.key_fmt)

    matches = read_matches_(arg.num_cameras, arg.num_frames, arg.match_fmt)

    cam_intrinsics = read_camera_intrinsics(arg.num_cameras, arg.cam_fmt)

    matches_mat = matches_to_matrix(matches);
    cams = get_initial_cameras(keys, matches_mat, args.order)

    points = triangulate_points(cams, keys, matches_mat)

    write_bal_file(args.out_fname, cams, points, keys, matches_mat)

if __name__ == "__main__":
    main();
