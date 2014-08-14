#!/usr/bin/env python

from random import shuffle
import argparse
import features_ as feat
import calibration_ as calib
import os.path
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Match keypoints and calibrate camera");
    parser.add_argument('pattern_key_fname', type=str, help='reference pattern keypoint file')
    parser.add_argument('num_frames', type=int, help='Number of scene frames')
    parser.add_argument('scene_key_fmt', type=str, help='Printf-formatted string representing keypoint files from N scene frames')
    parser.add_argument('example_image', type=str, help='Example image (used to get dimensions)')
    parser.add_argument('out_matches_fmt', type=str, help='printf string for output matches filenames')
    parser.add_argument('out_camera_fname', type=str, help='output camera intrinsic calibration filename')
    parser.add_argument('--num_iterations', nargs='?', type=int, default=0.6, help='number of iterations of match & homography re-estimation (2-3 is usually good)')
    parser.add_argument('--lowe_threshold', nargs='?', type=float, default=0.6, help='Use David Lowe\'s ratio criterion for pruning bad matches')
    parser.add_argument('--homography_threshold', nargs='?', type=float, default=0, help='fit a homography to matches and prune by this reprojection error threshold (zero ignores)')
    parser.add_argument('--f_threshold', nargs='?', type=float, default=2, help='fit a fundamental matrix to matches and prune by this error threshold (zero ignores)')
    parser.add_argument('--min_matches', nargs='?', type=int, default=25, help='Minimum number of matches in a frame to be accepted into calibration set')

    args = parser.parse_args()

    [pattern_keys, pattern_descs] = feat.read_features(args.pattern_key_fname);

    if args.num_frames == 1:
        frame_fnames = [args.scene_key_fmt]
        out_match_fnames = [args.out_matches_fmt]
    else:      
        I = range(1, args.num_frames+1)
        frame_fnames = [args.scene_key_fmt % x for x in I]
        out_match_fnames = [args.out_matches_fmt % x for x in I]

    tmp = [(a,b) for (a,b) in zip(frame_fnames, out_match_fnames) if os.path.isfile(a)]
    [frame_fnames, out_match_fnames] = zip(*tmp)

    if len(frame_fnames) == 0:
        print "No files found matching %s" % args.scene_key_fmt
        exit(1)

    print "reading keypoint from %d frames" % len(frame_fnames)
    frames_features = [feat.read_features(x) for x in frame_fnames]
    [frames_keys, frames_descs] = zip(*frames_features)

    img = cv2.imread(args.example_image)
    img_size = (img.shape[1], img.shape[0]);

    distortion = np.zeros((1,5))

    pattern_kp_mat = np.array([x.pt for x in pattern_keys], np.float32)
    frame_kp_mats = [np.array([x.pt for x in keys], np.float32) for keys in frames_keys]
    fixed_kp_mats = [x.copy() for x in frame_kp_mats]

    frames_matches = [feat.match_features(pattern_descs, descs, args.lowe_threshold) for descs in frames_descs]

    for i in range(0, args.num_iterations):
        print "Iteration %d" % i
        homo_matches = []
        homo_keys = []
        for [kp_mat, fixed_kp_mat, matches] in zip(frame_kp_mats, fixed_kp_mats, frames_matches):
            if len(matches) < args.min_matches:
                homo_matches.append([])
                homo_keys.append([])
                continue
            filtered_matches = matches;
            if args.homography_threshold > 0:
                filtered_matches = feat.filter_matches_by_homography(
                        pattern_kp_mat,
                        fixed_kp_mat,
                        filtered_matches,
                        args.homography_threshold)
                print "%d matches after homography filter" % len(filtered_matches)
            if args.f_threshold > 0:
                filtered_matches = feat.filter_matches_by_fundamental_matrix(
                        pattern_kp_mat,
                        fixed_kp_mat,
                        filtered_matches,
                        args.f_threshold)
                print "%d matches after fundamental matrix filter" % len(filtered_matches)

            homo_matches.append(filtered_matches)
            homo_keys.append(kp_mat)

        if len(homo_matches) == 0:
            print "Error: No frames with sufficient matches found"
            exit(1)

        # gather calibration data
        tmp = [(x,y) for (x,y) in zip(homo_matches, homo_keys) if len(x) >= args.min_matches]
        tmp = [calib.keypoint_matches_to_calibration_data(m, pattern_kp_mat, k) \
                for (m,k) in tmp]
        [obj_pts, img_pts] = zip(*tmp)

        # perform calibration
        [error, K, distortion, rvecs, tvecs] = cv2.calibrateCamera(obj_pts, img_pts, img_size)

        if i < args.num_iterations-1:
            # undistort image points, which will improve
            # homography filtering next iteration
            fixed_kp_mats = []
            for kp_mat in frame_kp_mats:
                tmp = np.reshape(kp_mat, (-1, 1, 2))
                tmp = cv2.undistortPoints(tmp, K, distortion, P=K)
                fixed_kp_mats.append(np.reshape(tmp, (-1, 2)))
     
    map(lambda (fname, m): feat.write_matches(fname, m), zip(out_match_fnames, homo_matches))

    cam = calib.Intrinsic_camera()
    cam.K = K
    cam.distortion = distortion
    cam.image_size = img_size
    calib.write_intrinsic_camera(args.out_camera_fname, cam)

if __name__ == "__main__":
    main();
