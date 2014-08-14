#!/usr/bin/env python

import argparse
import features_ as feat
import os.path
import numpy as np
import cv2
import calibration_ as calib

def matches_to_calib_pts(matches, pts1, pts2):
    assert(len(matches) > 0)

    obj_pts = np.array([pts1[x.queryIdx].pt for x in matches])
    img_pts = np.array([pts2[x.trainIdx].pt for x in matches])

    # add a zero z-coordinate
    N = len(matches)
    obj_pts = np.hstack([obj_pts, np.zeros((N, 1))])

    obj_pts = obj_pts.astype(np.float32)
    img_pts = img_pts.astype(np.float32)

    return (obj_pts, img_pts)
        
def to_calibration_data(match_sets, obj_keys, img_keysets, min_matches):
    match_keys = zip(match_sets, img_keysets)
    match_keys = [x for x in match_keys if len(x[0]) >= min_matches]

    calib_pts = [matches_to_calib_pts(m, obj_keys, k) \
        for (m,k) in match_keys]

    return zip(*calib_pts)

def calibrate_intrinsic(obj_pts, img_pts, img_size):
    [error, K, distortion, rvecs, tvecs] = cv2.calibrateCamera(obj_pts, img_pts, img_size)
    cam = calib.Intrinsic_camera()
    cam.K = K
    cam.distortion = distortion
    cam.image_size = img_size
    
    return cam

def main():
    parser = argparse.ArgumentParser(description="Calibrate using keypoint matches");
    parser.add_argument('pattern_key_fname', type=str, help='reference pattern keypoint file')
    parser.add_argument('num_frames', type=int, help='Number of scene frames')
    parser.add_argument('scene_key_fmt', type=str, help='Printf-formatted string representing keypoint files from N scene frames')
    parser.add_argument('match_fmt', type=str, help='Printf-formatted string representing keypoint match files')
    parser.add_argument('example_image', type=str, help='Example image to get dimensions from')
    parser.add_argument('out_fname', type=str, help='filename for intrinsic calibrated camera')
    parser.add_argument('--min_matches', default=20, type=str, help='omit frames with fewer than N matches')

    args = parser.parse_args()

    pattern_keys = feat.read_features(args.pattern_key_fname)[0];

    fnames = [(args.scene_key_fmt % x, args.match_fmt % x) for x in range(1,args.num_frames+1)]
    fnames = [x for x in fnames if os.path.isfile(x[0])]

    if len(fnames) == 0:
        print "No matching keypoint files"
        exit(1)

    missing = next((x[1] for x in fnames if not os.path.isfile(x[1])), None)
    if missing is not None:
        print "File not found: %s" % missing
        exit(1)

    [frame_fnames, match_fnames] = zip(*fnames)

    print "reading keypoint from %d frames" % len(frame_fnames)
    frames_keys = [feat.read_features(x)[0] for x in frame_fnames]
    print "reading matches from %d frames" % len(match_fnames)
    frames_matches = [feat.read_matches(x) for x in match_fnames]

    img = cv2.imread(args.example_image)
    img_size = (img.shape[1], img.shape[0]);


    [obj_pts, image_pts] = to_calibration_data(frames_matches, pattern_keys, frames_keys, args.min_matches)

    cam = calibrate_intrinsic(obj_pts, image_pts, img_size)

    calib.write_intrinsic_camera(args.out_fname, cam);

if __name__ == "__main__":
    main();
