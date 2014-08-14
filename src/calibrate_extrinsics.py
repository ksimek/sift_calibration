#!/usr/bin/env python 
import pickle
import argparse
import features_ as feat
import os.path
import numpy as np
import cv2
import calibration_ as calib
from tree_ import Tree
import sys


class My_match:
    """
    Imitation of cv2.DMatch, but this one
    is serializable using pickle
    """
    def __init__(self, dmatch):
        self.queryIdx = dmatch.queryIdx
        self.trainIdx = dmatch.trainIdx
        self.distance  = dmatch.distance

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate using keypoint matches");
    parser.add_argument('pattern_keys', type=str, help='reference pattern keypoint file')
    parser.add_argument('num_cams', type=int, help='number of cameras')
    parser.add_argument('num_frames', type=int, help='number of frames (missing frames are okay)')
    parser.add_argument('keys_fmt', type=str, help='printf string for keypoint files')
    parser.add_argument('cams_fmt', type=str, help='printf string for camera intrinsic parameters (two fields: cam, then frame)')
    parser.add_argument('matches_fmt', type=str, help='printf string for matches from pattern to cameras (two fields: cam, then frame)')
    parser.add_argument('camera_configuration', type=str, help='int-sequence or Tree-string representing the adjacency of cameras.  e.g. 1 2 3 4 5  or  1 2 nil 3 4 nil 5 6 nil nil nil nil')
    parser.add_argument('out_bal_file', type=str, help='output file to pass to ceres-solver (follows format from paper, "Bundle Adjustment in the Large")')
    parser.add_argument('--min_matches', type=int, default=5, help='minimum number of matches between cameras for a frame to be accepted')
    parser.add_argument('--min_visible', type=int, default=3, help='minimum number of views for a point to be accepted')
    parser.add_argument('--sparse_files', action='store_true', help='minimum number of views for a point to be accepted')

    return parser.parse_args()
      
def read_intrinsic_cams(num_cams, fmt):
    I = range(1,num_cams+1)
    fnames = [fmt % x for x in I]
    cams = [calib.read_intrinsic_camera(x) for x in fnames]
    return cams

# Generic reader for files corresponding to (cam, frame) pairs.
# Missing frames are omitted from the list, and indices of present
# frames are returned along with the objects themselves.
def read_cams_frames_(num_cams, num_frames, fmt, callback, sparse):
    I = range(0,num_cams)
    J = range(0,num_frames)

    fnames = [ [fmt % (i+1,j+1) for j in J] for i in I]
    if sparse:
        K = [ [j for j in J if os.path.isfile(fnames[i][j])] for i in I]
        fnames = [ [fnames[i][k] for k in K[i]] for i in I]
    objects = [[callback(fn) for fn in cam_fnames] for cam_fnames in fnames]
    return (objects, K)

def keys_to_mat(keys):
    return np.array([list(x.pt) for x in keys])

def read_matches(num_cams, num_frames, fmt, sparse):
    return read_cams_frames_(num_cams, num_frames, fmt, lambda x: [My_match(m) for m in feat.read_matches(x)], sparse)

def read_keys(num_cams, num_frames, fmt, sparse):
    return read_cams_frames_(num_cams, num_frames, fmt, lambda x: keys_to_mat(feat.read_features(x)[0]), sparse)

def cam_frames_indices_match(I, J):
    if len(I) != len(J):
        return false

    return all(i == j for (i,j) in zip(I,J))

def intersect_matches(m1, m2):
    I = dict((x.queryIdx, i) for i, x in enumerate(m1))
    J = dict((x.queryIdx, i) for i, x in enumerate(m2))
    K = set(I.keys()).intersection(J.keys())

    new_matches = [m1[I[k]] for k in K]
    new_matches_2 = [m2[J[k]] for k in K]

    return (new_matches, new_matches_2)




def calibrate_pair(pattern_keys, data_1, data_2, min_matches):
    [cam1, keys1, matches1, frame_indices1, cam_ind1] = data_1
    [cam2, keys2, matches2, frame_indices2, cam_ind2] = data_2

    if frame_indices1 is None:
        assert(frame_indices2 is None)
    else:
        # filter frames where both views have data
        I = dict((x, i) for i, x in enumerate(frame_indices1))
        J = dict((x, i) for i, x in enumerate(frame_indices2))
        K = set(I.keys()).intersection(J.keys())

        matches1 = [matches1[I[k]] for k in K]
        matches2 = [matches2[J[k]] for k in K]

        keys1 = [keys1[I[k]] for k in K]
        keys2 = [keys2[J[k]] for k in K]
        if(len(matches1)  == 0):
            print "Error: No overlapping frames between views %d and %d -- double check 'order' parameter" % (cam1, cam2)
            exit(1)

    # prune non-mutual matches
    tmp = [intersect_matches(m1, m2) for (m1, m2) in zip(matches1, matches2)]
    [matches1, matches2] = zip(*tmp)
    I = [i for i,x in enumerate(matches1) if len(x) >= min_matches]

    # prune frames with insufficient matches
    matches1 = [matches1[i] for i in I]
    matches2 = [matches2[i] for i in I]
    keys1 = [keys1[i] for i in I]
    keys2 = [keys2[i] for i in I]

    if(len(matches1)  == 0):
        print "Error: pair (%d, %d) have no shared keypoints" % (cam_ind1+1, cam_ind2+1)
        exit(1)
    print "pair (%d, %d):  %d matches" % (cam_ind1+1, cam_ind2+1, len(matches1))

    to_data = calib.keypoint_matches_to_calibration_data

    tmp = [to_data(m, pattern_keys, k) for (m,k) in zip(matches1, keys1)]
    [obj_pts, img_pts1] = zip(*tmp)

    tmp = [to_data(m, pattern_keys, k) for (m,k) in zip(matches2, keys2)]
    [obj_pts2, img_pts2] = zip(*tmp)

    assert(all([(a == b).all() for (a,b) in zip(obj_pts, obj_pts2)]))

    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC

    error,_,_,_,_,R,T,_,_ = cv2.stereoCalibrate(obj_pts, 
            img_pts1, img_pts2,
            (0,0),
            cam1.K, cam1.distortion,
            cam2.K, cam2.distortion,
            flags=flags)

    return calib.Calibrated_camera(R=R, T=T, intr=cam2)

def apply_extrinsics(cam1, cam2):
    """
    Apply the extrinsic transformations from camera 1 to camera 2,
    overwriting camera 2
    """
    cam2.T = cam2.T + np.dot(cam2.R, cam1.T)
    cam2.R = np.dot(cam2.R, cam1.R)
    return cam2

def pairwise_calib(pattern_keys, intrinsic_cams, keys, matches, frame_indices, config_tree, min_matches):
#EROROR: todo: handle fact that keys correspond to different frames
    assert(len(keys) == len(intrinsic_cams))
    assert(len(matches) == len(intrinsic_cams))
    calib_set = zip(intrinsic_cams, keys, matches, frame_indices, range(0,len(intrinsic_cams)))
    calib_data_tree = config_tree.dfs(lambda x: calib_set[x])

    root_intr = intrinsic_cams[config_tree.value]
    root_camera = calib.Calibrated_camera(intr=root_intr)

    calib_tree = calib_data_tree.dfs_pair(lambda x, y: calibrate_pair(pattern_keys, x, y, min_matches), root_value=root_camera)
    calib_tree.dfs_pair_inplace(apply_extrinsics)

    cams = []
    calib_tree.dfs(lambda x: cams.append(x))
    inds = []
    config_tree.dfs(lambda x: inds.append(x))

    tmp = sorted(zip(cams, inds), key=lambda x: x[1])
    cams = zip(*tmp)[0]
    return cams

def multi_triangulate(cams, pts, views, max_iter=1, stop_epsilon=1e-6):
    A = np.zeros((len(views)*2, 4))
    prev_weights = np.ones((len(views)))
    weights = np.ones((len(views)))

    assert(len(views) == len(pts))

    for it in range(0, max_iter):
        for i in range(0, len(views)):
            pt = pts[i]
            vi = views[i]
            M = cams[vi]
            r = 2*i
            A[r,:]   = (pt[0] * M[2,:] - M[0,:]) / weights[i]
            A[r+1,:] = (pt[1] * M[2,:] - M[1,:]) / weights[i]
        A_ = A[:,0:3]
        B_ = -A[:,3:4]
#        B_ *= -1
#        out = np.linalg.lstsq(A_, B_)[0] # alternative
        out = cv2.solve(A_,B_,flags=cv2.DECOMP_SVD)[1]
        for i in range(0, len(views)):
            vi = views[i]
            M = cams[vi]
            # compute w-component of projected point
            weights[i] = np.dot(M[2,0:3], out) + M[2,3]
        tmp = weights - prev_weights
        lo = np.min(tmp)
        hi = np.max(tmp)
        hi = max(-lo, hi)
        if hi <= stop_epsilon:
            break
        prev_weights,weights = weights,prev_weights
    return out

def group_and_triangulate_matches(cams, num_pts, keys, matches, frames_indices, min_visible):
    """
    group mutually matching points and triangulate them.  
    """
    cam_mats = map(lambda x: x.to_mat(), cams)
    num_cams = len(cams)
    assert(num_cams == len(keys))
    assert(num_cams == len(matches))

    num_frames = len(matches[0])
    assert(len(keys[0]) == num_frames)

    all_pts3d = []
    all_pt_matches = []

    # for each frame, which cameras are present?
    if frame_indices[0] is None:
        assert(all(x is None for x in frame_indices))

        # stanard case: all cameras are in all frames
        frame_cams = range(0,num_cams) * num_frames;
    else:
        assert(len(frame_indices) == num_cams)
        frame_cams = [list() for _ in range(0, num_frames)]

        for cam_i in range(0, num_frames):
            for fi in frame_indices[cam_i]:
                frame_cams[fi].append(cam_i)

    for frame_i in range(0, num_frames):
        # gather matches in all cameras
        for cam_i in frame_cams[frame_i]:
            cur_matches = matches[cam_i][frame_i]
            print len(cur_matches) 
            for m in cur_matches:
                pt = keys[cam_i][frame_i][m.trainIdx,:]
                pt_matches[m.queryIdx].append((cam_i, pt))

        pt_matches = [x for x in pt_matches if len(x) >= min_visible]

        # triangulate matches and save 
        print "Triangulating %d points" % len(pt_matches)
        for i, pt_match in enumerate(pt_matches):
            num_obs = len(pt_match)
            views, pts = zip(*pt_match)
            pt3d = multi_triangulate(cam_mats, pts, views)
            all_pts3d.append(pt3d)
            all_pt_matches.append(pt_match)
            assert(len(views) == num_obs)
            assert(len(pts) == num_obs)
            print '\r%d' % i,
            sys.stdout.flush()

    return all_pt_matches,all_pts3d

def write_bal_file(fname, cams, matches, pts3d):
    """
    write to a text file following the format from
    "Bundle Adjustment in the Large" Argawal, et al.
    http://grail.cs.washington.edu/projects/bal/
    """
    assert(len(matches) == len(pts3d))
    
    num_observations = sum([len(x) for x in matches])
    f = open(fname)
    assert(f is not None)

    f.write("%d %d %d\n" % len(cams), len(pts3d), num_observations)

    for i, pt_matches in enumerate(pts_matches):
        for m in pt_matches:
            cam_index = m[0]
            pt = m[1]
            f.write("%d %d %f %f\n" % (cam_index, i, pt[0], pt[1]))

    for c in cams:
        R = cv2.Rodrigues(c.R)
        T = c.T
        f = (c.intr.K[0,0] + c.intr.K[1,1])/2
        k1 = c.intr.distortion[0]
        k2 = c.intr.distortion[1]
        f.write("%d %d %d " % (R[0], R[1], R[2]))
        f.write("%d %d %d " % (T[0], T[1], T[2]))
        f.write("%d %d %d\n" % (f, k1, k2))

    for pt in pts3d:
        f.write("%d %d %d\n" % (pt[0], pt[1], pt[2]))
    f.close()

def get_config_tree(config_str):
    config_tokens = config_str.split()

    if 'nil' not in config_tokens:
        config_str += ' ' + ' '.join(['nil'] * len(config_tokens))

    tree = Tree()
    tree.parse(config_str, int)
    # convert from 1-indexed to zero-indexed
    tree = tree.dfs(lambda x: x-1)

    return tree

def main():
    args = parse_args()

    dumping = False
    if dumping:
        pattern_keys = keys_to_mat(feat.read_features(args.pattern_keys)[0])

        intrinsic_cams = read_intrinsic_cams(args.num_cams, args.cams_fmt)

        print "Reading keys"
        [keys, I_keys] = read_keys(args.num_cams, args.num_frames, args.keys_fmt, args.sparse_files)

        print "Reading matches"
        [matches, I_matches] = read_matches(args.num_cams, args.num_frames, args.matches_fmt, args.sparse_files)
        assert(cam_frames_indices_match(I_keys, I_matches))

        if not args.sparse_files:
            assert(all([len(x) == args.num_frames for x in I_keys]))
            frame_indices = None * args.num_cams
        else:
            frame_indices = I_keys

        pickle.dump((pattern_keys, intrinsic_cams, keys, matches, frame_indices), open("debug.p", 'wb'))
    else:
        (pattern_keys, intrinsic_cams, keys, matches, frame_indices) = pickle.load(open("debug.p", 'rb'))

    config_tree = get_config_tree(args.camera_configuration)
    print config_tree.to_string()

    cams = pairwise_calib(pattern_keys, intrinsic_cams, keys, matches, frame_indices, config_tree, args.min_matches)

    [grouped_matches, pts3d] = group_and_triangulate_matches(cams, len(pattern_keys), keys, matches, frame_indices, args.min_visible)
    write_bal_file(args.out_fname, cams, grouped_matches, pts3d)


if __name__ == "__main__":
    main()

