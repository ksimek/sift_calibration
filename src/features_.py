import cv2
import numpy as np

def read_features(in_fname):
    infile = open(in_fname, 'r')
    if infile is None:
        return None
    tokens = infile.readline().split()
    assert(len(tokens) == 2)
    num_pts = int(tokens[0])
    num_dims = int(tokens[1])

    descs = np.zeros((num_pts, num_dims), np.float32)

    kps = []

    values = map(float, infile.read().split())

    j = 0
    for i in range(0, num_pts):
        x = values[j]
        j += 1
        y = values[j]
        j += 1
        sz = values[j]
        j += 1
        angle = values[j]
        j += 1

        descs[i, :] = values[j:j+num_dims]
        j += num_dims

        kps.append(cv2.KeyPoint(x, y, sz, angle))

    return (kps, descs)

def write_features(out_fname, kps, descs):
    outfile = open(out_fname, 'w')
    num_pts = len(kps)
    num_dims = descs.shape[1]
    assert(descs.shape[0] == num_pts)

    line = "%d %d" % (num_pts, num_dims)
    outfile.write(line)
    
    for i in range(0, num_pts):
        kp = kps[i]
        desc = descs[i,:]

        outfile.write("\n%f %f %f %f" % (kp.pt[0], kp.pt[1], kp.size, kp.angle))
        for i in range(0, num_dims):
            if i % 20 == 0:
                outfile.write('\n')
            else:
                outfile.write(' ')
            outfile.write('%d' % desc[i])

def extract_features(fname, use_surf=True, upright=False):
    if use_surf:
        feat = cv2.SURF(400, 4, 2, True, upright)
    else:
        feat = cv2.SIFT()

    img = cv2.imread(fname, False)
    if img is None:
        return (None, None)
    [kp, desc] = feat.detectAndCompute(img,None)
    print "%d keypoints found" % len(kp)

    return (kp, desc)

def match_features(desc1, desc2, lowe_ratio=0.6):
    if desc1.shape[1] != desc2.shape[1]:
        raise Exception("incompatible feature vector dimensions")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    return good_matches

def filter_matches_by_fundamental_matrix(key_pts1, key_pts2, matches, threshold):
    N = len(matches)
    i = 0

    I = [x.queryIdx for x in matches]
    src_pts = key_pts1[I,:].copy()

    I = [x.trainIdx for x in matches]
    dst_pts = key_pts2[I,:].copy()

    N = len(matches)
    [H, mask] = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, threshold)

    mask = np.nonzero(np.reshape(mask, (-1)))[0]
    return [matches[i] for i in mask]

def filter_matches_by_homography(key_pts1, key_pts2, matches, threshold):
    N = len(matches)
    i = 0

    I = [x.queryIdx for x in matches]
    src_pts = key_pts1[I,:].copy()

    I = [x.trainIdx for x in matches]
    dst_pts = key_pts2[I,:].copy()

    N = len(matches)
    [H, mask] = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)

    mask = np.nonzero(np.reshape(mask, (-1)))[0]
    return [matches[i] for i in mask]
def read_matches(fname):
    f = open(fname, 'r')
    if f is None:
        return None
    tokens = f.readline().split()
    assert(len(tokens) == 1)
    num_matches = int(tokens[0])

    matches = []
    for i in range(0, num_matches):
        tokens = f.readline().split()
        assert(len(tokens) == 3)
        matches.append(cv2.DMatch(int(tokens[0]), int(tokens[1]), float(tokens[2])))

    return matches

def write_matches(fname, matches):
    f = open(fname, 'w')
    
    f.write("%d" % len(matches))

    for m in matches:
        f.write("\n%d %d %f" % (m.queryIdx, m.trainIdx, m.distance))

def appendimages(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    w1 = img1.shape[1]
    h1 = img1.shape[0]
    w2 = img2.shape[1]
    h2 = img2.shape[0]

    max_height = np.maximum(h1, h2)

    out_img = np.zeros((max_height, w1+w2, 3), np.uint8)
    out_img[0:h1, 0:w1, :] = img1
    out_img[0:h2, w1:w1+w2, :] = img2

    return out_img

def draw_matches(img1, img2, kp1, kp2, matches):
    img_out = appendimages(img1, img2)

    x_offset = img1.shape[1]

    for match in matches:
        i1 = match.queryIdx;
        i2 = match.trainIdx;
        p1 = kp1[i1].pt
        p2 = kp2[i2].pt
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]) + x_offset, int(p2[1]))

        cv2.line(img_out, p1, p2, (0,255,0))
    return img_out
