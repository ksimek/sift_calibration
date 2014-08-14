import numpy as np
import io_
import cv2

class Intrinsic_camera:
    """
    Intrinsically calibrated camera
    """
    def __init__(self):
        self.K = np.eye(3)
        self.distortion = np.zeros((1,5))
        self.image_size = (0,0)

class Calibrated_camera:
    """
    Fully calibrated camera (intrinsic and extrinsic)
    """
    def __init__(self, R = np.eye(3), T = np.zeros((3,1)), intr = Intrinsic_camera()):
        self.intrinsic = intr
        self.R = R
        self.T = T
    def to_mat(self):
        return np.dot(self.intrinsic.K, np.hstack([self.R, self.T]))

def write_calibrated_camera(fname, cam):
    f = open(fname, 'w')
    f.write("K:\n")
    f.write(io_.mat_to_str(cam.intr.K))
    f.write("R:\n")
    f.write(io_.mat_to_str(cam.R))
    f.write("T:\n")
    f.write(io_.mat_to_str(cam.T))
    f.write("\ndistortion:\n")
    f.write(io_.mat_to_str(cam.intr.distortion))
    f.write("\nimage_size:\n")
    f.write(" ".join(map(str, cam.intr.image_size)))
    f.write("\n")
    f.close()


def write_intrinsic_camera(fname, cam):
    f = open(fname, 'w')
    f.write("K:\n")
    f.write(io_.mat_to_str(cam.K))
    f.write("\ndistortion:\n")
    f.write(io_.mat_to_str(cam.distortion))
    f.write("\nimage_size:\n")
    f.write(" ".join(map(str, cam.image_size)))
    f.write("\n")
    f.close()

def read_intrinsic_camera(fname):
    cam = Intrinsic_camera() 
    f = open(fname, 'r')
    if f is None:
        print "file not found: " + fname
        exit(1)
    line = f.readline().strip()
    assert(line == "K:")
    line = f.readline().strip();
    assert(line == "6 3 3")
    cam.K = np.array([map(float, f.readline().split()) for x in range(0,3)])

    line = f.readline().strip()
    assert(line == "distortion:")
    tokens = map(int, f.readline().split());
    assert(len(tokens) == 3)
    assert(tokens[0] == 6)
    assert(tokens[1] == 1)
    cam.distortion = np.array(map(float, f.readline().split()))
    line = f.readline().strip()
    assert(line == "image_size:")
    cam.image_size = tuple(map(int, f.readline().split()));
    assert(len(cam.image_size) == 2)
    return cam

def read_calibrated_camera(fname):
    cam = Intrinsic_camera() 
    f = open(fname, 'r')
    line = f.readline().strip()
    assert(line == "K:")
    line = f.readline().strip();
    assert(line == "6 3 3")
    cam.intr.K = np.array([map(float, f.readline().split()) for x in range(0,3)])

    line = f.readline().strip();
    assert(line == "R:")
    line = f.readline().strip();
    assert(line == "6 3 3")
    cam.R = np.array([map(float, f.readline().split()) for x in range(0,3)])

    line = f.readline().strip();
    assert(line == "T:")
    line = f.readline().strip();
    assert(line == "6 3 1")
    cam.T = np.array([map(float, f.readline().split()) for x in range(0,3)])

    line = f.readline().strip()
    assert(line == "distortion:")
    tokens = map(int, f.readline().split());
    assert(len(tokens) == 3)
    assert(tokens[0] == 6)
    assert(tokens[1] == 1)
    cam.intr.distortion = np.array(map(float, f.readline().split()))

    line = f.readline().strip()
    assert(line == "image_size:")
    cam.intr.image_size = tuple(map(int, f.readline().split()));
    assert(len(cam.image_size) == 2)

def keypoint_matches_to_calibration_data(matches, pts1, pts2):
    if type(pts1[0]) is cv2.KeyPoint:
        assert(type(pts2[0]) is cv2.KeyPoint)
        obj_pts = np.array([pts1[x.queryIdx].pt for x in matches])
        img_pts = np.array([pts2[x.trainIdx].pt for x in matches])
    else:
        assert(type(pts1[0]) is np.ndarray)
        assert(type(pts2[0]) is np.ndarray)
        I = [x.queryIdx for x in matches]
        obj_pts = pts1[I,:]
        I = [x.trainIdx for x in matches]
        img_pts = pts2[I,:]

    # add a zero z-coordinate
    N = len(matches)
    obj_pts = np.hstack([obj_pts, np.zeros((N, 1))])

    obj_pts = obj_pts.astype(np.float32)
    img_pts = img_pts.astype(np.float32)

    return (obj_pts, img_pts)

