import os
import glob

import numpy as np
import cv2
import menpo.io as mio

from menpo.image import Image, BooleanImage
from menpo.shape import TriMesh, PointCloud

from .lineerror import interpolate
from skimage import filters

# IO

def import_image(img_path):
    img = cv2.imread(str(img_path))
    original_image = Image.init_from_channels_at_back(img[:,:,-1::-1])

    try:
        original_image_lms = mio.import_landmark_file('{}/{}.ljson'.format(img_path.parent, img_path.stem)).lms.points.astype(np.float32)
        original_image.landmarks['LJSON'] = PointCloud(original_image_lms)
    except:
        pass

    return original_image

# binary based shape descriptor ----------------------------
def sample_points(target, range_x, range_y, edge=None, x=0, y=0):
    ret_img = Image.init_blank((range_x, range_y))

    if edge is None:
        for pts in target:
            ret_img.pixels[0, pts[0]-y, pts[1]-x] = 1
    else:
        for eg in edge:
            for pts in interpolate(target[eg], 0.1):
                try:
                    ret_img.pixels[0, int(pts[0]-y), int(pts[1]-x)] = 1
                except:
                    print('Index out of Bound')
                    print(pts)

    return ret_img

def binary_shape(pc, xr, yr, groups=None):
    return sample_points(pc.points, xr, yr, groups)


def distance_transform_shape(pc, xr, yr, groups=None):
    rsi = binary_shape(pc, xr, yr, groups)
    ret = distance_transform_edt(rsi.rolled_channels().squeeze())

    return Image.init_from_rolled_channels(ret.T)


def shape_context_shape(pc, xr, yr, groups=None, sampls=None, r_inner=0.125, r_outer=2, nbins_r=5, nbins_theta=12):
    nbins = nbins_r*nbins_theta

    def get_angle(p1,p2):
        """Return angle in radians"""
        return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))

    def compute_one(pt, points, mean_dist):

        distances = np.array([euclidean(pt,p) for p in points])
        r_array_n = distances / mean_dist

        r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)

        r_array_q = np.zeros(len(points))
        for m in xrange(nbins_r):
           r_array_q +=  (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        def _get_angles(self, x):
            result = zeros((len(x), len(x)))
            for i in xrange(len(x)):
                for j in xrange(len(x)):
                    result[i,j] = get_angle(x[i],x[j])
            return result

        theta_array = np.array([get_angle(pt,p)for p in points])
        # 2Pi shifted
        theta_array_2 = theta_array + 2*math.pi * (theta_array < 0)
        theta_array_q = 1 + np.floor(theta_array_2 /(2 * math.pi / nbins_theta))

        sn = np.zeros((nbins_r, nbins_theta))
        for j in xrange(len(points)):
            if (fz[j]):
                sn[r_array_q[j] - 1, theta_array_q[j] - 1] += 1

        return sn.reshape(nbins)


    rsi = binary_shape(pc, xr, yr, groups)
    pixels = rsi.pixels.squeeze()
    pts = np.argwhere(pixels > 0)
    if sampls:
        pts = pts[np.random.randint(0,pts.shape[0],sampls)]
    mean_dist = dist([0,0],[xr,yr]) / 2

    sc = np.zeros((xr,yr,nbins))

    for x in xrange(xr):
        for y in xrange (yr):
            sc[x,y,:] = compute_one(np.array([x,y]), pts, mean_dist)

    return Image.init_from_rolled_channels(sc)
# ----------------------------------------------------------


# SVS based Shape Descriptor -------------------------------

def multi_channel_svs(svs_pts, h,w, groups,c=3):
    msvs = Image.init_blank((h,w), n_channels=len(groups))
    for ch,g in enumerate(groups):
        if len(g):
            msvs.pixels[ch, ... ] = svs_shape(svs_pts, h,w, groups=[g],c=c).pixels[0]
    msvs.pixels /= np.max(msvs.pixels)
    return msvs

def svs_shape(pc, xr, yr, groups=None, c=1):
    store_image = Image.init_blank((xr,yr))
    ni = binary_shape(pc, xr, yr, groups)
    store_image.pixels[0,:,:] = filters.gaussian(np.squeeze(ni.pixels), c)
    return store_image


def hog_svs_shape(pc, xr, yr, groups=None):
    store_image = hog(svs_shape(pc, xr, yr, groups))
    return store_image


def sift_svs_shape(pc, xr, yr, groups=None):
    store_image = dsift(svs_shape(pc, xr, yr, groups))
    return store_image


def sample_gaussian_shape(pc, xr, yr, groups=None):
    ni = Image.init_blank((xr, yr))
    for pts in pc.points:
        ni.pixels[0, pts[0], pts[1]] = 1
    store_image = Image.init_blank(ni.shape)
    store_image.pixels[0,:,:] = filters.gaussian(np.squeeze(ni.pixels), 4)
    return store_image
