from sklearn import svm
from menpo.shape import PointCloud
from menpo.shape import TriMesh
from menpo.image import MaskedImage, Image
from menpo.visualize.base import Viewable
from scipy.spatial.distance import euclidean as dist

import numpy as np



class SVS(Viewable):
    def __init__(self, points, tplt_edge=None, nu=0.5, kernel='rbf', gamma=0.03,
                 tolerance=0.5, max_f=5, bg_sample_rate=0.0, xr=None, yr=None):
        self.points = points
        self.reject_dist = tolerance * max_f
        if len(points) > 0:
            self._build(nu, kernel, gamma, tolerance, tplt_edge, max_f, bg_sample_rate, xr, yr)

    def _build(self, nu, kernel, gamma, tolerance, tplt_edge, max_f, bg_sample_rate, xr, yr):
        accept_rate = 0.5
        margin = 10
        sample_step = 1
        min_p = np.min(self.points, axis=0).astype('int')
        max_p = np.max(self.points, axis=0).astype('int')

        if xr:
            self._range_x = range_x = xr
        else:
            self._range_x = range_x = np.arange(
                min_p[0]-margin, max_p[0]+margin, sample_step
            )

        if yr:
            self._range_y = range_y = yr
        else:
            self._range_y = range_y = np.arange(
                min_p[1]-margin, max_p[1]+margin, sample_step
            )

        # Generate negtive points
        # Build Triangle Mesh
        if tplt_edge is None:
            tplt_tri = TriMesh(self.points).trilist

            # Generate Edge List
            tplt_edge = tplt_tri[:, [0, 1]]
            tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [0, 2]]))
            tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [1, 2]]))
            tplt_edge = np.sort(tplt_edge)

            # Get Unique Edge
            b = np.ascontiguousarray(tplt_edge).view(
                np.dtype(
                    (np.void, tplt_edge.dtype.itemsize * tplt_edge.shape[1])
                )
            )
            _, idx = np.unique(b, return_index=True)
            tplt_edge = tplt_edge[idx]

        # Sample Points
        training_points_negative = []
        training_points_positive = []

        for i in range_x:
            for j in range_y:
                valid = True
                max_dist = 100*tolerance
                for e in tplt_edge:
                    min_dist = minimum_distance(
                        self.points[e[0]],
                        self.points[e[1]],
                        np.array([i, j]),
                        accept_rate
                    )
                    if min_dist < max_dist:
                        max_dist = min_dist
                    if min_dist < tolerance:
                        valid = False
                    if min_dist < accept_rate:
                        training_points_positive.append([i, j])
                        break

                if valid and max_dist < max_f*tolerance:
                    training_points_negative.append([i, j])

                if np.random.rand() < bg_sample_rate:
                    training_points_negative.append([i, j])

        training_points_negative = np.array(training_points_negative)
        training_points_positive = np.vstack((
            np.array(training_points_positive), self.points
        ))

        self._positive_pts = training_points_positive
        self._negative_pts = training_points_negative

        # Build SVS
        n = training_points_positive.shape[0]
        m = training_points_negative.shape[0]

        training_points = np.vstack((training_points_positive,
                                     training_points_negative))
        classification = np.hstack((np.ones(n), np.zeros(m)))

        weights = classification*(m/n-1) + 1

        svs = svm.NuSVC(nu=nu, kernel=kernel, gamma=gamma)
        svs.fit(training_points, classification, sample_weight=weights)

        self.svs = svs

    def view_samples(self):
        if len(self.points) > 0:
            PointCloud(self._negative_pts).view(marker_face_colour='b')
            PointCloud(self._positive_pts).view(marker_face_colour='w')

    def view(self, xr=None, yr=None):
        self.svs_image(xr, yr).view()

    def svs_image(self, xr=None, yr=None):
        w = len(xr)
        h = len(yr)
        img = Image.init_blank((w, h))
        if len(self.points) > 0:
            for i, x in enumerate(xr):
                for j, y in enumerate(yr):
                    img.pixels[0, i, j] = self.svs.decision_function([[x, y]])[0]

            minp = np.min(img.pixels)
            maxp = np.max(img.pixels)
            img.pixels = (img.pixels - minp) / (maxp-minp)

        return img


class MultiSVS(Viewable):
    """docstring for MultiSVS"""
    def __init__(self, points_list, tplt_edge_lists=None, nu=0.5, kernel='rbf', gamma=0.03,
                 tolerance=0.5, max_f=5, bg_sample_rate=0.0,xr=None, yr=None):


        if not tplt_edge_lists:
            tplt_edge_lists = []
            for points in points_list:
                if len(points) > 0:
                    tplt_edge = np.arange(len(points)).reshape(-1,1)
                    tplt_edge = np.hstack([tplt_edge[:-1,:],tplt_edge[1:,:]])
                    tplt_edge_lists.append(tplt_edge)
                else:
                    tplt_edge_lists.append([])

        self.svs_list = [SVS(points, tplt_edge, nu, kernel, gamma, tolerance, max_f, bg_sample_rate,xr,yr) for points,tplt_edge in zip(points_list, tplt_edge_lists)]

    def svs_image(self, xr=None, yr=None):
        w = len(xr)
        h = len(yr)
        img = Image.init_blank((w, h), n_channels=len(self.svs_list))
        for k,svs in enumerate(self.svs_list):
            img.pixels[k,...] = svs.svs_image(xr=xr,yr=yr).pixels

        return img

    def view(self, xr=None, yr=None, is_merge=False):
        img = self.svs_image(xr, yr)
        if is_merge:
            img = Image(np.sum(img.pixels, axis=0))
        img.view()




def minimum_distance(v, w, p, tolerance=1):
#     Return minimum distance between line segment (v,w) and point p
    l2 = dist(v, w)  # i.e. |w-v|^2 -  avoid a sqrt
    if l2 == 0.0:
        return dist(p, v)

#     Consider the line extending the segment, parameterized as v + t (w - v).
#     We find projection of point p onto the line.
#     It falls where t = [(p-v) . (w-v)] / |w-v|^2
    t = np.dot((p - v) / l2, (w - v) / l2)
    if t < 0.0:
        return dist(p, v) + tolerance      # // Beyond the 'v' end of the segment
    elif t > 1.0:
        return dist(p, w) + tolerance  # // Beyond the 'w' end of the segment

    projection = v + t * (w - v) # // Projection falls on the segment
    return dist(p, projection)
