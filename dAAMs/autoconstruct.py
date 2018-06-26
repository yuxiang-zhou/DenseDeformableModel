import numpy as np
from menpofit.aam import HolisticAAM
from menpo.feature import igo
from menpofit.aam import LucasKanadeAAMFitter
from menpofit.fitter import align_shape_with_bounding_box
from menpo.shape import PointCloud
from menpofit.sdm import RegularizedSDM
from menpo.feature import hellinger_vector_128_dsift
import matplotlib.pyplot as plt
import multiprocessing
from menpofit.error import euclidean_distance_normalised_error
from menpo.shape import PointUndirectedGraph, TriMesh
from scipy.sparse import csr_matrix
import copy
import functools
import menpo.io as mio

def patch_features(x):
    return hellinger_vector_128_dsift(x / 255.0, dtype=np.float32)


# connectivity suppose to be in format [[pt1, pt2], ...]
def generate_adjacen_matrix(connectivity):
    max_shape = np.max(connectivity) + 1
    connectivity = np.vstack([connectivity, connectivity[:,-1::-1]])
    row_x, row_y = connectivity.T
    adjacency_matrix = csr_matrix(
        (
            [1] * len(row_x), (
                row_x,
                row_y
            )
        ), shape=(max_shape, max_shape)
    )
    return adjacency_matrix


def worker(arg, max_iters=10):
    fitter, img, shape, gt = arg
    return fitter.fit_from_shape(img,shape,gt_shape=gt, max_iters=max_iters)


def plot(errors):
    f, axis = plt.subplots(1,1,figsize=(12,5))
    axis.plot(errors)
    plt.tight_layout()
    plt.show()


def alignment_error(source, gt):
    def distance(shape, gts):
        return np.sqrt(np.sum(np.power(PointCloud(gts).bounding_box().range(), 2)))

    sample_source = PointCloud(np.vstack([np.mean(source.points[[29,30]],axis=0), source.points[[0,18]]]))

    return euclidean_distance_normalised_error(sample_source, gt, distance)


def mp_fit(DB, fitter, group='auto', max_iters=10, n_processes=8):
    fn = functools.partial(worker, max_iters=max_iters)

    pool = multiprocessing.Pool(processes=n_processes)
    frs = pool.map(fn, (
        [
            fitter,
            img,
            img.landmarks[group].lms,
            img.landmarks['PTS'].lms
        ] for img in DB
    ))
    pool.close()
    pool.join()

    # frs = []
    # for arg in [
    #     [
    #         fitter,
    #         img,
    #         img.landmarks[group].lms,
    #         img.landmarks['PTS'].lms
    #     ] for img in DB
    # ]:
    #     frs.append(fn(arg))

    return frs


def generative_construct(DB, fitter, trilist, label=None,
    fit_group='mean', train_group='final',
    feature=igo,
    diagonal=200,
    scales=(0.5, 1.0),
    n_processes=24,
    model_class=HolisticAAM,
    increament_model=None,
    original_shape_model=None,
    shape_forgetting_factor=1.0,
    appearance_forgetting_factor=1.0,
    max_iters=10,

):
    # fix appearance optimize shape
    error = []

#       Multi Processs Computation
    frs = mp_fit(DB, fitter, group=fit_group, n_processes=n_processes, max_iters=max_iters)

    for fr, img in zip(frs, DB):
        img.landmarks[train_group] = TriMesh(fr.final_shape.points, trilist=trilist)
        error.append(fr.final_error(alignment_error))
        if label:
            img.landmarks[label] = TriMesh(fr.final_shape.points, trilist=trilist)

    del fitter
    if increament_model:
        pdm = copy.deepcopy(increament_model)
        pdm.increment(DB, verbose=True, group=train_group,
            shape_forgetting_factor=shape_forgetting_factor,
            appearance_forgetting_factor=appearance_forgetting_factor)
    else:
        pdm = model_class(DB, holistic_features=feature, diagonal=diagonal, scales=scales, verbose=True, group=train_group)

    if original_shape_model:
        pdm.shape_models = original_shape_model

    if increament_model:
        del increament_model

    return pdm, np.mean(error)


def auto_construct(pdm, images, trilist=None,
    fit_group='init', train_group='final',
    models=[], errors=[], costs=[], isplot=False,
    feature=[igo] * 10,
    diagonal=200,
    scales=(0.5, 1.0),
    n_shape=[2, 4],
    n_appearance=[20, 30],
    max_iters=10,
    generative_iter=30,
    discriminative_iter = 10,
    n_processes=24,
    inc_appearance=0,
    model_class=HolisticAAM,
    increament=False,
    update_shape=False,
    shape_forgetting_factor=1.0,
    appearance_forgetting_factor=1.0,
    export_path=None
):

#     initialisation
    DB_size = len(images) / 2
    DB1 = images[:DB_size]
    DB2 = images[DB_size:]


    init_shape = pdm.shape_models[-1].model.mean()
    n_iteration = 0

    if trilist is None:
        trilist = TriMesh(init_shape.points).trilist

    for j in xrange(discriminative_iter):
        i_appearance = np.array(n_appearance) + np.array(inc_appearance)
        if (i_appearance > 1).any():
            i_appearance = i_appearance.astype(int).tolist()
        else:
            i_appearance = i_appearance.tolist()
# ------------ generative iterations -------------
        for i in xrange(generative_iter):
            print 'Discriminative Iter: {}, Generative Iter: {}'.format(j, i)

            aam_fitter = LucasKanadeAAMFitter(pdm, n_shape=n_shape, n_appearance=i_appearance)

            pdm, error = generative_construct(
                DB1,
                aam_fitter, trilist,
                fit_group=fit_group,
                train_group=train_group,
                label='iteration_{:03d}'.format(n_iteration),
                feature=feature[j],
                diagonal=diagonal,
                scales=scales,
                original_shape_model=None if update_shape else pdm.shape_models,
                n_processes=n_processes,
                model_class=model_class,
                increament_model=pdm if increament else None,
                shape_forgetting_factor=shape_forgetting_factor,
                appearance_forgetting_factor=appearance_forgetting_factor,
                max_iters=max_iters
            )

            n_iteration += 1
            models.append(pdm)
            errors.append(error)

            if export_path:
                mio.export_pickle([images, models, errors], export_path, overwrite=True)

            if isplot:
                plot(errors)

# ----------- discriminative iterations------------
        aam_fitter = LucasKanadeAAMFitter(pdm, n_shape=n_shape, n_appearance=i_appearance)

        frs = mp_fit(DB2, aam_fitter, group=fit_group, n_processes=n_processes, max_iters=max_iters)
        for img, fr in zip(DB2, frs):
            img.landmarks[train_group] = fr.final_shape

        sdm = RegularizedSDM(
            DB2,
            diagonal=diagonal,
            alpha=100,
            group=train_group,
            n_iterations=4,
            scales=(0.5,0.5,1.0,1.0),
            patch_features=patch_features,
            n_perturbations=30,
            patch_shape=[(25, 25), (15, 15), (15,15), (9,9)],
            verbose=True
        )

        pdm, error = generative_construct(
            DB1,
            sdm, trilist,
            fit_group=fit_group,
            train_group=train_group,
            label='discriminative_{:02d}'.format(j),
            original_shape_model=None if update_shape else pdm.shape_models,
            feature=feature[j],
            diagonal=diagonal,
            scales=scales,
            n_processes=n_processes,
            model_class=model_class,
            increament_model=pdm if increament else None,
            shape_forgetting_factor=shape_forgetting_factor,
            appearance_forgetting_factor=appearance_forgetting_factor,
            max_iters=max_iters
        )
        models.append(pdm)
        errors.append(error)

        if export_path:
            mio.export_pickle([images, models, errors], export_path, overwrite=True)

        if isplot:
            plot(errors)

    return models[-2]
