"""Calculate totals given minimal area request."""
import argparse
import logging
import os
import multiprocessing
import time

import pygeoprocessing
import numpy
import scipy.ndimage.morphology
import taskgraph
from osgeo import gdal
from osgeo import osr


gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

LOGGER = logging.getLogger(__name__)


def _make_test_data_smooth(dir_path, n, m):
    """Make random raster test data."""
    os.makedirs(dir_path, exist_ok=True)
    raster_path_list = []
    for raster_path, (pi, pj) in [
            (f'{index}.tif', (int(n*index/m), 0)) for index in range(m)]:
        base_array = numpy.ones((n, n))
        base_array[pi, pj] = 0
        dist_array = scipy.ndimage.morphology.distance_transform_edt(
            base_array)
        pygeoprocessing.numpy_array_to_raster(
            dist_array, -1, (1, -1), (0, 0),
            osr.SRS_WKT_WGS84_LAT_LONG, raster_path)
        raster_path_list.append(raster_path)
    return raster_path_list


def _make_test_data_random(dir_path, n, m):
    """Make random raster test data."""
    os.makedirs(dir_path, exist_ok=True)
    raster_path_list = []
    for raster_path, (pi, pj) in [
            (f'{index}.tif', (int(n*index/m), 0)) for index in range(m)]:
        base_array = numpy.random.random((n, n))
        pygeoprocessing.numpy_array_to_raster(
            base_array, -1, (1, -1), (0, 0),
            osr.SRS_WKT_WGS84_LAT_LONG, raster_path)
        raster_path_list.append(raster_path)
    return raster_path_list


def _callback(prob):
    LOGGER.debug(f'phase: {prob.phase}, status: {prob.status} {prob.message}')


def _sum_raster(raster_path):
    """Return the non-nodata sum of the raster."""
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    nodata = raster_info['nodata'][0]
    running_sum = 0.0
    for _, array in pygeoprocessing.iterblocks((raster_path, 1)):
        if nodata is not None:
            valid_mask = (array != nodata)
        else:
            valid_mask = numpy.ones(array.shape, dtype=bool)
        running_sum += numpy.sum(array[valid_mask])
    return running_sum


def multigrid_optimize(
        raster_path_list, min_proportion, target_raster_path,
        win_xoff, win_yoff, win_xsize, win_ysize,
        prop_tol=1e-12, grid_size=64,):
    """Solve a multigrid optimization problem.

    Args:
        raster_path_list (list): list of rasters of equal shape to
            use for value objective matching.
        min_proportion (float): minimum proportion of rasters to optimize
            for.
        target_raster_path (str): path to target mask raster
        win_xoff, win_yoff, win_xsize, win_ysize (int): the offset into the
            global rasters to solve a sub optimization problem on.
        prop_tol (float): propotion of raster sum to use as tolerance.
        grid_size (int): the size to subdivide the optimization problem on.

    Return:
        {
            'objective_sum_list': [sum of selected objectives],
            'proportion_list': [proportion of raster selected],
            'area_list': [area of raster selected]
        }, runtime in seconds
    """
    start_time = time.time()
    raster_info = pygeoprocessing.get_raster_info(raster_path_list[0])

    n_cols, n_rows = raster_info['raster_size']

    col_stepsize = max(win_xsize // grid_size, 1)
    row_stepsize = max(win_ysize // grid_size, 1)

    n_col_grids = int(numpy.ceil(win_xsize / col_stepsize))
    n_row_grids = int(numpy.ceil(win_xsize / row_stepsize))
    mask_array = numpy.full(
        (n_row_grids, n_col_grids), -1, dtype=numpy.float32)

    A_list = [[] for _ in range(len(raster_path_list))]
    raster_sum_list = [0.0] * len(raster_path_list)
    b_list = []
    offset_list = []
    for x_index in range(n_col_grids):
        local_xoff = win_xoff + x_index * col_stepsize
        local_win_xsize = col_stepsize
        next_xoff = win_xoff + (x_index+1)*col_stepsize
        if next_xoff > n_cols:
            local_win_xsize += n_cols-next_xoff

        for y_index in range(n_row_grids):
            local_yoff = win_yoff + y_index * row_stepsize
            local_win_ysize = row_stepsize
            next_yoff = win_yoff + (y_index+1)*row_stepsize
            if next_yoff > n_rows:
                local_win_ysize += n_rows-next_yoff

            tol = 1.0

            # load the subgrid
            offset_dict = {
                'xoff': local_xoff,
                'yoff': local_yoff,
                'win_xsize': local_win_xsize,
                'win_ysize': local_win_ysize,
            }
            valid_mask = numpy.ones(
                (local_win_ysize, local_win_xsize), dtype=bool)
            array_list = []
            for raster_path in raster_path_list:
                raster = gdal.OpenEx(raster_path)
                band = raster.GetRasterBand(1)
                array = band.ReadAsArray(**offset_dict)

                nodata = band.GetNoDataValue()
                band = None
                raster = None
                if nodata is not None:
                    valid_mask &= (array != nodata)

                array_list.append(array)
            if not numpy.any(valid_mask):
                continue
            mask_array[x_index, y_index] = 0.0
            for array_index, array in enumerate(array_list):
                grid_sum = numpy.sum(array[valid_mask])
                A_list[array_index].append(-grid_sum)
                raster_sum_list[array_index] += grid_sum

            # record the current grid offset for sub-multigrid
            offset_list.append(offset_dict)
    b_list = [-min_proportion*tot_val for tot_val in raster_sum_list]
    tol = min([prop_tol*val for val in raster_sum_list])
    c_vector = numpy.full(len(A_list[0]), col_stepsize*row_stepsize)
    res = scipy.optimize.linprog(
        c_vector,
        A_ub=A_list,
        b_ub=b_list,
        bounds=[0, 1],
        options={'tol': tol, 'disp': False})
    valid_mask = mask_array == 0
    mask_array[valid_mask] = res.x
    mask_array[~valid_mask] = 0

    if col_stepsize == 1 and row_stepsize == 1:
        raster = gdal.OpenEx(
            target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
        band = raster.GetRasterBand(1)
        band.WriteArray(mask_array.T, xoff=win_xoff, yoff=win_yoff)
        band = None
        raster = None
    else:
        for local_offset, local_prop in zip(
                offset_list, mask_array[valid_mask]):
            if local_prop > 1:
                local_prop = 1
            n_local_pixels = (
                local_offset['win_xsize'] * local_offset['win_ysize'])
            predicted_pixels_to_set = round(local_prop * n_local_pixels)
            if predicted_pixels_to_set == 0:
                raster = gdal.OpenEx(
                    target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
                band = raster.GetRasterBand(1)
                band.WriteArray(
                    numpy.zeros(
                        (local_offset['win_ysize'],
                         local_offset['win_xsize'])),
                    xoff=local_offset['xoff'], yoff=local_offset['yoff'])
                band = None
                raster = None
                continue
            if round(local_prop * n_local_pixels) == n_local_pixels:
                raster = gdal.OpenEx(
                    target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
                band = raster.GetRasterBand(1)
                band.WriteArray(
                    numpy.ones(
                        (local_offset['win_ysize'],
                         local_offset['win_xsize'])),
                    xoff=local_offset['xoff'], yoff=local_offset['yoff'])
                band = None
                raster = None
                continue
            multigrid_optimize(
                raster_path_list, local_prop, target_raster_path,
                local_offset['xoff'], local_offset['yoff'],
                local_offset['win_xsize'], local_offset['win_ysize'],
                prop_tol=prop_tol, grid_size=grid_size)

    return time.time() - start_time


def _scipy_optimize(raster_path_list, min_proportion):
    raster_info = pygeoprocessing.get_raster_info(raster_path_list[0])
    n_rows, n_cols = raster_info['raster_size']
    LOGGER.info('construct area list')
    c_vector = numpy.ones(n_rows*n_cols)
    A_list = []
    b_list = []

    tol = 1.0
    tot_val = 0.0
    for raster_path in raster_path_list:
        LOGGER.info(f'processing {raster_path}')
        raster = gdal.OpenEx(raster_path)
        array = raster.ReadAsArray().flatten()
        A_list.append(-array.flatten())
        array_sum = numpy.sum(array)
        tot_val += array_sum
        b_list.append(-min_proportion*array_sum)
        tol = min(tol, 1e-12*array_sum)

    LOGGER.debug('solving problem')
    res = scipy.optimize.linprog(
        c_vector,
        A_ub=A_list,
        b_ub=b_list,
        bounds=[0, 1],
        options={'tol': tol, 'disp': True})
    raster_results = [numpy.sum(A * res.x)/numpy.sum(A) for A in A_list]
    return res, tot_val, raster_results


def _sum_over_mask(base_raster_path, mask_raster_path):
    """Return the sum of mask==1 pixels over base."""
    running_sum = 0.0
    for (_, base_array), (_, mask_array) in zip(
            pygeoprocessing.iterblocks((base_raster_path, 1)),
            pygeoprocessing.iterblocks((mask_raster_path, 1))):
        running_sum += numpy.sum(base_array[mask_array >= 0.5])
    return running_sum


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Calc sum given min area')
    parser.add_argument('grid_size', type=int,)
    parser.add_argument('n_cells', type=int,)
    parser.add_argument('min_size', type=int,)
    args = parser.parse_args()
    raster_side_length = args.n_cells
    LOGGER.info('construct test data')
    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count())

    test_data_task = task_graph.add_task(
        func=_make_test_data_smooth,  # _make_test_data_random,
        args=('test_data', raster_side_length, 10),
        store_result=True,
        task_name='make smooth test data')
    raster_path_list = test_data_task.get()
    LOGGER.info('construct optimization problem')
    min_proportion = 0.5

    current_grid_size = args.grid_size
    opt_task_list = []
    with open(f'result_{raster_side_length}_{args.grid_size}.csv', 'w') as \
            csv_file:
        csv_file.write('grid size,run time,')
        csv_file.write(','.join([
            os.path.basename(os.path.splitext(path)[0])
            for path in raster_path_list]))
        while current_grid_size >= args.min_size:
            LOGGER.debug(f'processing grid size {current_grid_size}')
            csv_file.write(f'\n{current_grid_size}')
            target_raster_path = (
                f'optimal_mask_{args.n_cells}_{current_grid_size}.tif')
            if not os.path.exists(target_raster_path):
                pygeoprocessing.new_raster_from_base(
                    raster_path_list[0], target_raster_path, gdal.GDT_Float32,
                    [-1])

            raster_info = pygeoprocessing.get_raster_info(raster_path_list[0])
            n_cols, n_rows = raster_info['raster_size']
            LOGGER.debug(f'{n_rows} {n_cols} {current_grid_size}')
            optimization_task = task_graph.add_task(
                func=multigrid_optimize,
                args=(
                    raster_path_list, min_proportion, target_raster_path,
                    0, 0, n_cols, n_rows),
                kwargs={'prop_tol': 1e-12, 'grid_size': current_grid_size},
                store_result=True)
            opt_task_list.append((optimization_task, target_raster_path))

    for optimization_task, target_raster_path in opt_task_list:
        runtime = optimization_task.get()
        csv_file.write(f',{runtime}')
        for base_raster_path in raster_path_list:
            val_sum = _sum_over_mask(base_raster_path, target_raster_path)
            csv_file.write(f',{val_sum}')
            current_grid_size = current_grid_size // 2
    return

    # multi-grid
    # 1) coarsen raster, or make grids of coarseness
    # 2) solve coarse raster, project to finer one and solve those subproblems
    #   constraint is the total area selected
    #   condition is maximize the sum?d


if __name__ == '__main__':
    main()
