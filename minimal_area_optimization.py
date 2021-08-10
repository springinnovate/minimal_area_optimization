"""Calculate totals given minimal area request."""
import argparse
import logging
import os

import ecoshard.geoprocessing
import scipy.ndimage.morphology
import numpy
from osgeo import gdal
from osgeo import osr
import pulp


gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

LOGGER = logging.getLogger(__name__)


def _make_test_data(dir_path, n, m):
    """Make random raster test data."""
    os.makedirs(dir_path, exist_ok=True)
    raster_path_list = []
    for raster_path, (pi, pj) in [
            (f'{index}.tif', (int(n*index/m), 0)) for index in range(m)]:
        base_array = numpy.ones((n, n))
        base_array[pi, pj] = 0
        dist_array = scipy.ndimage.morphology.distance_transform_edt(
            base_array)
        ecoshard.geoprocessing.numpy_array_to_raster(
            dist_array, -1, (1, -1), (0, 0),
            osr.SRS_WKT_WGS84_LAT_LONG, raster_path)
        raster_path_list.append(raster_path)
    return raster_path_list


def _callback(prob):
    LOGGER.debug(f'phase: {prob.phase}, status: {prob.status} {prob.message}')


def _sum_raster(raster_path):
    """Return the non-nodata sum of the raster."""
    raster_info = ecoshard.geoprocessing.get_raster_info(raster_path)
    nodata = raster_info['nodata'][0]
    running_sum = 0.0
    for _, array in ecoshard.geoprocessing.iterblocks((raster_path, 1)):
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
        prop_tol (float): propotion of raster sum to use as tolerance.
        grid_size (int): the size to subdivide the optimization problem on.

    Return:
        {
            'objective_sum_list': [sum of selected objectives],
            'proportion_list': [proportion of raster selected],
            'area_list': [area of raster selected]
        }
    """
    raster_info = ecoshard.geoprocessing.get_raster_info(raster_path_list[0])
    sum_list = [_sum_raster(path) for path in raster_path_list]

    n_cols, n_rows = raster_info['raster_size']

    n_col_steps = int(numpy.ceil(win_xsize / grid_size))
    n_row_steps = int(numpy.ceil(win_ysize / grid_size))
    while n_col_steps > 1 and n_row_steps > 1:
        LOGGER.debug(f'{n_col_steps}, {n_row_steps}')
        x_stepsize = win_xsize // n_col_steps
        y_stepsize = win_ysize // n_row_steps

        c_vector = numpy.ones(n_col_steps*n_row_steps)
        A_list = []
        b_list = []

        for raster_path in raster_path_list:
            LOGGER.info(f'processing {raster_path}')
            raster = gdal.OpenEx(raster_path)
            band = raster.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            a_constraint_list = []
            tot_val = 0.0
            var_index = 0
            offset_list = []
            for x_index in range(n_col_steps):
                local_xoff = x_index * x_stepsize
                if (x_index+1)*x_stepsize < n_cols:
                    local_win_xsize = x_stepsize
                else:
                    local_win_xsize = n_cols - (x_index+1)*n_col_steps

                for y_index in range(n_row_steps):
                    local_yoff = y_index * y_stepsize
                    if (y_index+1)*y_stepsize < n_rows:
                        local_win_ysize = y_stepsize
                    else:
                        local_win_ysize = n_rows - (y_index+1)*y_stepsize

                    tol = 1.0

                    # load the subgrid
                    offset_dict = {
                        'xoff': local_xoff,
                        'yoff': local_yoff,
                        'win_xsize': local_win_xsize,
                        'win_ysize': local_win_ysize,
                    }
                    offset_list.append(offset_dict)
                    array = band.ReadAsArray(**offset_dict)
                    if nodata is not None:
                        valid_mask = array != nodata
                    else:
                        valid_mask = numpy.ones(array.shape, dtype=bool)
                    # aggregate it up to a single value
                    array_sum = numpy.sum(array[valid_mask])
                    a_constraint_list.append(-array_sum)
                    tot_val += array_sum
            A_list.append(a_constraint_list)
            b_list.append(-min_proportion*tot_val)

        c_vector = numpy.ones(len(a_constraint_list))
        res = scipy.optimize.linprog(
            c_vector,
            A_ub=A_list,
            b_ub=b_list,
            bounds=[0, 1],
            options={'tol': tol, 'disp': True})
        for local_offset, local_prop in zip(offset_list, res.x):
            LOGGER.debug(res.x)
            multigrid_optimize(
                raster_path_list, min_proportion, target_raster_path,
                local_offset['xoff'], local_offset['yoff'],
                local_offset['win_xsize'], local_offset['win_ysize'],
                prop_tol=1e-12, grid_size=64,)
        # iterate over each solution of x here and solve that subproblem

                #LOGGER.debug(
                #    f'sampling offset for x{var_index}\n'
                #    f'\t{xoff}, {yoff}, {win_xsize}, {win_ysize} ({xoff+win_xsize}, {yoff+win_ysize})')

                #var_index += 1

        grid_size *= 2
        n_col_steps = int(numpy.ceil(n_cols / grid_size))
        n_row_steps = int(numpy.ceil(n_rows / grid_size))


def _scipy_optimize(raster_path_list, min_proportion):
    raster_info = ecoshard.geoprocessing.get_raster_info(raster_path_list[0])
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

def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Calc sum given min area')
    parser.add_argument(
        'raster_list_path',  help='Path to .txt file with list of rasters.')
    args = parser.parse_args()
    n = 64
    LOGGER.info('construct test data')
    raster_path_list = _make_test_data('test_data', n, 10)
    LOGGER.info('construct optimization problem')
    #problem = _construct_optimization_problem(raster_path_list)
    #LOGGER.info('solve optimization problem')
    #problem.solve()
    min_proportion = 0.9

    #problem, tot_val, raster_results = _scipy_optimize(raster_path_list, min_proportion)

    target_raster_path = 'result.tif'
    multigrid_optimize(
        raster_path_list, min_proportion, target_raster_path,
        prop_tol=1e-12, grid_size=64)
    return


    LOGGER.debug(problem.fun)
    LOGGER.debug(problem.fun/(n*n))
    area_proportion = sum(problem.x)/(n*n)
    result_proportion = problem.fun/tot_val
    LOGGER.info(f'result\n\tarea_proportion: {area_proportion:.3f}\n\tsolution proportion for each raster: {raster_results}')
    LOGGER.info(problem.x)
    ecoshard.geoprocessing.numpy_array_to_raster(
        (problem.x).reshape((n, n)), -1, (1, -1), (0, 0),
        osr.SRS_WKT_WGS84_LAT_LONG, 'result.tif')
    #for v in problem.variables():
    #    print(v.name, "=", v.varValue)

    # multi-layer solution?
    # 1) coarsen raster, or make grids of coarseness
    # 2) solve coarse raster, project to finer one and solve those subproblems
    #   constraint is the total area selected
    #   condition is maximize the sum?d


if __name__ == '__main__':
    main()
