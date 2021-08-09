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


def _construct_optimization_problem(raster_path_list):
    """Construct optimization problem from raster path list."""
    raster_info = ecoshard.geoprocessing.get_raster_info(raster_path_list[0])
    prob = pulp.LpProblem('NCP_area_problem', pulp.LpMinimize)
    n_rows, n_cols = raster_info['raster_size']
    variable_list = []
    LOGGER.info('construct area list')
    for row in range(n_rows):
        for col in range(n_cols):
            flat_index = row*n_cols+col
            var = pulp.LpVariable(f'x{flat_index}', 0, 1)
            variable_list.append(var)
    for raster_path in raster_path_list:
        #LOGGER.info(f'processing {raster_path}')
        raster = gdal.OpenEx(raster_path)
        array = raster.ReadAsArray()

        ncp_area_value_list = []
        for row in range(n_rows):
            for col in range(n_cols):
                flat_index = row*n_cols+col
                ncp_area_value_list.append(
                    variable_list[flat_index] * array[row, col])
        prob += sum(ncp_area_value_list) >= 0.9 * numpy.sum(array), f'{os.path.basename(raster_path)}_ncp'
    prob += sum(variable_list)
    return prob

def _callback(prob):
    LOGGER.debug(f'phase: {prob.phase}, status: {prob.status} {prob.message}')

def _scipy_optimize(raster_path_list):
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
        b_list.append(-0.9*array_sum)
        tol = min(tol, 1e-6*array_sum)

    LOGGER.debug('solving problem')
    #print(c_vector)
    A_matrix = numpy.array(A_list)
    #print(A_matrix)
    #print(A_matrix.shape)
    #print(b_list)
    res = scipy.optimize.linprog(
        c_vector,
        A_ub=A_list,
        b_ub=b_list,
        bounds=[0, 1],
        method='revised simplex',
        #callback=_callback,
        options={'tol': tol, 'disp': True})
    return res, tot_val

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
    problem, tot_val = _scipy_optimize(raster_path_list)
    LOGGER.debug(problem.fun)
    LOGGER.debug(problem.fun/tot_val)
    LOGGER.debug(sum(problem.x)/(n*n))

    #for v in problem.variables():
    #    print(v.name, "=", v.varValue)

    # multi-layer solution?
    # 1) coarsen raster, or make grids of coarseness
    # 2) solve coarse raster, project to finer one and solve those subproblems
    #   constraint is the total area selected
    #   condition is maximize the sum?d


if __name__ == '__main__':
    main()
