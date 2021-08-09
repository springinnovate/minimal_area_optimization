"""Calculate totals given minimal area request."""
import argparse
import logging
import os

import ecoshard.geoprocessing
import scipy.ndimage.morphology
import numpy
from osgeo import gdal
from osgeo import osr

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

LOGGER = logging.getLogger(__name__)


def _make_test_data(dir_path):
    """Make random raster test data."""
    os.makedirs(dir_path, exist_ok=True)
    for raster_path, (pi, pj) in [
            ('a.tif', (0, 0)), ('b.tif', (9, 0)), ('c.tif', (5, 0))]:
        base_array = numpy.ones((10, 10))
        base_array[pi, pj] = 0
        dist_array = scipy.ndimage.morphology.distance_transform_edt(
            base_array)
        ecoshard.geoprocessing.numpy_array_to_raster(
            dist_array, -1, (1, -1), (0, 0),
            osr.SRS_WKT_WGS84_LAT_LONG, raster_path)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Calc sum given min area')
    parser.add_argument(
        'raster_list_path',  help='Path to .txt file with list of rasters.')
    args = parser.parse_args()
    _make_test_data('test_data')


if __name__ == '__main__':
    main()
