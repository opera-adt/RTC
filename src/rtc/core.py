#!/usr/bin/env python

import os
import sys
import shutil
import logging
import numpy as np
import tempfile

from osgeo import gdal, osr, ogr

# Buffer for antimeridian crossing test (33 arcsec: ~ 1km)
ANTIMERIDIAN_CROSSING_RIGHT_SIDE_TEST_BUFFER = 33 * 0.0002777


class Logger(object):
    """
    Class to redirect stdout and stderr to the logger
    """
    def __init__(self, logger, level, prefix=''):
        """
        Class constructor
        """
        self.logger = logger
        self.level = level
        self.prefix = prefix
        self.buffer = ''

    def write(self, message):

        # Add message to the buffer until "\n" is found
        if '\n' not in message:
            self.buffer += message
            return

        message = self.buffer + message

        # check if there is any character after the last \n
        # if so, move it to the buffer
        message_list = message.split('\n')
        if not message.endswith('\n'):
            self.buffer = message_list[-1]
            message_list = message_list[:-1]
        else:
            self.buffer = ''

        # print all characters before the last \n
        for line in message_list:
            if not line:
                continue
            self.logger.log(self.level, self.prefix + line)

    def flush(self):
        if self.buffer != '':
            self.logger.log(self.level, self.buffer)
        self.buffer = ''


def save_as_cog(filename, scratch_dir='.', logger=None,
                flag_compress=True, ovr_resamp_algorithm=None,
                compression='DEFLATE', nbits=None):
    """Save (overwrite) a GeoTIFF file as a cloud-optimized GeoTIFF.

       Parameters
       ----------
       filename: str
              GeoTIFF to be saved as a cloud-optimized GeoTIFF
       scratch_dir: str (optional)
              Temporary Directory
       ovr_resamp_algorithm: str (optional)
              Resampling algorithm for overviews.
              Options: "AVERAGE", "AVERAGE_MAGPHASE", "RMS", "BILINEAR",
              "CUBIC", "CUBICSPLINE", "GAUSS", "LANCZOS", "MODE",
              "NEAREST", or "NONE". Defaults to "NEAREST", if integer, and
              "CUBICSPLINE", otherwise.
        compression: str (optional)
              Compression type.
              Optional: "NONE", "LZW", "JPEG", "DEFLATE", "ZSTD", "WEBP",
              "LERC", "LERC_DEFLATE", "LERC_ZSTD", "LZMA"

    """
    if logger is None:
        logger = logging.getLogger('rtc_s1')

    logger.info('        COG step 1: add overviews')
    gdal_ds = gdal.Open(filename, gdal.GA_Update)
    gdal_dtype = gdal_ds.GetRasterBand(1).DataType
    dtype_name = gdal.GetDataTypeName(gdal_dtype).lower()

    overviews_list = [4, 16, 64, 128]

    is_integer = 'byte' in dtype_name or 'int' in dtype_name
    if ovr_resamp_algorithm is None and is_integer:
        ovr_resamp_algorithm = 'NEAREST'
    elif ovr_resamp_algorithm is None:
        ovr_resamp_algorithm = 'CUBICSPLINE'

    logger.info('            overview resampling algorithm:'
                f' {ovr_resamp_algorithm}')
    logger.info(f'            overview list: {overviews_list}')

    gdal_ds.BuildOverviews(ovr_resamp_algorithm, overviews_list,
                           gdal.TermProgress_nocb)

    del gdal_ds  # close the dataset (Python object and pointers)
    external_overview_file = filename + '.ovr'
    if os.path.isfile(external_overview_file):
        os.remove(external_overview_file)

    logger.info('        COG step 2: save as COG')
    temp_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

    # Blocks of 512 x 512 => 256 KiB (UInt8) or 1MiB (Float32)
    tile_size = 512
    gdal_translate_options = ['BIGTIFF=IF_SAFER',
                              'MAX_Z_ERROR=0',
                              'TILED=YES',
                              f'BLOCKXSIZE={tile_size}',
                              f'BLOCKYSIZE={tile_size}',
                              'COPY_SRC_OVERVIEWS=YES']

    if compression:
        gdal_translate_options += [f'COMPRESS={compression}']

    if is_integer:
        gdal_translate_options += ['PREDICTOR=2']
    else:
        gdal_translate_options += ['PREDICTOR=3']

    if nbits is not None:
        gdal_translate_options += [f'NBITS={nbits}']

        # suppress type casting errors
        gdal.SetConfigOption('CPL_LOG', '/dev/null')

    gdal.Translate(temp_file, filename,
                   creationOptions=gdal_translate_options)

    shutil.move(temp_file, filename)

    logger.info('        COG step 3: validate')
    try:
        from rtc.extern.validate_cloud_optimized_geotiff import main as \
            validate_cog
    except ModuleNotFoundError:
        logger.info('WARNING could not import module'
                    ' validate_cloud_optimized_geotiff')
        return

    argv = ['--full-check=yes', filename]
    validate_cog_ret = validate_cog(argv)
    if validate_cog_ret == 0:
        logger.info(f'        file "{filename}" is a valid cloud optimized'
                    ' GeoTIFF')
    else:
        logger.warning(f'        file "{filename}" is NOT a valid cloud'
                       f' optimized GeoTIFF!')


def _get_ogr_polygon(min_x, max_y, max_x, min_y, file_srs):
    file_ring = ogr.Geometry(ogr.wkbLinearRing)
    file_ring.AddPoint(min_x, max_y)
    file_ring.AddPoint(max_x, max_y)
    file_ring.AddPoint(max_x, min_y)
    file_ring.AddPoint(min_x, min_y)
    file_ring.AddPoint(min_x, max_y)
    file_polygon = ogr.Geometry(ogr.wkbPolygon)
    file_polygon.AddGeometry(file_ring)
    file_polygon.AssignSpatialReference(file_srs)
    assert file_polygon.IsValid()
    return file_polygon


def get_tile_srs_bbox(tile_min_y_projected, tile_max_y_projected,
                      tile_min_x_projected, tile_max_x_projected,
                      tile_srs, polygon_srs, logger=None):
    """Get tile bounding box for a given spatial reference system (SRS)

       Parameters
       ----------
       tile_min_y_projected: float
              Tile minimum Y-coordinate
       tile_max_y_projected: float
              Tile maximum Y-coordinate
       tile_min_x_projected: float
              Tile minimum X-coordinate
       tile_max_x_projected: float
              Tile maximum X-coordinate
       tile_srs: osr.SpatialReference
              Tile original spatial reference system (SRS). If the polygon
              SRS is geographic, its Axis Mapping Strategy will
              be updated to osr.OAMS_TRADITIONAL_GIS_ORDER
       polygon_srs: osr.SpatialReference
              Polygon spatial reference system (SRS). If the polygon
              SRS is geographic, its Axis Mapping Strategy will
              be updated to osr.OAMS_TRADITIONAL_GIS_ORDER
       logger : logging.Logger, optional
              Logger object
       Returns
       -------
       tile_polygon: ogr.Geometry
              Rectangle representing polygon SRS bounding box
       tile_min_y: float
              Tile minimum Y-coordinate (polygon SRS)
       tile_max_y: float
              Tile maximum Y-coordinate (polygon SRS)
       tile_min_x: float
              Tile minimum X-coordinate (polygon SRS)
       tile_max_x: float
              Tile maximum X-coordinate (polygon SRS)
    """
    if logger is None:
        logger = logging.getLogger('rtc_s1')

    # forces returned values from TransformPoint() to be (x, y, z)
    # rather than (y, x, z) for geographic SRS
    if tile_srs.IsGeographic():
        try:
            tile_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            logger.warning('WARNING Could not set the ancillary input SRS axis'
                           ' mapping strategy (SetAxisMappingStrategy())'
                           ' to osr.OAMS_TRADITIONAL_GIS_ORDER')
    if polygon_srs.IsGeographic():
        try:
            polygon_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            logger.warning('WARNING Could not set the ancillary input SRS axis'
                           ' mapping strategy (SetAxisMappingStrategy())'
                           ' to osr.OAMS_TRADITIONAL_GIS_ORDER')
    transformation = osr.CoordinateTransformation(tile_srs, polygon_srs)

    elevation = 0
    tile_x_array = np.zeros(4)
    tile_y_array = np.zeros(4)
    tile_x_array[0], tile_y_array[0], z = transformation.TransformPoint(
        tile_min_x_projected, tile_max_y_projected, elevation)
    tile_x_array[1], tile_y_array[1], z = transformation.TransformPoint(
        tile_max_x_projected, tile_max_y_projected, elevation)
    tile_x_array[2], tile_y_array[2], z = transformation.TransformPoint(
        tile_max_x_projected, tile_min_y_projected, elevation)
    tile_x_array[3], tile_y_array[3], z = transformation.TransformPoint(
        tile_min_x_projected, tile_min_y_projected, elevation)
    tile_min_y = np.min(tile_y_array)
    tile_max_y = np.max(tile_y_array)
    tile_min_x = np.min(tile_x_array)
    tile_max_x = np.max(tile_x_array)

    # handles antimeridian: tile_max_x around +180 and tile_min_x around -180
    if polygon_srs.IsGeographic() and tile_max_x - tile_min_x > 180:
        # unwrap negative longitude values
        # move longitude range from [-180, 180] to [0, 360]
        new_tile_x_array = [x + (x < 0) * 360 for x in tile_x_array]
        tile_min_x = np.min(new_tile_x_array)
        tile_max_x = np.max(new_tile_x_array)

    tile_ring = ogr.Geometry(ogr.wkbLinearRing)
    tile_ring.AddPoint(tile_min_x, tile_max_y)
    tile_ring.AddPoint(tile_max_x, tile_max_y)
    tile_ring.AddPoint(tile_max_x, tile_min_y)
    tile_ring.AddPoint(tile_min_x, tile_min_y)
    tile_ring.AddPoint(tile_min_x, tile_max_y)
    tile_polygon = ogr.Geometry(ogr.wkbPolygon)
    tile_polygon.AddGeometry(tile_ring)
    tile_polygon.AssignSpatialReference(polygon_srs)
    return tile_polygon, tile_min_y, tile_max_y, tile_min_x, tile_max_x


def _antimeridian_crossing_requires_special_handling(
        file_srs, ancillary_min_x, ancillary_max_x, tile_min_x, tile_max_x):
    '''
    Check if ancillary input requires special handling due to
    the antimeridian crossing

    Parameters
    ----------
    file_srs: osr.SpatialReference
        Ancillary file spatial reference system (SRS)
    ancillary_min_x: float
        Ancillary file min longitude value in degrees
    ancillary_max_x: float
        Ancillary file max longitude value in degrees
    tile_min_x: float
        Tile min longitude value in degrees
    tile_max_x: float
        Tile max longitude value in degrees

    Returns
    -------
    flag_requires_special_handling : bool
        Flag that indicate if the ancillary input requires special handling
    '''

    # The coordinate wrapping around the ancillary file
    # discontinuities such as the antimeridian only happens
    # if the DEM is provided in geographic coordinates
    if not file_srs.IsGeographic():
        flag_requires_special_handling = False
        return flag_requires_special_handling

    # Check whether the ancillary file covers the entire longitude range
    # Mark flag as True if the longitude coordinates span completes
    # or is close to complete the circle (360 degrees).
    # We use 359 instead of 360 degrees to have some buffer
    # for incomplete ancillary files
    flag_ancillary_covers_entire_longitude_range = \
        (ancillary_max_x - ancillary_min_x) > 359

    if not flag_ancillary_covers_entire_longitude_range:
        flag_requires_special_handling = False
        return flag_requires_special_handling

    # Flag to indicate if the tile crosses the ancillary file
    # discontinuity (ancillary_max_x).
    # Notice that`ancillary_max_x` refers to the eastern edge
    # of the ancillary file, which, for the antimeridian case,
    # may not lay exactly on +/- 180. For example, it may be
    # 179.9998611111111
    flag_tile_crosses_discontinuity = \
        tile_min_x < ancillary_max_x < tile_max_x

    flag_requires_special_handling = \
        flag_tile_crosses_discontinuity

    return flag_requires_special_handling


def check_ancillary_inputs(check_ancillary_inputs_coverage,
                           dem_file, geogrid,
                           metadata_dict, logger=None):
    """Check for the existence and coverage of provided ancillary inputs
       (e.g., DEM) wrt. to a reference geogrid. The function
       also updates the product's dictionary metadata indicating the coverage
       of each ancillary input wrt. to the reference geogrid

       Parameters
       ----------
       check_ancillary_inputs_coverage: bool
               Flag that enable/disable checks for all ancillary inputs
               excluding the shoreline shapefile
       check_shoreline_shapefile: bool
               Flag that checks for the shoreline shapefile
       dem_file: str
               DEM filename
       geogrid: isce3.product.GeoGridParameters
               Product's ISCE3 geogrid object
       metadata_dict: collections.OrderedDict
               Metadata dictionary
       logger : logging.Logger, optional
              Logger object
    """
    if logger is None:
        logger = logging.getLogger('rtc_s1')
    logger.info("Check ancillary inputs' coverage:")

    # file description (to be printed to the user if an error happens)
    dem_file_description = 'DEM file'

    if not check_ancillary_inputs_coverage:

        # print messages to the user
        logger.info(f'    {dem_file_description} coverage:'
                    ' (not tested)')

        # update RTC-S1 product metadata
        metadata_dict['DEM_COVERAGE'] = 'NOT_TESTED'

        return

    rasters_to_check_dict = {'DEM': (dem_file_description, dem_file)}

    geogrid_x0_projected = geogrid.start_x
    geogrid_y0_projected = geogrid.start_y
    # define end (final) geogrid X/Y edge coordinates
    geogrid_xf_projected = (geogrid.start_x +
                            geogrid.spacing_x * geogrid.width)
    geogrid_yf_projected = (geogrid.start_y +
                            geogrid.spacing_y * geogrid.length)

    geogrid_srs = osr.SpatialReference()
    geogrid_srs.ImportFromEPSG(geogrid.epsg)

    for ancillary_file_type, \
        (ancillary_file_description, ancillary_file_name) in \
            rasters_to_check_dict.items():

        # check if file was provided
        if not ancillary_file_name:
            error_msg = f'ERROR {ancillary_file_description} not provided'
            logger.error(error_msg)
            raise ValueError(error_msg)

        # check if file exists
        if not os.path.isfile(ancillary_file_name):
            error_msg = f'ERROR {ancillary_file_description} not found:'
            error_msg += f' {ancillary_file_name}'
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # test if the reference geogrid is fully covered by the ancillary input
        # by checking all ancillary input vertices are located
        # outside of the reference geogrid.
        ancillary_gdal_ds = gdal.Open(ancillary_file_name, gdal.GA_ReadOnly)

        ancillary_geotransform = ancillary_gdal_ds.GetGeoTransform()
        ancillary_projection = ancillary_gdal_ds.GetProjection()
        ancillary_x0, ancillary_dx, _, ancillary_y0, _, ancillary_dy = \
            ancillary_geotransform
        ancillary_width = ancillary_gdal_ds.GetRasterBand(1).XSize
        ancillary_length = ancillary_gdal_ds.GetRasterBand(1).YSize

        del ancillary_gdal_ds

        # define end (final) ancillary input X/Y edge coordinates
        ancillary_xf = ancillary_x0 + ancillary_width * ancillary_dx
        ancillary_yf = ancillary_y0 + ancillary_length * ancillary_dy

        ancillary_srs = osr.SpatialReference()
        ancillary_srs.ImportFromProj4(ancillary_projection)

        ret = get_tile_srs_bbox(geogrid_yf_projected,
                                geogrid_y0_projected,
                                geogrid_x0_projected,
                                geogrid_xf_projected,
                                geogrid_srs,
                                ancillary_srs)
        geogrid_polygon, geogrid_yf, geogrid_y0, geogrid_x0, geogrid_xf = ret

        # Create input ancillary polygon
        ancillary_polygon = _get_ogr_polygon(ancillary_x0,
                                             ancillary_y0,
                                             ancillary_xf,
                                             ancillary_yf,
                                             ancillary_srs)

        coverage_logger_str = ancillary_file_description + ' coverage'
        coverage_metadata_str = ancillary_file_type + '_COVERAGE'

        if geogrid_polygon.Within(ancillary_polygon):
            # print messages to the user
            logger.info(f'    {coverage_logger_str}: Full')

            # update RTC-S1 product metadata
            metadata_dict[coverage_metadata_str] = 'FULL'
            continue

        # If needed, test for antimeridian ("dateline") crossing
        if _antimeridian_crossing_requires_special_handling(
                ancillary_srs, ancillary_x0, ancillary_xf,
                geogrid_x0, geogrid_xf):

            logger.info(f'The input RTC-S1 product crosses the antimeridian'
                        ' (dateline). Verifying the'
                        f' {ancillary_file_description}:'
                        f' {ancillary_file_name}')

            # Left side of the antimeridian crossing: -180 -> +180
            ancillary_polygon_1 = _get_ogr_polygon(-180, 90, ancillary_xf, -90,
                                                   ancillary_srs)
            intersection_1 = geogrid_polygon.Intersection(ancillary_polygon_1)
            flag_1_ok = intersection_1.Within(ancillary_polygon)
            check_1_str = 'ok' if flag_1_ok else 'fail'
            logger.info(f'    left side (-180 -> +180): {check_1_str}')

            # Right side of the antimeridian crossing: +180 -> +360
            #
            # Get the intersection between the geogrid and the right side
            # of the antimeridian (with a litter buffer represented by 
            # ANTIMERIDIAN_CROSSING_RIGHT_SIDE_TEST_BUFFER)
            antimeridian_right_side_polygon = _get_ogr_polygon(
                ancillary_xf + ANTIMERIDIAN_CROSSING_RIGHT_SIDE_TEST_BUFFER,
                90, ancillary_xf + 360, -90, ancillary_srs)

            intersection_2 = geogrid_polygon.Intersection(
                antimeridian_right_side_polygon)

            # Create a polygon representing the ancillary dataset
            # at the right side of the antimeridian
            ancillary_polygon_2 = _get_ogr_polygon(
                ancillary_x0 + 360, ancillary_y0,
                ancillary_xf + 360, ancillary_yf,
                ancillary_srs)

            # Check if the geogrid-intersected area (if valid) is within
            # the ancillary polygon
            flag_2_ok = (intersection_2.IsEmpty() or
                         intersection_2.Within(ancillary_polygon_2))
            check_2_str = 'ok' if flag_2_ok else 'fail'
            logger.info(f'    right side (+180 -> +360): {check_2_str}')

            if flag_1_ok and flag_2_ok:
                # print messages to the user
                logger.info(f'    {coverage_logger_str}:'
                            ' Full (with antimeridian crossing)')

                # update RTC-S1 product metadata
                metadata_dict[coverage_metadata_str] = \
                    'FULL_WITH_ANTIMERIDIAN_CROSSING'
                continue

        # prepare message to the user
        msg = f'ERROR the {ancillary_file_description} with extents'
        msg += f' S/N: [{ancillary_yf},{ancillary_y0}]'
        msg += f' W/E: [{ancillary_x0},{ancillary_xf}],'
        msg += ' does not fully cover product geogrid with'
        msg += f' extents S/N: [{geogrid_yf},{geogrid_y0}]'
        msg += f' W/E: [{geogrid_x0},{geogrid_xf}]'

        logger.error(msg)
        raise ValueError(msg)


def create_logger(log_file, full_log_formatting=None):
    """Create logger object for a log file

       Parameters
       ----------
       log_file: str
              Log file
       full_log_formatting : bool
              Flag to enable full formatting of logged messages

       Returns
       -------
       logger : logging.Logger
              Logger object
    """
    if log_file:
        logfile_directory = os.path.dirname(log_file)
    else:
        logfile_directory = None

    if logfile_directory:
        os.makedirs(logfile_directory, exist_ok=True)
    # create logger

    logger = logging.getLogger('rtc_s1')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    # configure full log format, if enabled
    if full_log_formatting:
        msgfmt = ('%(asctime)s.%(msecs)03d, %(levelname)s, RTC-S1, '
                  '%(module)s, 999999, %(pathname)s:%(lineno)d, "%(message)s"')

        formatter = logging.Formatter(msgfmt, "%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter('%(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    if log_file:
        file_handler = logging.FileHandler(log_file)

        file_handler.setFormatter(formatter)

        # add file handler to logger
        logger.addHandler(file_handler)

    sys.stdout = Logger(logger, logging.INFO)
    sys.stderr = Logger(logger, logging.ERROR, prefix='[StdErr] ')

    return logger


def build_empty_vrt(filename, length, width, fill_value, dtype='Float32',
                    geotransform=None):
    """Build an empty VRT file, i.e, not pointing to any rasters,
       with given input dimensions (length and width), data type, and
       fill value.

       Parameters
       ----------
       filename: str
              VRT file name
       length: int
              VRT data length
       width: int
              VRT data width
       fill_value: scalar
              VRT data fill value
       dtype: str
              VRT data type
       geotransform: list(scalar), optional
              VRT data geotransform

       Returns
       -------
       logger : logging.Logger
              Logger object
    """
    vrt_contents = f'<VRTDataset rasterXSize="{width}"'
    vrt_contents += f' rasterYSize="{length}"> \n'
    if geotransform is not None:
        assert len(geotransform) == 6
        geotransform_str = ', '.join([str(x) for x in geotransform])
        vrt_contents += f'  <GeoTransform> {geotransform_str}'
        vrt_contents += ' </GeoTransform> \n'
    vrt_contents += (
        f'  <VRTRasterBand dataType="{dtype}" band="1"> \n'
        f'    <NoDataValue>{fill_value}</NoDataValue> \n'
        f'    <HideNoDataValue>{fill_value}</HideNoDataValue> \n'
        f'  </VRTRasterBand> \n'
        f'</VRTDataset> \n')

    with open(filename, 'w') as out:
        out.write(vrt_contents)

    if os.path.isfile(filename):
        print('file saved:', filename)
