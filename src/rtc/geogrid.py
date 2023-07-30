'''
Collection of function for determining and setting the geogrid
'''

import numpy as np
import logging
from osgeo import osr

from nisar.workflows.geogrid import _grid_size
import isce3

from rtc.helpers import burst_bboxes_from_db
from rtc.core import get_tile_srs_bbox

logger = logging.getLogger('rtc_s1')

# 1 arcsec (not using 1.0/3600.0 to avoid repeating decimals)
SECONDS_IN_DEG = 0.0002777


def assign_check_geogrid(geogrid, xmin=None, ymax=None,
                         xmax=None, ymin=None):
    '''
    Initialize geogrid with user defined parameters.
    Check the validity of user-defined parameters

    Parameters
    ----------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object defining the geogrid
    xmin: float
        Geogrid top-left X coordinate
    ymax: float
        Geogrid top-left Y coordinate
    xmax: float
        Geogrid bottom-right X coordinate
    ymin: float
        Geogrid bottom-right Y coordinate

    Returns
    -------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 geogrid initialized with user-defined inputs
    '''

    # Check assigned input coordinates and initialize geogrid accordingly
    if xmin is not None:
        current_end_x = geogrid.start_x + geogrid.spacing_x * geogrid.width
        geogrid.start_x = xmin
        geogrid.width = int(np.ceil((current_end_x - xmin) /
                                     geogrid.spacing_x))
    # Restore geogrid end point if provided by the user
    if xmax is not None:
        geogrid.width = int(np.ceil((xmax - geogrid.start_x) /
                                     geogrid.spacing_x))
    if ymax is not None:
        current_end_y = geogrid.start_y + geogrid.spacing_y * geogrid.length
        geogrid.start_y = ymax
        geogrid.length = int(np.ceil((current_end_y - ymax) /
                                      geogrid.spacing_y))
    if ymin is not None:
        geogrid.length = int(np.ceil((ymin - geogrid.start_y) /
                                      geogrid.spacing_y))

    return geogrid


def intersect_geogrid(geogrid, xmin=None, ymax=None,
                      xmax=None, ymin=None):
    '''
    Return intersected geogrid with user defined parameters.

    Parameters
    ----------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object defining the geogrid
    xmin: float
        Geogrid top-left X coordinate
    ymax: float
        Geogrid top-left Y coordinate
    xmax: float
        Geogrid bottom-right X coordinate
    ymin: float
        Geogrid bottom-right Y coordinate

    Returns
    -------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 geogrid
    '''

    if xmin is not None and xmin > geogrid.start_x:
        current_end_x = geogrid.start_x + geogrid.spacing_x * geogrid.width
        geogrid.start_x = xmin
        geogrid.width = int(np.ceil((current_end_x - xmin) /
                                     geogrid.spacing_x))

    if (xmax is not None and
            (xmax < geogrid.start_x + geogrid.width * geogrid.spacing_x)):
        geogrid.width = int(np.ceil((xmax - geogrid.start_x) /
                                     geogrid.spacing_x))
    if ymax is not None and ymax < geogrid.start_y:
        current_end_y = geogrid.start_y + geogrid.spacing_y * geogrid.length
        geogrid.start_y = ymax
        geogrid.length = int(np.ceil((current_end_y - ymax) /
                                      geogrid.spacing_y))

    if (ymin is not None and 
            (ymin > geogrid.start_y + geogrid.length * geogrid.spacing_y)):
        geogrid.length = int(np.ceil((ymin - geogrid.start_y) /
                                      geogrid.spacing_y))

    return geogrid


def check_geogrid_endpoints(geogrid, xmax=None, ymin=None):
    '''
    Check validity of geogrid end points

    Parameters
    -----------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object defining the geogrid
    xmax: float
        Geogrid bottom right X coordinate
    ymin: float
        Geogrid bottom right Y coordinates

    Returns
    -------
    xmax: float
        Verified geogrid bottom-right X coordinate
    ymin: float
        Verified geogrid bottom-right Y coordinate
    '''
    end_pt = lambda start, sz, spacing: start + spacing * sz

    if xmax is None:
        xmax = end_pt(geogrid.start_x, geogrid.spacing_x, geogrid.width)
    if ymin is None:
        ymin = end_pt(geogrid.start_y, geogrid.spacing_y, geogrid.length)

    return xmax, ymin


def check_snap_values(x_snap, y_snap, x_spacing, y_spacing):
    '''
    Check validity of snap values

    Parameters
    ----------
    x_snap: float
        Snap value along X-direction
    y_snap: float
        Snap value along Y-direction
    x_spacing: float
        Spacing of the geogrid along X-direction
    y_spacing: float
        Spacing of the geogrid along Y-direction
    '''

    # Check that snap values in X/Y-directions are positive
    if x_snap is not None and x_snap <= 0:
        err_str = f'Snap value in X direction must be > 0 (x_snap: {x_snap})'
        logger.error(err_str)
        raise ValueError(err_str)
    if y_snap is not None and y_snap <= 0:
        err_str = f'Snap value in Y direction must be > 0 (y_snap: {y_snap})'
        logger.error(err_str)
        raise ValueError(err_str)

    # Check that snap values in X/Y are integer multiples of the geogrid
    # spacings in X/Y directions
    if x_snap is not None and x_snap % x_spacing != 0.0:
        err_str = 'x_snap must be exact multiple of spacing in X direction (x_snap % x_spacing !=0)'
        logger.error(err_str)
        raise ValueError(err_str)
    if y_snap is not None and y_snap % y_spacing != 0.0:
        err_str = 'y_snap must be exact multiple of spacing in Y direction (y_snap % y_spacing !=0)'
        logger.error(err_str)
        raise ValueError(err_str)


def snap_coord(val, snap, round_func):
    '''
    Returns the snapped values of the input value

    Parameters
    -----------
    val : float
        Input value to snap
    snap : float
        Snapping step
    round_func : function pointer
        A function used to round `val` i.e. round, ceil, floor

    Return:
    --------
    snapped_value : float
        snapped value of `var` by `snap`

    '''
    snapped_value = round_func(float(val) / snap) * snap
    return snapped_value


def snap_geogrid(geogrid, x_snap, y_snap):
    '''
    Snap geogrid based on user-defined snapping values

    Parameters
    ----------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object definining the geogrid
    x_snap: float
        Snap value along X-direction
    y_snap: float
        Snap value along Y-direction

    Returns
    -------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object containing the snapped geogrid
    '''
    xmax = geogrid.start_x + geogrid.width * geogrid.spacing_x
    ymin = geogrid.start_y + geogrid.length * geogrid.spacing_y

    if x_snap is not None:
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        end_x = snap_coord(xmax, x_snap, np.ceil)
        geogrid.width = _grid_size(end_x, geogrid.start_x, geogrid.spacing_x)
    if y_snap is not None:
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)
        end_y = snap_coord(ymin, y_snap, np.floor)
        geogrid.length = _grid_size(end_y, geogrid.start_y,
                                     geogrid.spacing_y)
    return geogrid


def get_point_epsg(lat, lon):
    '''
    Get EPSG code based on latitude and longitude
    coordinates of a point

    Parameters
    ----------
    lat: float
        Latitude coordinate of the point
    lon: float
        Longitude coordinate of the point

    Returns
    -------
    epsg: int
        UTM zone, Polar stereographic (North / South)
    '''

    # "Warp" longitude value into [-180.0, 180.0]
    if (lon >= 180.0) or (lon <= -180.0):
        lon = (lon + 180.0) % 360.0 - 180.0

    if lat >= 75.0:
        return 3413
    elif lat <= -75.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    else:
        err_str = "'Could not determine EPSG for {0}, {1}'.format(lon, lat))"
        logger.error(err_str)
        raise ValueError(err_str)


def _check_pixel_spacing(y_spacing_positive, x_spacing, epsg, product_str):
    '''
    Verify pixel spacing

    Parameters
    ----------
    y_spacing_positive: float
        Y/latitude pixel spacing (absolute value)
    x_spacing: float
        X/longitude pixel spacing
    epsg: int
        EPSG code
    product_str: str
        Product type string (e.g., "Mosaic" or "Bursts")

    Returns
    -------
    y_spacing: float
        Y/latitude pixel spacing
    x_spacing: float
        X/longitude pixel spacing
    '''
    # Check spacing in X direction
    if x_spacing is not None and x_spacing <= 0:
        err_str = f'{product_str} pixel spacing in X/longitude direction must'
        err_str += f' be positive (x_spacing: {x_spacing})'
        logger.error(err_str)
        raise ValueError(err_str)
    elif x_spacing is None and epsg and epsg == 4326:
        x_spacing = SECONDS_IN_DEG
    elif x_spacing is None and epsg:
        x_spacing = 30

    # Check spacing in Y direction
    if y_spacing_positive is not None and y_spacing_positive <= 0:
        err_str = f'{product_str} pixel spacing in Y/latitude direction must'
        err_str += f' be positive (y_spacing: {y_spacing_positive})'
        logger.error(err_str)
        raise ValueError(err_str)
    elif y_spacing_positive is None and epsg and epsg == 4326:
        y_spacing = -SECONDS_IN_DEG
    elif y_spacing_positive is None and epsg:
        y_spacing = -30
    elif y_spacing_positive is not None:
        y_spacing = -y_spacing_positive

    return y_spacing, x_spacing


def generate_geogrids_from_db(bursts, geo_dict, mosaic_dict, burst_db_file):
    '''
    Compute frame and bursts geogrids based on burst database

    Parameters
    ----------
    bursts: list[s1reader.s1_burst_slc.Sentinel1BurstSlc]
        List of S-1 burst SLC objects
    geo_dict: dict
        Dictionary containing runconfig processing.geocoding
        parameters
    mosaic_dict: dict
        Dictionary containing runconfig processing.mosaicking
        parameters
    burst_db_file : str
        Location of burst database sqlite file

    Returns
    -------
    geogrid_mosaic_snapped: isce3.product.GeoGridParameters
        Mosaic geogrid
    geogrids_dict: dict
        Dict containing bursts' geogrids indexed by burst_id
    '''

    # Unpack values from geocoding dictionary
    mosaic_geogrid_dict = mosaic_dict['mosaic_geogrid']
    epsg_mosaic = mosaic_geogrid_dict['output_epsg']
    xmin_mosaic = mosaic_geogrid_dict['top_left']['x']
    ymax_mosaic = mosaic_geogrid_dict['top_left']['y']
    x_spacing_mosaic = mosaic_geogrid_dict['x_posting']
    y_spacing_positive_mosaic = mosaic_geogrid_dict['y_posting']
    xmax_mosaic = mosaic_geogrid_dict['bottom_right']['x']
    ymin_mosaic = mosaic_geogrid_dict['bottom_right']['y']
    x_snap_mosaic = mosaic_geogrid_dict['x_snap']
    y_snap_mosaic = mosaic_geogrid_dict['y_snap']

    bursts_geogrid_dict = geo_dict['bursts_geogrid']
    epsg_bursts = bursts_geogrid_dict['output_epsg']
    xmin_bursts = bursts_geogrid_dict['top_left']['x']
    ymax_bursts = bursts_geogrid_dict['top_left']['y']
    x_spacing_bursts = bursts_geogrid_dict['x_posting']
    y_spacing_positive_bursts = bursts_geogrid_dict['y_posting']
    xmax_bursts = bursts_geogrid_dict['bottom_right']['x']
    ymin_bursts = bursts_geogrid_dict['bottom_right']['y']
    x_snap_bursts = bursts_geogrid_dict['x_snap']
    y_snap_bursts = bursts_geogrid_dict['y_snap']

    product_burst_str = 'Bursts'
    y_spacing_bursts, x_spacing_bursts = _check_pixel_spacing(
        y_spacing_positive_bursts, x_spacing_bursts,
        epsg_bursts, product_burst_str)

    xmin_all_bursts = np.inf
    ymax_all_bursts = -np.inf
    xmax_all_bursts = -np.inf
    ymin_all_bursts = np.inf

    geogrids_dict = {}

    # get all burst IDs and their EPSGs + bounding boxes
    burst_ids = [b for b in bursts]
    epsg_bbox_dict = burst_bboxes_from_db(burst_ids, burst_db_file)

    # compute majority of EPSG codes within `epsg_bbox_dict`
    epsg_count_dict = {}
    for epsg_burst, _ in epsg_bbox_dict.values():
        if epsg_burst not in epsg_count_dict:
            epsg_count_dict[epsg_burst] = 0
        else:
            epsg_count_dict[epsg_burst] += 1
    epsg_bursts_majority = max(epsg_count_dict, key=epsg_count_dict.get)
    srs_bursts_majority = osr.SpatialReference()
    srs_bursts_majority.ImportFromEPSG(epsg_bursts_majority)

    if epsg_mosaic is None:
        epsg_mosaic = epsg_bursts_majority

    product_mosaic_str = 'Mosaic'
    y_spacing_mosaic, x_spacing_mosaic = _check_pixel_spacing(
        y_spacing_positive_mosaic, x_spacing_mosaic,
        epsg_mosaic, product_mosaic_str)


    for burst_id, burst_pol in bursts.items():
        pol_list = list(burst_pol.keys())
        burst = burst_pol[pol_list[0]]
        if burst_id in geogrids_dict.keys():
            continue

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        # extract EPSG and bbox for current burst from dict
        # bottom right = (xmax, ymin) and top left = (xmin, ymax)
        epsg_burst_db, (xmin, ymin, xmax, ymax) = epsg_bbox_dict[burst_id]
        if epsg_bursts is None:
            epsg_burst = epsg_burst_db
        else:
            epsg_burst = epsg_bursts

        # If burst X/Y pixel spacings are not set, use default values (~30 m)
        if x_spacing_bursts is None and epsg_burst == 4326:
            x_spacing_bursts = SECONDS_IN_DEG
        elif x_spacing_bursts is None:
            x_spacing_bursts = 30

        if y_spacing_bursts is None and epsg_burst == 4326:
            y_spacing_bursts = -SECONDS_IN_DEG
        elif y_spacing_bursts is None:
            y_spacing_bursts = -30

        # Initialize geogrid with estimated boundaries
        width = int(np.ceil((xmax - xmin) / x_spacing_bursts))
        length = int(np.ceil((ymin - ymax) / y_spacing_bursts))

        geogrid_burst = isce3.product.GeoGridParameters(
            xmin, ymax, x_spacing_bursts, y_spacing_bursts, width,
            length, epsg_burst)

        # intersect burst geogrid with runconfig parameters
        geogrid_burst = intersect_geogrid(
            geogrid_burst, xmin_bursts, ymax_bursts, xmax_bursts,
            ymin_bursts)

        # Check snap values
        check_snap_values(x_snap_bursts, y_snap_bursts,
                          x_spacing_bursts, y_spacing_bursts)

        # Snap coordinates
        geogrid_burst_snapped = snap_geogrid(geogrid_burst, x_snap_bursts,
                                             y_snap_bursts)

        # Final burst coordinates
        geogrids_dict[burst_id] = geogrid_burst_snapped
        xmin_burst = geogrid_burst_snapped.start_x
        ymax_burst = geogrid_burst_snapped.start_y
        xmax_burst = (geogrid_burst_snapped.start_x +
                      geogrid_burst_snapped.spacing_x *
                      geogrid_burst_snapped.width)
        ymin_burst = (geogrid_burst_snapped.start_y +
                      geogrid_burst_snapped.spacing_y *
                      geogrid_burst_snapped.length)

        # Update bursts envelope.
        # We use EPSG bursts majority because the bursts datatabase
        # use only UTM or polar stereographic and we want to avoid
        # to use geographic (e.g., EPSG = 4326), which may have
        # antimeridian crossing issues
        # (for example, computing max long. between -179 and +179)

        if epsg_burst != epsg_bursts_majority:

            srs_burst = osr.SpatialReference()
            srs_burst.ImportFromEPSG(epsg_burst)

            _, ymin_burst, ymax_burst, xmin_burst, xmax_burst = \
                get_tile_srs_bbox(ymin_burst, ymax_burst,
                                  xmin_burst, xmax_burst,
                                  srs_burst, srs_bursts_majority)

        xmin_all_bursts = min([xmin_all_bursts, xmin_burst])
        ymax_all_bursts = max([ymax_all_bursts, ymax_burst])
        xmax_all_bursts = max([xmax_all_bursts, xmax_burst])
        ymin_all_bursts = min([ymin_all_bursts, ymin_burst])

    if epsg_mosaic != epsg_bursts_majority:

        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(epsg_mosaic)

        _, ymin_all_bursts, ymax_all_bursts, xmin_all_bursts, xmax_all_bursts = \
            get_tile_srs_bbox(ymin_all_bursts, ymax_all_bursts,
                              xmin_all_bursts, xmax_all_bursts,
                              srs_bursts_majority, srs_mosaic)

    # make sure that user's runconfig parameters have precedence
    if xmin_mosaic is None:
        xmin_mosaic = xmin_all_bursts
    if ymax_mosaic is None:
        ymax_mosaic = ymax_all_bursts
    if xmax_mosaic is None:
        xmax_mosaic = xmax_all_bursts
    if ymin_mosaic is None:
        ymin_mosaic = ymin_all_bursts

    width_mosaic = _grid_size(xmax_mosaic, xmin_mosaic, x_spacing_mosaic)
    length_mosaic = _grid_size(ymin_mosaic, ymax_mosaic, y_spacing_mosaic)
    geogrid_mosaic = isce3.product.GeoGridParameters(
        xmin_mosaic, ymax_mosaic, x_spacing_mosaic, y_spacing_mosaic,
        width_mosaic, length_mosaic, epsg_mosaic)

    # Check snap values
    check_snap_values(x_snap_mosaic, y_snap_mosaic, x_spacing_mosaic,
                      y_spacing_mosaic)

    # Snap coordinates
    geogrid_mosaic_snapped = snap_geogrid(geogrid_mosaic, x_snap_mosaic,
                                          y_snap_mosaic)
    return geogrid_mosaic_snapped, geogrids_dict


def generate_geogrids(bursts, geo_dict, mosaic_dict):
    '''
    Compute frame and bursts geogrids

    Parameters
    ----------
    bursts: list[s1reader.s1_burst_slc.Sentinel1BurstSlc]
        List of S-1 burst SLC objects
    geo_dict: dict
        Dictionary containing runconfig processing.geocoding
        parameters
    mosaic_dict: dict
        Dictionary containing runconfig processing.mosaicking
        parameters

    Returns
    -------
    geogrid_all_snapped: isce3.product.GeoGridParameters
        Mosaic geogrid
    geogrids_dict: dict
        Dict containing bursts' geogrids indexed by burst_id
    '''
    mosaic_geogrid_dict = mosaic_dict['mosaic_geogrid']
    epsg_mosaic = mosaic_geogrid_dict['output_epsg']
    xmin_mosaic = mosaic_geogrid_dict['top_left']['x']
    ymax_mosaic = mosaic_geogrid_dict['top_left']['y']
    x_spacing_mosaic = mosaic_geogrid_dict['x_posting']
    y_spacing_positive_mosaic = mosaic_geogrid_dict['y_posting']
    xmax_mosaic = mosaic_geogrid_dict['bottom_right']['x']
    ymin_mosaic = mosaic_geogrid_dict['bottom_right']['y']
    x_snap_mosaic = mosaic_geogrid_dict['x_snap']
    y_snap_mosaic = mosaic_geogrid_dict['y_snap']

    bursts_geogrid_dict = geo_dict['bursts_geogrid']
    epsg_bursts = bursts_geogrid_dict['output_epsg']
    xmin_bursts = bursts_geogrid_dict['top_left']['x']
    ymax_bursts = bursts_geogrid_dict['top_left']['y']
    x_spacing_bursts = bursts_geogrid_dict['x_posting']
    y_spacing_positive_bursts = bursts_geogrid_dict['y_posting']
    xmax_bursts = bursts_geogrid_dict['bottom_right']['x']
    ymin_bursts = bursts_geogrid_dict['bottom_right']['y']
    x_snap_bursts = bursts_geogrid_dict['x_snap']
    y_snap_bursts = bursts_geogrid_dict['y_snap']

    xmin_all_bursts = np.inf
    ymax_all_bursts = -np.inf
    xmax_all_bursts = -np.inf
    ymin_all_bursts = np.inf

    if epsg_mosaic is None and epsg_bursts is not None:
        epsg_mosaic = epsg_bursts

    # Compute mosaic and burst EPSG codes if not assigned in runconfig
    if epsg_mosaic is None:
        y_list = []
        x_list = []
        for burst_pol in bursts.values():
            pol_list = list(burst_pol.keys())
            burst = burst_pol[pol_list[0]]
            y_list.append(burst.center.y)
            x_list.append(burst.center.x)
        y_mean = np.nanmean(y_list)
        x_mean = np.nanmean(x_list)
        epsg_mosaic = get_point_epsg(y_mean, x_mean)
    assert 1024 <= epsg_mosaic <= 32767

    if epsg_bursts is None and epsg_mosaic is not None:
        epsg_bursts = epsg_mosaic


    product_mosaic_str = 'Mosaic'
    y_spacing_mosaic, x_spacing_mosaic = _check_pixel_spacing(
        y_spacing_positive_mosaic, x_spacing_mosaic,
        epsg_mosaic, product_mosaic_str)

    product_burst_str = 'Bursts'
    y_spacing_bursts, x_spacing_bursts = _check_pixel_spacing(
        y_spacing_positive_bursts, x_spacing_bursts,
        epsg_bursts, product_burst_str)

    geogrids_dict = {}
    for burst_id, burst_pol in bursts.items():
        pol_list = list(burst_pol.keys())
        burst = burst_pol[pol_list[0]]
        if burst_id in geogrids_dict.keys():
            continue

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        geogrid_burst = None
        if len(bursts) > 1 or None in [xmin_bursts, ymax_bursts, xmax_bursts,
                                       ymin_bursts]:
            # Initialize geogrid with estimated boundaries
            geogrid_burst = isce3.product.bbox_to_geogrid(
                radar_grid, orbit, isce3.core.LUT2d(), x_spacing_bursts,
                y_spacing_bursts, epsg_bursts)

            if len(bursts) == 1 and None in [xmin_bursts, ymax_bursts,
                                             xmax_bursts, ymin_bursts]:
                # Check and further initialize geogrid
                geogrid_burst = assign_check_geogrid(
                    geogrid_burst, xmin_bursts, ymax_bursts, xmax_bursts,
                    ymin_bursts)
            geogrid_burst = intersect_geogrid(
                geogrid_burst, xmin_bursts, ymax_bursts, xmax_bursts,
                ymin_bursts)
        else:
            # If all the start/end coordinates have been assigned,
            # initialize the geogrid with them
            width = _grid_size(xmax_bursts, xmin_bursts, x_spacing_bursts)
            length = _grid_size(ymin_bursts, ymax_bursts, y_spacing_bursts)
            geogrid_burst = isce3.product.GeoGridParameters(
                xmin_bursts, ymax_bursts, x_spacing_bursts, y_spacing_bursts,
                width, length, epsg_bursts)

        # Check snap values
        check_snap_values(x_snap_bursts, y_snap_bursts, x_spacing_bursts,
                          y_spacing_bursts)
        # Snap coordinates
        geogrid_snapped = snap_geogrid(geogrid_burst, x_snap_bursts,
                                       y_snap_bursts)

        xmin_all_bursts = min([xmin_all_bursts, geogrid_snapped.start_x])
        ymax_all_bursts = max([ymax_all_bursts, geogrid_snapped.start_y])
        xmax_all_bursts = max([xmax_all_bursts,
                                geogrid_snapped.start_x +
                                geogrid_snapped.spacing_x *
                                geogrid_snapped.width])
        ymin_all_bursts = min([ymin_all_bursts,
                                geogrid_snapped.start_y +
                                geogrid_snapped.spacing_y *
                                geogrid_snapped.length])

        geogrids_dict[burst_id] = geogrid_snapped

    if xmin_mosaic is None:
        xmin_mosaic = xmin_all_bursts
    if ymax_mosaic is None:
        ymax_mosaic = ymax_all_bursts
    if xmax_mosaic is None:
        xmax_mosaic = xmax_all_bursts
    if ymin_mosaic is None:
        ymin_mosaic = ymin_all_bursts

    width = _grid_size(xmax_mosaic, xmin_mosaic, x_spacing_mosaic)
    length = _grid_size(ymin_mosaic, ymax_mosaic, y_spacing_mosaic)
    geogrid_all = isce3.product.GeoGridParameters(
        xmin_mosaic, ymax_mosaic, x_spacing_mosaic, y_spacing_mosaic,
        width, length, epsg_mosaic)

    # Check snap values
    check_snap_values(x_snap_mosaic, y_snap_mosaic, x_spacing_mosaic,
                      y_spacing_mosaic)

    # Snap coordinates
    geogrid_all_snapped = snap_geogrid(geogrid_all, x_snap_mosaic, y_snap_mosaic)
    return geogrid_all_snapped, geogrids_dict


def geogrid_as_dict(grid):
    geogrid_dict = {attr:getattr(grid, attr) for attr in grid.__dir__()
                    if attr != 'print' and attr[:2] != '__'}
    return geogrid_dict
