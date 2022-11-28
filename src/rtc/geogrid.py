'''
Collection of function for determining and setting the geogrid
'''

import numpy as np
import logging

from nisar.workflows.geogrid import _grid_size
import isce3

logger = logging.getLogger('rtc_s1')

def assign_check_geogrid(geogrid, x_start=None, y_start=None,
                         x_end=None, y_end=None):
    '''
    Initialize geogrid with user defined parameters.
    Check the validity of user-defined parameters

    Parameters
    ----------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object defining the geogrid
    x_start: float
        Geogrid top-left X coordinate
    y_start: float
        Geogrid top-left Y coordinate
    x_end: float
        Geogrid bottom-right X coordinate
    y_end: float
        Geogrid bottom-right Y coordinate

    Returns
    -------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 geogrid initialized with user-defined inputs
    '''

    # Check assigned input coordinates and initialize geogrid accordingly
    if x_start is not None:
        current_end_x = geogrid.start_x + geogrid.spacing_x * geogrid.width
        geogrid.start_x = x_start
        geogrid.width = int(np.ceil((current_end_x - x_start) /
                                     geogrid.spacing_x))
    # Restore geogrid end point if provided by the user
    if x_end is not None:
        geogrid.width = int(np.ceil((x_end - geogrid.start_x) /
                                     geogrid.spacing_x))
    if y_start is not None:
        current_end_y = geogrid.start_y + geogrid.spacing_y * geogrid.length
        geogrid.start_y = y_start
        geogrid.length = int(np.ceil((current_end_y - y_start) /
                                      geogrid.spacing_y))
    if y_end is not None:
        geogrid.length = int(np.ceil((y_end - geogrid.start_y) /
                                      geogrid.spacing_y))

    return geogrid


def intersect_geogrid(geogrid, x_start=None, y_start=None,
                      x_end=None, y_end=None):
    '''
    Return intersected geogrid with user defined parameters.

    Parameters
    ----------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object defining the geogrid
    x_start: float
        Geogrid top-left X coordinate
    y_start: float
        Geogrid top-left Y coordinate
    x_end: float
        Geogrid bottom-right X coordinate
    y_end: float
        Geogrid bottom-right Y coordinate

    Returns
    -------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 geogrid
    '''

    if x_start is not None and x_start > geogrid.start_x:
        current_end_x = geogrid.start_x + geogrid.spacing_x * geogrid.width
        geogrid.start_x = x_start
        geogrid.width = int(np.ceil((current_end_x - x_start) /
                                     geogrid.spacing_x))

    if (x_end is not None and
            (x_end < geogrid.start_x + geogrid.width * geogrid.spacing_x)):
        geogrid.width = int(np.ceil((x_end - geogrid.start_x) /
                                     geogrid.spacing_x))
    if y_start is not None and y_start < geogrid.start_y:
        current_end_y = geogrid.start_y + geogrid.spacing_y * geogrid.length
        geogrid.start_y = y_start
        geogrid.length = int(np.ceil((current_end_y - y_start) /
                                      geogrid.spacing_y))

    if (y_end is not None and 
            (y_end > geogrid.start_y + geogrid.length * geogrid.spacing_y)):
        geogrid.length = int(np.ceil((y_end - geogrid.start_y) /
                                      geogrid.spacing_y))

    return geogrid


def check_geogrid_endpoints(geogrid, x_end=None, y_end=None):
    '''
    Check validity of geogrid end points

    Parameters
    -----------
    geogrid: isce3.product.GeoGridParameters
        ISCE3 object defining the geogrid
    x_end: float
        Geogrid bottom right X coordinate
    y_end: float
        Geogrid bottom right Y coordinates

    Returns
    -------
    x_end: float
        Verified geogrid bottom-right X coordinate
    y_end: float
        Verified geogrid bottom-right Y coordinate
    '''
    end_pt = lambda start, sz, spacing: start + spacing * sz

    if x_end is None:
        x_end = end_pt(geogrid.start_x, geogrid.spacing_x, geogrid.width)
    if y_end is None:
        y_end = end_pt(geogrid.start_y, geogrid.spacing_y, geogrid.length)
    return x_end, y_end


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

    Parameters:
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
    x_end = geogrid.start_x + geogrid.width * geogrid.spacing_x
    y_end = geogrid.start_y + geogrid.length * geogrid.spacing_y

    if x_snap is not None:
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        end_x = snap_coord(x_end, x_snap, np.ceil)
        geogrid.width = _grid_size(end_x, geogrid.start_x, geogrid.spacing_x)
    if y_snap is not None:
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)
        end_y = snap_coord(y_end, y_snap, np.floor)
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


def generate_geogrids(bursts, geo_dict, dem):

    dem_raster = isce3.io.Raster(dem)

    # Unpack values from geocoding disctionary
    epsg = geo_dict['output_epsg']
    x_start = geo_dict['top_left']['x']
    y_start = geo_dict['top_left']['y']
    x_spacing = geo_dict['x_posting']
    y_spacing_positive = geo_dict['y_posting']
    x_end = geo_dict['bottom_right']['x']
    y_end = geo_dict['bottom_right']['y']
    x_snap = geo_dict['x_snap']
    y_snap = geo_dict['y_snap']

    x_start_all_bursts = np.inf
    y_start_all_bursts = -np.inf
    x_end_all_bursts = -np.inf
    y_end_all_bursts = np.inf

    # Compute burst EPSG if not assigned in runconfig
    if epsg is None:
        y_list = []
        x_list = []
        for burst_pol in bursts.values():
            pols = list(burst_pol.keys())
            burst = burst_pol[pols[0]]
            y_list.append(burst.center.y)
            x_list.append(burst.center.x)
        y_mean = np.nanmean(y_list)
        x_mean = np.nanmean(x_list)
        epsg = get_point_epsg(y_mean, x_mean)
    assert 1024 <= epsg <= 32767

    # Check spacing in X direction
    if x_spacing is not None and x_spacing <= 0:
        err_str = 'Pixel spacing in X/longitude direction needs to be positive'
        err_str += f' (x_spacing: {x_spacing})'
        logger.error(err_str)
        raise ValueError(err_str)
    elif x_spacing is None and epsg == 4326:
        x_spacing = -0.00027
    elif x_spacing is None:
        x_spacing = -30

    # Check spacing in Y direction
    if y_spacing_positive is not None and y_spacing_positive <= 0:
        err_str = 'Pixel spacing in Y/latitude direction needs to be positive'
        err_str += f'(y_spacing: {y_spacing_positive})'
        logger.error(err_str)
        raise ValueError(err_str)
    elif y_spacing_positive is None and epsg == 4326:
        y_spacing = -0.00027
    elif y_spacing_positive is None:
        y_spacing = -30
    else:
        y_spacing = - y_spacing_positive

    geogrids_dict = {}
    for burst_obj, burst_pol in bursts.items():
        burst_id = str(burst_obj)
        pols = list(burst_pol.keys())
        burst = burst_pol[pols[0]]
        if burst_id in geogrids_dict:
            continue

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        geogrid_burst = None
        if len(bursts) > 1 or None in [x_start, y_start, x_end, y_end]:
            # Initialize geogrid with estimated boundaries
            geogrid_burst = isce3.product.bbox_to_geogrid(
                radar_grid, orbit, isce3.core.LUT2d(), x_spacing, y_spacing,
                epsg)

            if len(bursts) == 1 and None in [x_start, y_start, x_end, y_end]:
                # Check and further initialize geogrid
                geogrid_burst = assign_check_geogrid(
                    geogrid_burst, x_start, y_start, x_end, y_end)
            geogrid_burst = intersect_geogrid(
                geogrid_burst, x_start, y_start, x_end, y_end)
        else:
            # If all the start/end coordinates have been assigned,
            # initialize the geogrid with them
            width = _grid_size(x_end, x_start, x_spacing)
            length = _grid_size(y_end, y_start, y_spacing)
            geogrid_burst = isce3.product.GeoGridParameters(
                x_start, y_start, x_spacing, y_spacing, width, length,
                epsg)

        # Check snap values
        check_snap_values(x_snap, y_snap, x_spacing, y_spacing)
        # Snap coordinates
        geogrid = snap_geogrid(geogrid_burst, x_snap, y_snap)

        x_start_all_bursts = min([x_start_all_bursts, geogrid.start_x])
        y_start_all_bursts = max([y_start_all_bursts, geogrid.start_y])
        x_end_all_bursts = max([x_end_all_bursts,
                                geogrid.start_x + geogrid.spacing_x *
                                geogrid.width])
        y_end_all_bursts = min([y_end_all_bursts,
                                geogrid.start_y + geogrid.spacing_y *
                                geogrid.length])

        geogrids_dict[burst_id] = geogrid

    if x_start is None:
        x_start = x_start_all_bursts
    if y_start is None:
        y_start = y_start_all_bursts
    if x_end is None:
        x_end = x_end_all_bursts
    if y_end is None:
        y_end = y_end_all_bursts

    width = _grid_size(x_end, x_start, x_spacing)
    length = _grid_size(y_end, y_start, y_spacing)
    geogrid_all = isce3.product.GeoGridParameters(
        x_start, y_start, x_spacing, y_spacing, width, length, epsg)

    # Check snap values
    check_snap_values(x_snap, y_snap, x_spacing, y_spacing)

    # Snap coordinates
    geogrid_all = snap_geogrid(geogrid_all, x_snap, y_snap)
    return geogrid_all, geogrids_dict


def geogrid_as_dict(grid):
    geogrid_dict = {attr:getattr(grid, attr) for attr in grid.__dir__()
                    if attr != 'print' and attr[:2] != '__'}
    return geogrid_dict
