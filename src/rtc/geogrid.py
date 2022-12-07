'''
Collection of function for determining and setting the geogrid
'''

import numpy as np
import logging

from nisar.workflows.geogrid import _grid_size
import isce3

from rtc.helpers import burst_bboxes_from_db

logger = logging.getLogger('rtc_s1')


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








def generate_geogrids_from_db(bursts, geo_dict, dem, burst_db_file):
    '''
    Compute frame and bursts geogrids

    Parameters
    ----------
    bursts: list[s1reader.s1_burst_slc.Sentinel1BurstSlc]
        List of S-1 burst SLC objects
    geo_dict: dict
        Dictionary containing runconfig processing.geocoding
        parameters
    dem_file: str
        Dem file
    burst_db_file : str
        Location of burst database sqlite file

    Returns
    -------
    geogrid_all: isce3.product.GeoGridParameters
        Frame geogrid
    geogrids_dict: dict
        Dict containing bursts geogrids indexed by burst_id
    '''

    # TODO use `dem_raster` to update `epsg`, if not provided in the runconfig
    # dem_raster = isce3.io.Raster(dem)

    # Unpack values from geocoding disctionary
    epsg_runconfig = geo_dict['output_epsg']
    xmin_runconfig = geo_dict['top_left']['x']
    ymax_runconfig = geo_dict['top_left']['y']
    x_spacing = geo_dict['x_posting']
    y_spacing_positive = geo_dict['y_posting']
    xmax_runconfig = geo_dict['bottom_right']['x']
    ymin_runconfig = geo_dict['bottom_right']['y']
    x_snap = geo_dict['x_snap']
    y_snap = geo_dict['y_snap']

    xmin_all_bursts = np.inf
    ymax_all_bursts = -np.inf
    xmax_all_bursts = -np.inf
    ymin_all_bursts = np.inf

    # Check spacing in X direction
    if x_spacing is not None and x_spacing <= 0:
        err_str = 'Pixel spacing in X/longitude direction needs to be positive'
        err_str += f' (x_spacing: {x_spacing})'
        logger.error(err_str)
        raise ValueError(err_str)

    # Check spacing in Y direction
    if y_spacing_positive is not None and y_spacing_positive <= 0:
        err_str = 'Pixel spacing in Y/latitude direction needs to be positive'
        err_str += f'(y_spacing: {y_spacing_positive})'
        logger.error(err_str)
        raise ValueError(err_str)
    elif y_spacing_positive is not None:
        y_spacing = - y_spacing_positive
    else:
        y_spacing = None

    geogrids_dict = {}

    # get all burst IDs and their EPSGs + bounding boxes
    burst_ids = [b[0] for b in bursts]
    epsg_bbox_dict = burst_bboxes_from_db(burst_ids, burst_db_file)

    for burst_id, burst_pol in bursts.items():
        pol_list = list(burst_pol.keys())
        burst = burst_pol[pol_list[0]]
        if burst_id in geogrids_dict.keys():
            continue

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        # extract EPSG and bbox for current burst from dict
        # bottom right = (xmax, ymin) and top left = (xmin, ymax)
        epsg, (xmin, ymin, xmax, ymax) = epsg_bbox_dict[burst_id]

        if epsg != epsg_runconfig:
            err_str = 'ERROR runconfig and burst-DB EPSG codes do not match:'
            err_str += f' runconfig: {epsg_runconfig}.'
            err_str += f' burst-DB: {epsg}'
            logger.error(err_str)
            raise NotImplementedError(err_str)

        if x_spacing is None and epsg == 4326:
            x_spacing = -0.00027
        elif x_spacing is None:
            x_spacing = -30

        if y_spacing_positive is None and epsg == 4326:
            y_spacing = -0.00027
        elif y_spacing_positive is None:
            y_spacing = -30

        # Initialize geogrid with estimated boundaries
        geogrid_burst = isce3.product.bbox_to_geogrid(
            radar_grid, orbit, isce3.core.LUT2d(), x_spacing, y_spacing,
            epsg)

        width = int(np.ceil((xmax - xmin) / x_spacing))
        length = int(np.ceil((ymin - ymax) / y_spacing))

        geogrid_burst = isce3.product.GeoGridParameters(
            xmin, ymax, x_spacing, y_spacing, width,
            length, epsg)

        # intersect burst geogrid with runconfig parameters
        geogrid_burst = intersect_geogrid(
            geogrid_burst, xmin_runconfig, ymax_runconfig, xmax_runconfig,
            ymin_runconfig)

        # Check snap values
        check_snap_values(x_snap, y_snap, x_spacing, y_spacing)

        # Snap coordinates
        geogrid = snap_geogrid(geogrid_burst, x_snap, y_snap)

        xmin_all_bursts = min([xmin_all_bursts, geogrid.start_x])
        ymax_all_bursts = max([ymax_all_bursts, geogrid.start_y])
        xmax_all_bursts = max([xmax_all_bursts,
                                geogrid.start_x + geogrid.spacing_x *
                                geogrid.width])
        ymin_all_bursts = min([ymin_all_bursts,
                                geogrid.start_y + geogrid.spacing_y *
                                geogrid.length])

        geogrids_dict[burst_id] = geogrid

    # make sure that user's runconfig parameters have precedence
    if xmin_runconfig is None:
        xmin_all_bursts = xmin_runconfig
    if ymax_runconfig is None:
        ymax_all_bursts = ymax_runconfig
    if xmax_runconfig is None:
        xmax_all_bursts = xmax_runconfig
    if ymin_runconfig is None:
        ymin_all_bursts = ymin_runconfig

    width_all = _grid_size(xmax_all_bursts, xmin_all_bursts, x_spacing)
    length_all = _grid_size(ymin_all_bursts, ymax_all_bursts, y_spacing)
    geogrid_all = isce3.product.GeoGridParameters(
        xmin_all_bursts, ymax_all_bursts, x_spacing, y_spacing, width_all,
        length_all, epsg)

    # Check snap values
    check_snap_values(x_snap, y_snap, x_spacing, y_spacing)

    # Snap coordinates
    geogrid_all = snap_geogrid(geogrid_all, x_snap, y_snap)
    return geogrid_all, geogrids_dict






def generate_geogrids(bursts, geo_dict, dem):
    '''
    Compute frame and bursts geogrids

    Parameters
    ----------
    bursts: list[s1reader.s1_burst_slc.Sentinel1BurstSlc]
        List of S-1 burst SLC objects
    geo_dict: dict
        Dictionary containing runconfig processing.geocoding
        parameters
    dem_file: str
        Dem file

    Returns
    -------
    geogrid_all: isce3.product.GeoGridParameters
        Frame geogrid
    geogrids_dict: dict
        Dict containing bursts geogrids indexed by burst_id
    '''

    # TODO use `dem_raster` to update `epsg`, if not provided in the runconfig
    # dem_raster = isce3.io.Raster(dem)

    # Unpack values from geocoding disctionary
    epsg = geo_dict['output_epsg']
    xmin = geo_dict['top_left']['x']
    ymax = geo_dict['top_left']['y']
    x_spacing = geo_dict['x_posting']
    y_spacing_positive = geo_dict['y_posting']
    xmax = geo_dict['bottom_right']['x']
    ymin = geo_dict['bottom_right']['y']
    x_snap = geo_dict['x_snap']
    y_snap = geo_dict['y_snap']

    xmin_all_bursts = np.inf
    ymax_all_bursts = -np.inf
    xmax_all_bursts = -np.inf
    ymin_all_bursts = np.inf

    # Compute burst EPSG if not assigned in runconfig
    if epsg is None:
        y_list = []
        x_list = []
        for burst_pol in bursts.values():
            pol_list = list(burst_pol.keys())
            burst = burst_pol[pol_list[0]]
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
    for burst_id, burst_pol in bursts.items():
        pol_list = list(burst_pol.keys())
        burst = burst_pol[pol_list[0]]
        if burst_id in geogrids_dict.keys():
            continue

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        geogrid_burst = None
        if len(bursts) > 1 or None in [xmin, ymax, xmax, ymin]:
            # Initialize geogrid with estimated boundaries
            geogrid_burst = isce3.product.bbox_to_geogrid(
                radar_grid, orbit, isce3.core.LUT2d(), x_spacing, y_spacing,
                epsg)

            if len(bursts) == 1 and None in [xmin, ymax, xmax, ymin]:
                # Check and further initialize geogrid
                geogrid_burst = assign_check_geogrid(
                    geogrid_burst, xmin, ymax, xmax, ymin)
            geogrid_burst = intersect_geogrid(
                geogrid_burst, xmin, ymax, xmax, ymin)
        else:
            # If all the start/end coordinates have been assigned,
            # initialize the geogrid with them
            width = _grid_size(xmax, xmin, x_spacing)
            length = _grid_size(ymin, ymax, y_spacing)
            geogrid_burst = isce3.product.GeoGridParameters(
                xmin, ymax, x_spacing, y_spacing, width, length,
                epsg)

        # Check snap values
        check_snap_values(x_snap, y_snap, x_spacing, y_spacing)
        # Snap coordinates
        geogrid = snap_geogrid(geogrid_burst, x_snap, y_snap)

        xmin_all_bursts = min([xmin_all_bursts, geogrid.start_x])
        ymax_all_bursts = max([ymax_all_bursts, geogrid.start_y])
        xmax_all_bursts = max([xmax_all_bursts,
                                geogrid.start_x + geogrid.spacing_x *
                                geogrid.width])
        ymin_all_bursts = min([ymin_all_bursts,
                                geogrid.start_y + geogrid.spacing_y *
                                geogrid.length])

        geogrids_dict[burst_id] = geogrid

    if xmin is None:
        xmin = xmin_all_bursts
    if ymax is None:
        ymax = ymax_all_bursts
    if xmax is None:
        xmax = xmax_all_bursts
    if ymin is None:
        ymin = ymin_all_bursts

    width = _grid_size(xmax, xmin, x_spacing)
    length = _grid_size(ymin, ymax, y_spacing)
    geogrid_all = isce3.product.GeoGridParameters(
        xmin, ymax, x_spacing, y_spacing, width, length, epsg)

    # Check snap values
    check_snap_values(x_snap, y_snap, x_spacing, y_spacing)

    # Snap coordinates
    geogrid_all = snap_geogrid(geogrid_all, x_snap, y_snap)
    return geogrid_all, geogrids_dict


def geogrid_as_dict(grid):
    geogrid_dict = {attr:getattr(grid, attr) for attr in grid.__dir__()
                    if attr != 'print' and attr[:2] != '__'}
    return geogrid_dict
