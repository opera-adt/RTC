'''
A module to mosaic Sentinel-1 geobursts from RTC workflow
'''

import os

import numpy as np
from osgeo import osr, gdal

def check_reprojection(geogrid_mosaic,
                       rtc_image: str,
                       nlooks_image: str = None) -> bool:
    '''
    Check if the reprojection is required to mosaic input raster

    Parameters:
    -----------
    geogrid_mosaic: isce3.product.GeoGridParameters
        Mosaic geogrid
    rtc_image: str
        Path to the geocoded RTC image
    nlooks_image: str (optional)
        Path to the nlooks raster

    Returns:
    flag_requires_reprojection: bool
        True if reprojection is necessary to mosaic inputs
        False if the images are aligned, so that no reprojection is necessary.
    '''

    # Accepted error in the coordinates as floating number
    maxerr_coord = 1.0e-6

    raster_rtc_image = gdal.Open(rtc_image, gdal.GA_ReadOnly)
    if nlooks_image is not None:
        raster_nlooks = gdal.Open(nlooks_image, gdal.GA_ReadOnly)

    # Compare geotransforms of RTC image and nlooks (if provided)
    if (nlooks_image is not None and
            raster_rtc_image.GetGeoTransform() !=
            raster_nlooks.GetGeoTransform()):
        error_str = (f'ERROR geolocations of {raster_rtc_image} and'
                     f' {raster_nlooks} do not match')
        raise ValueError(error_str)

    # Compare dimension - between RTC imagery and corresponding nlooks
    if (nlooks_image is not None and
        (raster_rtc_image.RasterXSize != raster_nlooks.RasterXSize or
            raster_rtc_image.RasterYSize != raster_nlooks.RasterYSize)):
        error_str = (f'ERROR dimensions of {raster_rtc_image} and'
                     f' {raster_nlooks} do not match')
        raise ValueError(error_str)

    rasters_to_check = [raster_rtc_image]
    if raster_nlooks is not None:
        rasters_to_check += [raster_nlooks]

    for raster in rasters_to_check:
        geotransform = raster.GetGeoTransform()
        projection = raster.GetProjection()

        x0 = geotransform[0]    
        dx = geotransform[1]
        y0 = geotransform[3]
        dy = geotransform[5]

        # check spacing
        if dx != geogrid_mosaic.spacing_x:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        if dy != geogrid_mosaic.spacing_y:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        # check projection
        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(geogrid_mosaic.epsg)

        if projection != srs_mosaic.ExportToWkt():
            srs_1 = osr.SpatialReference()
            srs_1.SetWellKnownGeogCS(projection)

            srs_2 = osr.SpatialReference()
            srs_2.SetWellKnownGeogCS(projection)

            if not srs_1.IsSame(srs_2):
                flag_requires_reprojection = True
                return flag_requires_reprojection

        # check the coordinates
        if (abs((x0 - geogrid_mosaic.start_x) % geogrid_mosaic.spacing_x) >
                maxerr_coord):
            flag_requires_reprojection = True
            return flag_requires_reprojection

        if (abs((y0 - geogrid_mosaic.start_y) % geogrid_mosaic.spacing_y) >
                maxerr_coord):
            flag_requires_reprojection = True
            return flag_requires_reprojection

    flag_requires_reprojection = False
    return flag_requires_reprojection


def _weighted_mosaic(list_rtc_images, list_nlooks,
                     geogrid_in=None, verbose = True):
    '''
    Mosaic S-1 geobursts and return mosaic dictionary
    paremeters:
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None
    Returns:
        mosaic_dict: dict
            Mosaic dictionary
    '''

    num_raster = len(list_rtc_images)
    num_bands = None
    posting_x = None
    posting_y = None

    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)

    for i, path_rtc in enumerate(list_rtc_images):
        if verbose:
            print(f'loading geocoding info: {i+1} of {num_raster}')
        
        raster_in = gdal.Open(path_rtc, gdal.GA_ReadOnly)
        list_geo_transform[i, :] = raster_in.GetGeoTransform()
        list_dimension[i, :] = (raster_in.RasterYSize, raster_in.RasterXSize)

        # Check if the number of bands are consistent over the input RTC rasters
        if num_bands is None:
            num_bands = raster_in.RasterCount
            continue
        elif num_bands != raster_in.RasterCount:
            raise ValueError(f'Anomaly detected on # of bands from source'
                             f' file: {os.path.basename(path_rtc)}')

        raster_in = None


    if geogrid_in is None:
        # determine GeoTransformation, posting, dimension, and projection from the input raster
        for i in range(num_raster):
            if list_geo_transform[:, 1].max() == list_geo_transform[:, 1].min():
                posting_x = list_geo_transform[0,1]

            if list_geo_transform[:, 5].max() == list_geo_transform[:, 5].min():
                posting_y = list_geo_transform[0,5]

        # determine the dimension and the upper left corner of the output mosaic
        xmin_mosaic = list_geo_transform[:, 0].min()
        ymax_mosaic = list_geo_transform[:, 3].max()
        xmax_mosaic = (list_geo_transform[:, 0] + list_geo_transform[:, 1]*list_dimension[:, 1]).max()
        ymin_mosaic = (list_geo_transform[:, 3] + list_geo_transform[:, 5]*list_dimension[:, 0]).min()

        dim_mosaic = (int(np.ceil((ymin_mosaic - ymax_mosaic) / posting_y)),
                      int(np.ceil((xmax_mosaic - xmin_mosaic) / posting_x)))

        with gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly) as raster_in:
            wkt_projection = raster_in.GetProjectionRef()

    else:
        # Directly bring the geogrid information from the input parameter
        xmin_mosaic = geogrid_in.start_x
        ymax_mosaic = geogrid_in.start_y
        posting_x = geogrid_in.spacing_x
        posting_y = geogrid_in.spacing_y

        dim_mosaic = (geogrid_in.length, geogrid_in.width)

        xmax_mosaic = xmin_mosaic + posting_x*dim_mosaic[1]
        ymin_mosaic = ymax_mosaic + posting_y*dim_mosaic[0]
        dim_mosaic = (geogrid_in.length, geogrid_in.width)

        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(geogrid_in.epsg)
        wkt_projection = srs_mosaic.ExportToWkt()

    if verbose:
        print(f'mosaic dimension: {dim_mosaic}, #bands: {num_bands}')

    arr_numerator = np.zeros((num_bands, dim_mosaic[0], dim_mosaic[1]))
    arr_denominator = np.zeros(dim_mosaic)

    for i, path_rtc in enumerate(list_rtc_images):
        path_nlooks = list_nlooks[i]
        if verbose:
            print(f'mosaicking: {i+1} of {num_raster}: {os.path.basename(path_rtc)}')

        if geogrid_in is not None and check_reprojection(
                geogrid_in, path_rtc, path_nlooks):
            # reprojection not implemented
            raise NotImplementedError

        # TODO: if geogrid_in is None, check reprojection
                           
        # calculate the burst RTC's offset wrt. the output mosaic in the image coordinate
        offset_imgx = int((list_geo_transform[i,0] - xmin_mosaic) / posting_x + 0.5)
        offset_imgy = int((list_geo_transform[i,3] - ymax_mosaic) / posting_y + 0.5)

        if verbose:
            print(f'image offset [x, y] = [{offset_imgx}, {offset_imgy}]')
        raster_rtc = gdal.Open(path_rtc,0)
        arr_rtc = raster_rtc.ReadAsArray()

        #reshape arr_rtc when it is a singleband raster: to make it compatible in the for loop below
        if num_bands==1:
            arr_rtc=arr_rtc.reshape((1, arr_rtc.shape[0], arr_rtc.shape[1]))

        # Replace NaN values with 0
        arr_rtc[np.isnan(arr_rtc)] = 0.0

        raster_nlooks = gdal.Open(path_nlooks, 0)
        arr_nlooks = raster_nlooks.ReadAsArray()
        invalid_ind = np.isnan(arr_nlooks) 
        arr_nlooks[invalid_ind] = 0.0

        for i_band in range(num_bands):
            arr_numerator[i_band,
                          offset_imgy:offset_imgy+list_dimension[i, 0],
                          offset_imgx:offset_imgx+list_dimension[i, 1]] += \
                            arr_rtc[i_band, :, :] * arr_nlooks

        arr_denominator[offset_imgy:offset_imgy + list_dimension[i, 0],
                        offset_imgx:offset_imgx + list_dimension[i, 1]] += \
                            arr_nlooks

        raster_rtc = None
        raster_nlooks = None

    for i_band in range(num_bands):
        valid_ind = np.where(arr_denominator > 0)
        arr_numerator[i_band][valid_ind] = \
            arr_numerator[i_band][valid_ind] / arr_denominator[valid_ind]

        invalid_ind = np.where(arr_denominator == 0)
        arr_numerator[i_band][invalid_ind] = np.nan

    mosaic_dict = {
        'mosaic_array': arr_numerator,
        'length': dim_mosaic[0],
        'width': dim_mosaic[1],
        'num_bands': num_bands,
        'wkt_projection': wkt_projection,
        'xmin_mosaic': xmin_mosaic,
        'ymax_mosaic': ymax_mosaic,
        'posting_x': posting_x,
        'posting_y': posting_y
    }
    return mosaic_dict




def weighted_mosaic(list_rtc_images, list_nlooks, geo_filename,
                    geogrid_in=None, verbose = True):
    '''
    Mosaic the snapped S1 geobursts
    paremeters:
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        geo_filename: str
            Path to the output mosaic
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None

    '''
    mosaic_dict = _weighted_mosaic(list_rtc_images, list_nlooks,
                                   geogrid_in=geogrid_in, verbose = verbose)

    arr_numerator = mosaic_dict['mosaic_array']
    length = mosaic_dict['length']
    width = mosaic_dict['width']
    num_bands = mosaic_dict['num_bands']
    wkt_projection = mosaic_dict['wkt_projection']
    xmin_mosaic = mosaic_dict['xmin_mosaic']
    ymax_mosaic = mosaic_dict['ymax_mosaic']
    posting_x = mosaic_dict['posting_x']
    posting_y = mosaic_dict['posting_y']

    # Retrieve the datatype information from the first input image
    reference_raster = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
    datatype_mosaic = reference_raster.GetRasterBand(1).DataType
    reference_raster = None

    # Write out the array
    drv_out = gdal.GetDriverByName('Gtiff')
    raster_out = drv_out.Create(geo_filename,
                                width, length, num_bands,
                                datatype_mosaic)

    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))

    raster_out.SetProjection(wkt_projection)

    for i_band in range(num_bands):
        raster_out.GetRasterBand(i_band+1).WriteArray(arr_numerator[i_band])



def weighted_mosaic_single_band(list_rtc_images, list_nlooks,
                                output_file_list,
                                geogrid_in=None, verbose = True):
    '''
    Mosaic the snapped S1 geobursts
    paremeters:
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        output_file_list: list
            Output file list
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None

    '''
    mosaic_dict = _weighted_mosaic(list_rtc_images, list_nlooks,
                                     geogrid_in=geogrid_in, verbose = verbose)

    arr_numerator = mosaic_dict['mosaic_array']
    length = mosaic_dict['length']
    width = mosaic_dict['width']
    num_bands = mosaic_dict['num_bands']
    wkt_projection = mosaic_dict['wkt_projection']
    xmin_mosaic = mosaic_dict['xmin_mosaic']
    ymax_mosaic = mosaic_dict['ymax_mosaic']
    posting_x = mosaic_dict['posting_x']
    posting_y = mosaic_dict['posting_y']

    if num_bands != len(output_file_list):
        error_str = (f'ERROR number of output files ({len(output_file_list)})'
                    f' does not match with the number'
                     f' of input bursts` bands ({num_bands})')
        raise ValueError(error_str)

    for i_band, output_file in enumerate(output_file_list):

        # Retrieve the datatype information from the first input image
        reference_raster = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
        datatype_mosaic = reference_raster.GetRasterBand(1).DataType
        reference_raster = None

        # Write out the array
        drv_out = gdal.GetDriverByName('Gtiff')
        raster_out = drv_out.Create(output_file,
                                    width, length, num_bands,
                                    datatype_mosaic)

        raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))

        raster_out.SetProjection(wkt_projection)

        for i_band in range(num_bands):
            raster_out.GetRasterBand(i_band+1).WriteArray(arr_numerator[i_band])
