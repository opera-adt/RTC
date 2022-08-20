'''
A module to mosaic Sentinel-1 geobursts from RTC workflow
'''

import os

import numpy as np
from osgeo import osr, gdal

def check_reprojection(list_rtc_imagery: list, list_nlooks: list,
                             geogrid_mosaic) -> bool:
    '''
    Check if the reprojection is required to mosaic the input list of the rasters

    Parameters:
    -----------
    list_rtc_imagery: list
        path to the geocoded RTC images to mosaic
    list_nlooks: list
        path to the nlooks raster that corresponds to list_rtc

    Returns:
    flag_requires_reprojection: bool
        True if reprojection is necessary to mosaic `list_rtc_imagery`;
        False if the images are aligned, so that no reprojection is necessary.
        None if the mosaicking is not possible.
    '''

    # Accepted error in the coordinates as floating number. Used when checking the snapping.
    maxerr_coord = 1.0e-6

    # Return value after this reprojection requirement check.
    # It will turn to `True` if reprojection is necessary,
    # or `None` if mosaicking is not possible.
    flag_requires_reprojection = False

    # Check the number of the raster files in list_rtc_imagery and list_nlooks
    num_rtc_imagery = len(list_rtc_imagery)
    num_nlooks = len(list_nlooks)
    if num_rtc_imagery != num_nlooks:
        print (f'# RTC ({num_rtc_imagery}) and # nlooks ({num_nlooks}) does not match.')
        flag_requires_reprojection = None

    else:
        # Variables to check the image parameters the geogrid-related information in the input rasters

        # Projection string of the 1st image in `list_rtc_imagery`
        str_proj_0 = None

        # x/y spacing of the 1st image in `list_rtc_imagery`
        spacing_x_0 = None
        spacing_y_0 = None

        # Number of the bands of the 1st image in `list_rtc_imagery`
        numbands_0 = None

        # Modulo of the x and y coordinates of the 1st image in `list_rtc_imagery`
        #  - To check if the images in the input lists are separated with multiples of the spacing
        mod_x_0 = None
        mod_y_0 = None

        for i, path_rtc_imagery in enumerate(list_rtc_imagery):
            path_nlooks = list_nlooks[i]

            raster_rtc_imagery = gdal.Open(path_rtc_imagery, gdal.GA_ReadOnly)
            raster_nlooks = gdal.Open(path_nlooks, gdal.GA_ReadOnly)

            geo_transformation_rtc_imagery = raster_rtc_imagery.GetGeoTransform()
            geo_transformation_nlooks = raster_nlooks.GetGeoTransform()

            # Compare Geotransform - between RTC imagery and corresponding nlooks
            if geo_transformation_rtc_imagery != geo_transformation_nlooks:
                print('GeoTransform does not match between '+\
                     f'{os.path.basename(path_rtc_imagery)} and {os.path.basename(path_nlooks)}')
                flag_requires_reprojection = True
                break

            # Compare dimension - between RTC imagery and corresponding nlooks
            if (raster_rtc_imagery.RasterXSize != raster_nlooks.RasterXSize) or\
            (raster_rtc_imagery.RasterYSize != raster_nlooks.RasterYSize):
                print('Dimension does not agree between '+\
                     f'{os.path.basename(path_rtc_imagery)} and {os.path.basename(path_nlooks)}')
                flag_requires_reprojection = True
                break

            # Check number of bands - for all RTC imagery
            if numbands_0 is None:
                numbands_0 = raster_rtc_imagery.RasterCount
            elif numbands_0 != raster_rtc_imagery.RasterCount:
                print(f'Band number anomaly detected from {os.path.basename(path_rtc_imagery)}')
                flag_requires_reprojection = None
                break

            # Check projection - for all RTC imagery and nlooks
            if str_proj_0 is None:
                str_proj_0 = raster_rtc_imagery.GetProjection()

            elif str_proj_0 != raster_rtc_imagery.GetProjection():
                print(f'Map projection anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_requires_reprojection = True
                break

            elif str_proj_0 != raster_nlooks.GetProjection():
                print(f'Map projection anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_requires_reprojection = True
                break

            # spacing x - for all RTC imagery and nlooks
            if spacing_x_0 is None:
                spacing_x_0 = geo_transformation_rtc_imagery[1]

            elif spacing_x_0 != geo_transformation_rtc_imagery[1]:
                print(f'spacing_x anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_requires_reprojection = True
                break

            elif spacing_x_0 != geo_transformation_nlooks[1]:
                print(f'spacing_y anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_requires_reprojection = True
                break

            # spacing y - for all RTC imagery and nlooks
            if spacing_y_0 is None:
                spacing_y_0 = geo_transformation_rtc_imagery[5]

            elif spacing_y_0 != geo_transformation_rtc_imagery[5]:
                print(f'spacing_y anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_requires_reprojection = True
                break

            elif spacing_y_0 != geo_transformation_nlooks[5]:
                print(f'spacing_y anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_requires_reprojection = True
                break

            # separation_by_n_times_of_spacing_x for all RTC imagery and nlooks
            if mod_x_0 is None:
                mod_x_0 = geo_transformation_rtc_imagery[0] % geo_transformation_rtc_imagery[1]

            elif abs(mod_x_0 - geo_transformation_rtc_imagery[0] % geo_transformation_rtc_imagery[1]) > maxerr_coord:
                print(f'snapping_x anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_requires_reprojection = True
                break

            elif abs(mod_x_0 - geo_transformation_nlooks[0] % geo_transformation_nlooks[1]) > maxerr_coord:
                print(f'snapping_x anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_requires_reprojection = True
                break

            # separation_by_n_times_of_spacing_y for all RTC imagery and nlooks
            if mod_y_0 is None:
                mod_y_0 = geo_transformation_rtc_imagery[3] % geo_transformation_rtc_imagery[5]
            
            elif abs(mod_y_0 - geo_transformation_rtc_imagery[3] % geo_transformation_rtc_imagery[5]) > maxerr_coord:
                print(f'snapping_y anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_requires_reprojection = True
                break

            elif abs(mod_y_0 - geo_transformation_nlooks[3] % geo_transformation_nlooks[5]) > maxerr_coord:
                print(f'snapping_y anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_requires_reprojection = True
                break

            raster_rtc_imagery = None
            raster_nlooks = None

        # `list_rtc_imagery` and `list_nooks` are looking good so far.
        # Now check `geogrid_mosaic` with the parameters from the previous tests

        #Check spacing: Use spacing_x_0, spacing_y_0
        if (not flag_requires_reprojection) and (geogrid_mosaic is not None):
            if spacing_x_0 != geogrid_mosaic.spacing_x:
                print('spacing_x of the input rasters does not match with that of the mosaic geogrid')
                flag_requires_reprojection = True

            elif spacing_y_0 != geogrid_mosaic.spacing_y:
                print('spacing_y of the input rasters does not match with that of the mosaic geogrid')
                flag_requires_reprojection = True

            #Check projection: Use str_proj_0
            srs_mosaic = osr.SpatialReference()
            srs_mosaic.ImportFromEPSG(geogrid_mosaic.epsg)
            if str_proj_0 != srs_mosaic.ExportToWkt():
                print('Projection of the rasters do not match with that of the mosaic geogrid')
                flag_requires_reprojection = True

            #Check the coord. separation in x and y: Use mod_x_0, mod_x_0
            if abs(mod_x_0 - geogrid_mosaic.start_x % geogrid_mosaic.spacing_x) > maxerr_coord:
                print('X coordinate of the raster corner do not align with the mosaic geogrid')
                flag_requires_reprojection = True

            elif abs(mod_y_0 - geogrid_mosaic.start_y % geogrid_mosaic.spacing_y) > maxerr_coord:
                print('Y coordinate of the raster corner do not align with the mosaic geogrid')
                flag_requires_reprojection = True

    return flag_requires_reprojection




def weighted_mosaic(list_rtc_imagery, list_nlooks, geo_filename, geogrid_in=None):
    '''
    Mosaic the snapped S1 geobursts
    paremeters:
    -----------
        list_rtc: list
            list of the path to the rtc geobursts
        list_nlooks: list
            list of the nlooks raster that corresponds to list_rtc
        geo_filename: str
            Path to the output mosaic
        geogrid_in: isce3.product.GeoGridParameters, default: None
            geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None

    '''

    num_raster = len(list_rtc_imagery)
    num_bands = None
    posting_x = None
    posting_y = None

    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)

    for i, path_rtc in enumerate(list_rtc_imagery):
        print(f'Loading geocoding info: {i+1} of {num_raster}')
        raster_in = gdal.Open(path_rtc, gdal.GA_ReadOnly)
        list_geo_transform[i, :] = raster_in.GetGeoTransform()
        list_dimension[i, :] = (raster_in.RasterYSize, raster_in.RasterXSize)

        # Check if the number of bands are consistent over the input RTC rasters
        if num_bands is None:
            num_bands = raster_in.RasterCount
            continue
        elif num_bands != raster_in.RasterCount:
            raise ValueError(f'Anomaly detected on # of bands from source file: {os.path.basename(path_rtc)}')

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

        with gdal.Open(list_rtc_imagery[0], gdal.GA_ReadOnly) as raster_in:
            wkt_projection = raster_in.GetProjectionRef()
            #epsg_mosaic = None

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

    print(f'mosaic dimension: {dim_mosaic}, #bands: {num_bands}')

    arr_numerator = np.zeros((num_bands, dim_mosaic[0], dim_mosaic[1]))
    arr_denominator = np.zeros(dim_mosaic)

    for i, path_rtc in enumerate(list_rtc_imagery):
        path_nlooks = list_nlooks[i]
        print(f'Mosaicking: {i+1} of {num_raster}: {os.path.basename(path_rtc)}', end=' ')

        # calculate the burst RTC's offset wrt. the output mosaic in the image coordinate
        offset_imgx = int((list_geo_transform[i,0] - xmin_mosaic) / posting_x + 0.5)
        offset_imgy = int((list_geo_transform[i,3] - ymax_mosaic) / posting_y + 0.5)

        print(f'img. offset [x, y] = [{offset_imgx}, {offset_imgy}]')
        raster_rtc = gdal.Open(path_rtc,0)
        arr_rtc = raster_rtc.ReadAsArray()

        #reshape arr_rtc when it is a singleband raster: to make it compatible in the for loop below
        if num_bands==1:
            arr_rtc=arr_rtc.reshape((1, arr_rtc.shape[0], arr_rtc.shape[1]))

        # Replace NaN values with 0
        arr_rtc[np.isnan(arr_rtc)] = 0.0

        raster_nlooks = gdal.Open(path_nlooks, 0)
        arr_nlooks = raster_nlooks.ReadAsArray()
        arr_nlooks[np.isnan(arr_nlooks)] = 0.0 #check it zero filling is necessary

        for i_band in range(num_bands):
            arr_numerator[i_band,
                          offset_imgy:offset_imgy+list_dimension[i, 0],
                          offset_imgx:offset_imgx+list_dimension[i, 1]] += arr_rtc[i_band, :, :] * arr_nlooks

        arr_denominator[offset_imgy:offset_imgy + list_dimension[i, 0],
                        offset_imgx:offset_imgx + list_dimension[i, 1]] += arr_nlooks

        raster_rtc = None
        raster_nlooks = None

    # Retreive the datatype information from the first input image
    reference_raster = gdal.Open(list_rtc_imagery[0], gdal.GA_ReadOnly)
    datatype_mosaic = reference_raster.GetRasterBand(1).DataType
    reference_raster = None

    # write out the array
    drv_out = gdal.GetDriverByName('Gtiff')
    raster_out = drv_out.Create(geo_filename,
                                dim_mosaic[1], dim_mosaic[0], num_bands,
                                datatype_mosaic)

    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))

    raster_out.SetProjection(wkt_projection)

    for i_band in range(num_bands):
        arr_band_writeout = arr_numerator[i_band, :, :] / arr_denominator

        # Deal with abnormal values in the array before writing it out
        arr_band_writeout[np.isinf(arr_band_writeout)] = float('nan')

        raster_out.GetRasterBand(i_band+1).WriteArray(arr_band_writeout)
