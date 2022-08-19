'''
A module to mosaic Sentinel-1 geobursts from RTC workflow
'''

import os

import numpy as np
from osgeo import osr, gdal

def check_reprojection(list_rtc_imagery: list, list_nlooks: list,
                             geogrid_mosaic) -> bool:
    '''
    Check if the reprojection is necessary to mosaic the input list of the rasters

    Parameters:
    -----------
    list_rtc_imagery: list
        path to the geocoded RTC images to mosaic
    list_nlooks: list
        path to the nlooks raster that corresponds to list_rtc

    Returns:
    flag_rtn: bool
        True if reprojection is necessary to mosaic `list_rtc_imagery`;
        False if the images are aligned, so that no reprojection is necessary.
    '''

    # Accepted error in the coordinates as floating number. Used when checking the snapping.
    maxerr_coord = 1.0e-6

    # Return value after this mosaicking eligibility check.
    # It will turn to False if any of the checks below fails.
    flag_rtn = False

    # Check the number of the raster files in list_rtc_imagery and list_nlooks
    num_rtc_imagery = len(list_rtc_imagery)
    num_nlooks = len(list_nlooks)
    if num_rtc_imagery != num_nlooks:
        #raise ValueError(f'# RTC ({num_rtc}) and # nlooks ({num_nlooks}) does not match.')
        print(f'# RTC ({num_rtc_imagery}) and # nlooks ({num_nlooks}) does not match.')
        flag_rtn = True

    else:
        # Variables to keep record of the geogrid-related information in the input rasters
        str_proj_prev = None
        spacing_x_prev = None
        spacing_y_prev = None
        numbands_prev = None
        mod_x_prev = None
        mod_y_prev = None

        for i, path_rtc_imagery in enumerate(list_rtc_imagery):
            path_nlooks = list_nlooks[i]

            raster_rtc_imagery = gdal.Open(path_rtc_imagery, gdal.GA_ReadOnly)
            raster_nlooks = gdal.Open(path_nlooks, gdal.GA_ReadOnly)

            geo_transformation_rtc_imagery = raster_rtc_imagery.GetGeoTransform()
            geo_transformation_nlooks = raster_nlooks.GetGeoTransform()

            # Compare Geotransform - between RTC and corresponding nlooks
            if geo_transformation_rtc_imagery != geo_transformation_nlooks:
                print('GeoTransform does not match between '+\
                     f'{os.path.basename(path_rtc_imagery)} and {os.path.basename(path_nlooks)}')
                flag_rtn = True

            # Compare dimension - between RTC and corresponding nlooks
            if (raster_rtc_imagery.RasterXSize != raster_nlooks.RasterXSize) or\
            (raster_rtc_imagery.RasterYSize != raster_nlooks.RasterYSize):
                print('Dimension does not agree between '+\
                     f'{os.path.basename(path_rtc_imagery)} and {os.path.basename(path_nlooks)}')
                flag_rtn = True

            # Check number of bands - for all RTC
            if numbands_prev is None:
                numbands_prev = raster_rtc_imagery.RasterCount
            elif numbands_prev != raster_rtc_imagery.RasterCount:
                print(f'Band number anomaly detected from {os.path.basename(path_rtc_imagery)}')
                flag_rtn = True

            # Check projection - for every RTC and nlooks
            if str_proj_prev is None:
                str_proj_prev = raster_rtc_imagery.GetProjection()

            elif str_proj_prev != raster_rtc_imagery.GetProjection():
                print(f'Map projection anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_rtn = True

            elif str_proj_prev != raster_nlooks.GetProjection():
                print(f'Map projection anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_rtn = True


            # spacing x - for all RTC and nlooks
            if spacing_x_prev is None:
                spacing_x_prev = geo_transformation_rtc_imagery[1]

            elif spacing_x_prev != geo_transformation_rtc_imagery[1]:
                print(f'spacing_x anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_rtn = True

            elif spacing_x_prev != geo_transformation_nlooks[1]:
                print(f'spacing_y anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_rtn = True

            # spacing y - for all RTC and nlooks
            if spacing_y_prev is None:
                spacing_y_prev = geo_transformation_rtc_imagery[5]

            elif spacing_y_prev != geo_transformation_rtc_imagery[5]:
                print(f'spacing_y anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_rtn = True

            elif spacing_y_prev != geo_transformation_nlooks[5]:
                print(f'spacing_y anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_rtn = True

            # snapping_x - by calculating the mod of the corner coords - for all RTC and nlooks
            if mod_x_prev is None:
                mod_x_prev = geo_transformation_rtc_imagery[0] % geo_transformation_rtc_imagery[1]

            elif abs(mod_x_prev - geo_transformation_rtc_imagery[0] % geo_transformation_rtc_imagery[1]) > maxerr_coord:
                print(f'snapping_x anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_rtn = True

            elif abs(mod_x_prev - geo_transformation_nlooks[0] % geo_transformation_nlooks[1]) > maxerr_coord:
                print(f'snapping_x anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_rtn = True

            # snapping_y - by calculating the mod of the corner coords - for all RTC and nlooks
            if mod_y_prev is None:
                mod_y_prev = geo_transformation_rtc_imagery[3] % geo_transformation_rtc_imagery[5]
            
            elif abs(mod_y_prev - geo_transformation_rtc_imagery[3] % geo_transformation_rtc_imagery[5]) > maxerr_coord:
                print(f'snapping_y anomaly detected from : {os.path.basename(path_rtc_imagery)}')
                flag_rtn = True

            elif abs(mod_y_prev - geo_transformation_nlooks[3] % geo_transformation_nlooks[5]) > maxerr_coord:
                print(f'snapping_y anomaly detected from : {os.path.basename(path_nlooks)}')
                flag_rtn = True

            raster_rtc_imagery = None
            raster_nlooks = None

        # `list_rtc_imagery` and `list_nooks` are looking good so far.
        # Now check `geogrid_mosaic` with the parameters from the tests so far

        #Check spacing: Use spacing_x_prev, spacing_y_prev
        if geogrid_mosaic is not None:
            if spacing_x_prev != geogrid_mosaic.spacing_x:
                print('spacing_x of the input rasters does not match with that of the mosaic geogrid')
                flag_rtn = True

            if spacing_y_prev != geogrid_mosaic.spacing_y:
                print('spacing_y of the input rasters does not match with that of the mosaic geogrid')
                flag_rtn = True

            #Checo projection: Use str_proj_prev
            srs_mosaic=osr.SpatialReference()
            srs_mosaic.ImportFromEPSG(geogrid_mosaic.epsg)
            if str_proj_prev != srs_mosaic.ExportToWkt():
                print('Projection of the rasters do not match with that of the mosaic geogrid')
                flag_rtn = True


            #Check snapping: Use mod_x_prev, mod_y_prev
            if abs(mod_y_prev - geogrid_mosaic.start_x % geogrid_mosaic.spacing_x) > maxerr_coord:
                print('X coordinate of the raster corner do not align with the mosaic geogrid')
                flag_rtn = True

            if abs(mod_y_prev - geogrid_mosaic.start_y % geogrid_mosaic.spacing_y) > maxerr_coord:
                print('Y coordinate of the raster corner do not align with the mosaic geogrid')
                flag_rtn = True



        print('Passed the mosaic eligiblity test.')

    return flag_rtn




def weighted_mosaic(list_rtc, list_nlooks, geo_filename, geogrid_in=None):
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

    num_raster = len(list_rtc)
    num_bands = None
    posting_x = None
    posting_y = None

    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)

    for i, path_rtc in enumerate(list_rtc):
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

        with gdal.Open(list_rtc[0], gdal.GA_ReadOnly) as raster_in:
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

    for i, path_rtc in enumerate(list_rtc):
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

    # write out the array
    drv_out = gdal.GetDriverByName('Gtiff')
    raster_out = drv_out.Create(geo_filename,
                                dim_mosaic[1], dim_mosaic[0], num_bands,
                                gdal.GDT_Float32)

    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))

    #raster_srs_src = gdal.Open(list_rtc[0], 0)
    raster_out.SetProjection(wkt_projection)
    #raster_srs_src = None

    for i_band in range(num_bands):
        arr_band_writeout = arr_numerator[i_band, :, :] / arr_denominator

        # Deal with abnormal values in the array before writing it out
        arr_band_writeout[np.isinf(arr_band_writeout)] = float('nan')

        raster_out.GetRasterBand(i_band+1).WriteArray(arr_band_writeout)
