'''
A module to mosaic Sentinel-1 geobursts from RTC workflow
'''

import os

import numpy as np
import tempfile
from osgeo import osr, gdal
from scipy import ndimage


def check_reprojection(geogrid_mosaic,
                       rtc_image: str,
                       nlooks_image: str = None) -> bool:
    '''
    Check if the reprojection is required to mosaic input raster

    Parameters
    -----------
    geogrid_mosaic: isce3.product.GeoGridParameters
        Mosaic geogrid
    rtc_image: str
        Path to the geocoded RTC image
    nlooks_image: str (optional)
        Path to the nlooks raster

    Returns
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
    if nlooks_image is not None:
        rasters_to_check += [raster_nlooks]

    for raster in rasters_to_check:
        x0, dx, _, y0, _, dy = raster.GetGeoTransform()
        projection = raster.GetProjection()

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


def _compute_distance_to_burst_center(image, geotransform):
    '''
    Compute distance from burst center

    Parameters:
    -----------
       image: np.ndarray
           Input image
       geotransform: list(float)
           Data geotransform

    Returns
        distance_image: np.ndarray
            Distance image
    '''

    length, width = image.shape
    center_of_mass = ndimage.center_of_mass(np.isfinite(image))

    x_vector = np.arange(width, dtype=np.float32)
    y_vector = np.arange(length, dtype=np.float32)

    _, dx, _, _, _, dy = geotransform

    x_distance_image, y_distance_image = np.meshgrid(x_vector, y_vector)
    distance = np.sqrt((dy * (y_distance_image - center_of_mass[0])) ** 2 +
                       (dx * (x_distance_image - center_of_mass[1])) ** 2 )

    return distance


def compute_mosaic_array(list_rtc_images, list_nlooks, mosaic_mode, scratch_dir='',
                         geogrid_in=None, temp_files_list=None, verbose=True):
    '''
    Mosaic S-1 geobursts and return the mosaic as dictionary

    Parameters:
    -----------
       list_rtc: list
           List of the path to the rtc geobursts
       list_nlooks: list
           List of the nlooks raster that corresponds to list_rtc
       mosaic_mode: str
            Mosaic mode. Choices: "average", "first", and "bursts_center"
       scratch_dir: str (optional)
            Directory for temporary files
       geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None
       temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
       verbose: flag (optional)
            Flag to enable (True) or disable (False) the verbose mode
    Returns
        mosaic_dict: dict
            Mosaic dictionary
    '''

    mosaic_mode_choices_list = ['average', 'first', 'bursts_center']
    if mosaic_mode.lower() not in mosaic_mode_choices_list:
        raise ValueError(f'ERROR invalid mosaic mode: {mosaic_mode}.'
                         f' Choices: {", ".join(mosaic_mode_choices_list)}')

    num_raster = len(list_rtc_images)
    description_list = []
    num_bands = None
    posting_x = None
    posting_y = None

    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)

    for i, path_rtc in enumerate(list_rtc_images):

        raster_in = gdal.Open(path_rtc, gdal.GA_ReadOnly)
        list_geo_transform[i, :] = raster_in.GetGeoTransform()
        list_dimension[i, :] = (raster_in.RasterYSize, raster_in.RasterXSize)

        # Check if the number of bands are consistent over the input RTC rasters
        if num_bands is None:
            num_bands = raster_in.RasterCount

        elif num_bands != raster_in.RasterCount:
            raise ValueError(f'ERROR: the file "{os.path.basename(path_rtc)}"'
                             f' has {raster_in.RasterCount} bands. Expected:'
                             f' {num_bands}.') 

        if len(description_list) == 0:
            for i_band in range(num_bands):
                description_list.append(
                    raster_in.GetRasterBand(i_band+1).GetDescription())

        # Close GDAL dataset
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
        xmax_mosaic = (list_geo_transform[:, 0] +
                       list_geo_transform[:, 1]*list_dimension[:, 1]).max()
        ymin_mosaic = (list_geo_transform[:, 3] +
                       list_geo_transform[:, 5]*list_dimension[:, 0]).min()

        dim_mosaic = (int(np.ceil((ymin_mosaic - ymax_mosaic) / posting_y)),
                      int(np.ceil((xmax_mosaic - xmin_mosaic) / posting_x)))

        gdal_ds_raster_in = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
        wkt_projection = gdal_ds_raster_in.GetProjectionRef()
        del gdal_ds_raster_in

    else:
        # Directly bring the geogrid information from the input parameter
        xmin_mosaic = geogrid_in.start_x
        ymax_mosaic = geogrid_in.start_y
        posting_x = geogrid_in.spacing_x
        posting_y = geogrid_in.spacing_y

        dim_mosaic = (geogrid_in.length, geogrid_in.width)

        xmax_mosaic = xmin_mosaic + posting_x * dim_mosaic[1]
        ymin_mosaic = ymax_mosaic + posting_y * dim_mosaic[0]

        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(geogrid_in.epsg)
        wkt_projection = srs_mosaic.ExportToWkt()

    if verbose:
        print(f'    mosaic dimension: {dim_mosaic}, number of bands: {num_bands}')

    if mosaic_mode.lower() == 'average':
        arr_numerator = np.zeros((num_bands, dim_mosaic[0], dim_mosaic[1]),
                                 dtype=float)
        arr_denominator = np.zeros(dim_mosaic, dtype=float)
    else:
        arr_numerator = np.full((num_bands, dim_mosaic[0], dim_mosaic[1]),
                                np.nan, dtype=float)
        if mosaic_mode.lower() == 'bursts_center':
            arr_distance = np.full(dim_mosaic, np.nan, dtype=float)

    for i, path_rtc in enumerate(list_rtc_images):
        if i < len(list_nlooks):
            path_nlooks = list_nlooks[i]
        else:
            path_nlooks = None

        if verbose:
            print(f'    mosaicking ({i+1}/{num_raster}): {os.path.basename(path_rtc)}')

        if geogrid_in is not None and check_reprojection(
                geogrid_in, path_rtc, path_nlooks):
            if verbose:
                print('        the image requires reprojection/relocation')

                relocated_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

                print('        reprojecting image to temporary file:',
                      relocated_file)

                if temp_files_list is not None:
                    temp_files_list.append(relocated_file)

                gdal.Warp(relocated_file, path_rtc,
                          format='GTiff',
                          dstSRS=wkt_projection,
                          outputBounds=[
                              geogrid_in.start_x,
                              geogrid_in.start_y +
                                geogrid_in.length * geogrid_in.spacing_y,
                              geogrid_in.start_x +
                                geogrid_in.width * geogrid_in.spacing_x,
                              geogrid_in.start_y],
                          multithread=True,
                          xRes=geogrid_in.spacing_x,
                          yRes=abs(geogrid_in.spacing_y),
                          resampleAlg='average',
                          errorThreshold=0,
                          dstNodata=np.nan)
                path_rtc = relocated_file

                if path_nlooks is not None:
                    relocated_file_nlooks = tempfile.NamedTemporaryFile(
                        dir=scratch_dir, suffix='.tif').name

                    print('        reprojecting number of looks layer to temporary'
                          ' file:', relocated_file_nlooks)

                    if temp_files_list is not None:
                        temp_files_list.append(relocated_file_nlooks)

                    gdal.Warp(relocated_file_nlooks, path_nlooks,
                            format='GTiff',
                            dstSRS=wkt_projection,
                            outputBounds=[
                                geogrid_in.start_x,
                                geogrid_in.start_y +
                                    geogrid_in.length * geogrid_in.spacing_y,
                                geogrid_in.start_x +
                                    geogrid_in.width * geogrid_in.spacing_x,
                                geogrid_in.start_y],
                            multithread=True,
                            xRes=geogrid_in.spacing_x,
                            yRes=abs(geogrid_in.spacing_y),
                            resampleAlg='cubic',
                            errorThreshold=0,
                          dstNodata=np.nan)
                    path_nlooks = relocated_file_nlooks

            offset_imgx = 0
            offset_imgy = 0
        else:

            # calculate the burst RTC's offset wrt. the output mosaic in the image coordinate
            offset_imgx = int((list_geo_transform[i, 0] - xmin_mosaic) / posting_x + 0.5)
            offset_imgy = int((list_geo_transform[i, 3] - ymax_mosaic) / posting_y + 0.5)

        if verbose:
            print(f'        image offset (x, y): ({offset_imgx}, {offset_imgy})')

        if path_nlooks is not None:
            nlooks_gdal_ds = gdal.Open(path_nlooks, gdal.GA_ReadOnly)
            arr_nlooks = nlooks_gdal_ds.ReadAsArray()
            invalid_ind = np.isnan(arr_nlooks)
            arr_nlooks[invalid_ind] = 0.0
        else:
            arr_nlooks = 1

        rtc_image_gdal_ds = gdal.Open(path_rtc, gdal.GA_ReadOnly)

        for i_band in range(num_bands):

            band_ds = rtc_image_gdal_ds.GetRasterBand(i_band + 1)
            arr_rtc = band_ds.ReadAsArray()
            if i_band == 0:
                length, width = arr_rtc.shape

            if mosaic_mode.lower() == 'average':
                # Replace NaN values with 0
                arr_rtc[np.isnan(arr_rtc)] = 0.0

                arr_numerator[i_band,
                            offset_imgy: offset_imgy + length,
                            offset_imgx: offset_imgx + width] += \
                    arr_rtc * arr_nlooks

                if path_nlooks is not None:
                    arr_denominator[offset_imgy: offset_imgy + length,
                                    offset_imgx: offset_imgx + width] += arr_nlooks
                else:
                    arr_denominator[offset_imgy: offset_imgy + length,
                                    offset_imgx: offset_imgx + width] += np.asarray(
                        arr_rtc > 0, dtype=np.byte)

                continue

            arr_temp = arr_numerator[i_band, offset_imgy: offset_imgy + length,
                                     offset_imgx: offset_imgx + width].copy()

            if i_band == 0 and mosaic_mode.lower() == 'first':
                ind = np.where(np.isnan(arr_temp))
            elif i_band == 0 and mosaic_mode.lower() == 'bursts_center':
                geotransform = rtc_image_gdal_ds.GetGeoTransform()

                arr_new_distance = _compute_distance_to_burst_center(
                    arr_rtc, geotransform)

                arr_distance_temp = arr_distance[offset_imgy: offset_imgy + length,
                                                 offset_imgx: offset_imgx + width]
                ind = np.where(np.logical_or(np.isnan(arr_distance_temp),
                                             arr_new_distance <= arr_distance_temp))

                arr_distance_temp[ind] = arr_new_distance[ind]
                arr_distance[offset_imgy: offset_imgy + length,
                             offset_imgx: offset_imgx + width] = arr_distance_temp

                del arr_distance_temp


            arr_temp[ind] = arr_rtc[ind]
            arr_numerator[i_band,
                            offset_imgy: offset_imgy + length,
                            offset_imgx: offset_imgx + width] = arr_temp

        rtc_image_gdal_ds = None
        nlooks_gdal_ds = None
 
    if mosaic_mode.lower() == 'average':
        # Mode: average
        # `arr_numerator` holds the accumulated sum. Now, we divide it
        # by `arr_denominator` to get the average value
        for i_band in range(num_bands):
            valid_ind = arr_denominator > 0
            arr_numerator[i_band][valid_ind] = \
                arr_numerator[i_band][valid_ind] / arr_denominator[valid_ind]

            arr_numerator[i_band][arr_denominator == 0] = np.nan

    mosaic_dict = {
        'mosaic_array': arr_numerator,
        'description_list': description_list,
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


def mosaic_single_output_file(list_rtc_images, list_nlooks, mosaic_filename,
                              mosaic_mode, scratch_dir='', geogrid_in=None,
                              temp_files_list=None, verbose=True):
    '''
    Mosaic the snapped S1 geobursts

    Parameters:
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        mosaic_filename: str
            Path to the output mosaic
        scratch_dir: str (optional)
            Directory for temporary files
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None
        temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
        verbose : bool
            Flag to enable/disable the verbose mode
    '''
    mosaic_dict = compute_mosaic_array(
        list_rtc_images, list_nlooks, mosaic_mode, scratch_dir=scratch_dir,
        geogrid_in=geogrid_in, temp_files_list=temp_files_list,
        verbose=verbose)

    arr_numerator = mosaic_dict['mosaic_array']
    description_list = mosaic_dict['description_list']
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
    raster_out = drv_out.Create(mosaic_filename,
                                width, length, num_bands,
                                datatype_mosaic)

    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))
    raster_out.SetProjection(wkt_projection)

    for i_band in range(num_bands):
        gdal_band = raster_out.GetRasterBand(i_band+1)
        gdal_band.WriteArray(arr_numerator[i_band])
        gdal_band.SetDescription(description_list[i_band])


def mosaic_multiple_output_files(
        list_rtc_images, list_nlooks, output_file_list, mosaic_mode,
        scratch_dir='', geogrid_in=None, temp_files_list=None, verbose=True):
    '''
    Mosaic the snapped S1 geobursts

    Paremeters:
    -----------
        list_rtc_images: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        output_file_list: list
            Output file list
        mosaic_mode: str
            Mosaic mode. Choices: "average", "first", and "bursts_center"
        scratch_dir: str (optional)
            Directory for temporary files
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None
        temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
        verbose : bool
            Flag to enable/disable the verbose mode
            
    '''
    mosaic_dict = compute_mosaic_array(
        list_rtc_images, list_nlooks, mosaic_mode, scratch_dir=scratch_dir,
        geogrid_in=geogrid_in, temp_files_list=temp_files_list,
        verbose = verbose)

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
        nbands = 1
        raster_out = drv_out.Create(output_file,
                                    width, length, nbands,
                                    datatype_mosaic)

        raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))

        raster_out.SetProjection(wkt_projection)

        # for i_band in range(num_bands):
        raster_out.GetRasterBand(1).WriteArray(arr_numerator[i_band])
