'''
RTC-S1 Science Application Software (single job)
'''

import os
import time
import tempfile

import logging
import numpy as np
from osgeo import gdal
import argparse
from datetime import datetime

import isce3
from scipy import ndimage
import matplotlib.image as mpimg

from s1reader.s1_burst_slc import Sentinel1BurstSlc

from rtc.geogrid import snap_coord
from rtc.runconfig import RunConfig, STATIC_LAYERS_PRODUCT_TYPE
from rtc.mosaic_geobursts import (mosaic_single_output_file,
                                  mosaic_multiple_output_files)
from rtc.core import (save_as_cog, check_ancillary_inputs,
                      build_empty_vrt)
from rtc.h5_prep import (save_hdf5_file, create_hdf5_file,
                         get_metadata_dict,
                         all_metadata_dict_to_geotiff_metadata_dict,
                         layer_names_dict,
                         layer_description_dict,
                         DATA_BASE_GROUP,
                         DATE_TIME_FILENAME_FORMAT,
                         LAYER_NAME_LAYOVER_SHADOW_MASK,
                         LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0,
                         LAYER_NAME_NUMBER_OF_LOOKS,
                         LAYER_NAME_INCIDENCE_ANGLE,
                         LAYER_NAME_LOCAL_INCIDENCE_ANGLE,
                         LAYER_NAME_PROJECTION_ANGLE,
                         LAYER_NAME_RTC_ANF_PROJECTION_ANGLE,
                         LAYER_NAME_RANGE_SLOPE,
                         LAYER_NAME_DEM)
from rtc.version import VERSION as SOFTWARE_VERSION

logger = logging.getLogger('rtc_s1')

STATIC_LAYERS_LAYOVER_SHADOW_MASK_MULTILOOK_FACTOR = 3

STATIC_LAYERS_AZ_MARGIN = 1.2
STATIC_LAYERS_RG_MARGIN = 0.2


def populate_product_id(product_id, burst_in, processing_datetime,
                        product_version, pixel_spacing, product_type,
                        rtc_s1_static_validity_start_date, is_mosaic):
    '''
    Populate product_id string with S1/RTC-S1 parameters

    Parameters
    ----------
    product_id: str
        Input product ID string
    burst_in: Sentinel1BurstSlc
        Input burst SLC
    processing_datetime: datetime
        Processing start datetime
    product_version: scalar
        Product version
    pixel_spacing: scalar
        Pixel spacing
    product_type: string
        Product type
    rtc_s1_static_validity_start_date: int
        Validity start date (only applicable for the RTC-S1-STATIC product)
        in the format YYYYMMDD
    is_mosaic: bool
        Flag indicating whether the product ID refers to a mosaic or a
        burst product


    Returns
    -------
    _: str
        Product ID populated with S1/RTC-S1 parameters
    '''

    if product_id is None:
        product_id = '{product_id}'

    if ('{product_id}' in product_id and
            product_type != STATIC_LAYERS_PRODUCT_TYPE):
        product_id = ('OPERA_L2_RTC-S1_{burst_id}_{sensing_start_datetime}'
                      '_{processing_datetime}_{sensor}_{pixel_spacing}'
                      '_{product_version}')
    if '{product_id}' in product_id:
        if not rtc_s1_static_validity_start_date:
            error_msg = ('ERROR please provide a' +
                         ' `rtc_s1_static_validity_start_date`')
            raise ValueError(error_msg)
        product_id = ('OPERA_L2_RTC-S1-STATIC_{burst_id}'
                      f'_{rtc_s1_static_validity_start_date}'
                      '_{processing_datetime}_{sensor}_{pixel_spacing}'
                      '_{product_version}')

    # Populate product_id sensing_start_datetime
    sensing_start_datetime = burst_in.sensing_start.strftime(
        DATE_TIME_FILENAME_FORMAT)
    product_id = product_id.replace('{sensing_start_datetime}',
                                    sensing_start_datetime)

    # Populate product_id processing_datetime
    processing_datetime_filename = processing_datetime.strftime(
        DATE_TIME_FILENAME_FORMAT)
    product_id = product_id.replace(
        '{processing_datetime}', processing_datetime_filename)

    # Populate product_id sensor
    product_id = product_id.replace('{sensor}', burst_in.platform_id)

    # Populate product_id pixel_spacing
    product_id = product_id.replace('{pixel_spacing}', f'{pixel_spacing}')

    # Populate product_id version
    product_id = product_id.replace('{product_version}', f'v{product_version}')

    if not is_mosaic:
        burst_id_file_name = burst_in.burst_id.upper().replace('_', '-')
        product_id = product_id.replace('{burst_id}', f'T{burst_id_file_name}')
    else:
        product_id = product_id.replace('_{burst_id}', '')

    return product_id


def compute_correction_lut(burst_in, dem_raster, scratch_path,
                           rg_step_meters=120,
                           az_step_meters=120):
    '''
    Compute lookup table for geolocation correction.
    Applied corrections are: bistatic delay (azimuth),
                             static troposphere delay (range)

    Parameters
    ----------
    burst_in: Sentinel1BurstSlc
        Input burst SLC
    dem_raster: isce3.io.raster
        DEM to run rdr2geo
    scratch_path: str
        Scratch path where the radargrid rasters will be saved
    rg_step_meters: float
        LUT spacing in slant range. Unit: meters
    az_step_meters: float
        LUT spacing in azimth direction. Unit: meters

    Returns
    -------
    rg_lut, az_lut: isce3.core.LUT2d
        LUT2d for geolocation correction in slant range and azimuth direction
    '''

    # approximate conversion of az_step_meters from meters to seconds
    numrow_orbit = burst_in.orbit.position.shape[0]
    vel_mid = burst_in.orbit.velocity[numrow_orbit // 2, :]
    spd_mid = np.linalg.norm(vel_mid)
    pos_mid = burst_in.orbit.position[numrow_orbit // 2, :]
    alt_mid = np.linalg.norm(pos_mid)

    r = 6371000.0  # geometric mean of WGS84 ellipsoid

    az_step_sec = (az_step_meters * alt_mid) / (spd_mid * r)
    # Bistatic - azimuth direction
    bistatic_delay = burst_in.bistatic_delay(range_step=rg_step_meters,
                                             az_step=az_step_sec)

    # Calculate rdr2geo rasters
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    rdr_grid = burst_in.as_isce3_radargrid(az_step=az_step_sec,
                                           rg_step=rg_step_meters)

    grid_doppler = isce3.core.LUT2d()

    # Initialize the rdr2geo object
    rdr2geo_obj = isce3.geometry.Rdr2Geo(rdr_grid, burst_in.orbit,
                                         ellipsoid, grid_doppler,
                                         threshold=1.0e-8)

    # Get the rdr2geo raster needed for SET computation
    topo_output = {f'{scratch_path}/height.rdr': gdal.GDT_Float32,
                   f'{scratch_path}/incidence_angle.rdr': gdal.GDT_Float32}

    raster_list = []
    for fname, dtype in topo_output.items():
        topo_output_raster = isce3.io.Raster(fname,
                                             rdr_grid.width, rdr_grid.length,
                                             1, dtype, 'ENVI')
        raster_list.append(topo_output_raster)

    height_raster, incidence_raster = raster_list

    rdr2geo_obj.topo(dem_raster, x_raster=None, y_raster=None,
                     height_raster=height_raster,
                     incidence_angle_raster=incidence_raster)

    height_raster.close_dataset()
    incidence_raster.close_dataset()

    # Load height and incidence angle layers
    height_arr =\
        gdal.Open(f'{scratch_path}/height.rdr', gdal.GA_ReadOnly).ReadAsArray()
    incidence_angle_arr =\
        gdal.Open(f'{scratch_path}/incidence_angle.rdr',
                  gdal.GA_ReadOnly).ReadAsArray()

    # static troposphere delay - range direction
    # reference:
    # Breit et al., 2010, TerraSAR-X SAR Processing and Products,
    # IEEE Transactions on Geoscience and Remote Sensing, 48(2), 727-740.
    # DOI: 10.1109/TGRS.2009.2035497
    zenith_path_delay = 2.3
    reference_height = 6000.0
    tropo = (zenith_path_delay
             / np.cos(np.deg2rad(incidence_angle_arr))
             * np.exp(-1 * height_arr / reference_height))

    # Prepare the computation results into LUT2d
    az_lut = isce3.core.LUT2d(bistatic_delay.x_start,
                              bistatic_delay.y_start,
                              bistatic_delay.x_spacing,
                              bistatic_delay.y_spacing,
                              -bistatic_delay.data)
    
    rg_lut = isce3.core.LUT2d(bistatic_delay.x_start,
                              bistatic_delay.y_start,
                              bistatic_delay.x_spacing,
                              bistatic_delay.y_spacing,
                              tropo)

    return rg_lut, az_lut


def save_browse(imagery_list, browse_image_filename,
                pol_list, browse_image_height, browse_image_width,
                temp_files_list, scratch_dir, logger):
    """Create and save a browse image for the RTC-S1 product

       Parameters
       ----------
       imagery_list : list(str)
           List of imagery files (one file for each polarization channel)
       browse_image_filename : str
           Output browse file
       pol_list : list(str)
           List of polarization channels
       browse_image_height : int
           Browse image height
       browse_image_width : int
           Browse image width
       scratch_dir : str
           Directory for temporary files
       temp_files_list: list (optional)
           Mutable list of temporary files. If provided,
           paths to the temporary files generated will be
           appended to this list.
       logger : loggin.Logger
           Logger
    """

    BROWSE_IMAGE_MIN_PERCENTILE = 3
    BROWSE_IMAGE_MAX_PERCENTILE = 97

    logger.info(f'creating browse image: {browse_image_filename}')

    n_images = len(imagery_list)

    if n_images == 1:
        expected_pol_order = pol_list
    elif n_images == 2 and 'HH' in pol_list:
        expected_pol_order = ['HH', 'HV']
    elif n_images == 2:
        expected_pol_order = ['VV', 'VH']
    elif n_images == 3 or n_images == 4:
        expected_pol_order = ['HH', 'HV', 'VV']
    else:
        raise ValueError('Unexpected number of images in the imagery'
                         f' list {n_images} for generating browse'
                         'images')

    alpha_channel = None
    band_list = [None] * n_images

    for filename, pol in zip(imagery_list, pol_list):
        logger.info(f'    pol: {pol}')
        gdal_ds = gdal.Open(filename, gdal.GA_ReadOnly)
        image_width = gdal_ds.GetRasterBand(1).XSize
        image_height = gdal_ds.GetRasterBand(1).YSize

        if (browse_image_height is not None or
                browse_image_width is not None):

            del gdal_ds

            if browse_image_width is None:
                browse_image_width = int(np.round(
                    (browse_image_height * float(image_width) / image_height)))

            if browse_image_height is None:
                browse_image_height = int(np.round(
                    (browse_image_width * float(image_height) / image_width)))

            logger.info(f'        browse length: {browse_image_height}')
            logger.info(f'        browse width: {browse_image_width}')

            browse_temp_file = tempfile.NamedTemporaryFile(
                dir=scratch_dir, suffix='.tif').name

            if temp_files_list is not None:
                temp_files_list.append(browse_temp_file)

            resamp_algorithm = 'AVERAGE'

            # Translate the existing geotiff to the .png format
            gdal.Translate(browse_temp_file,
                           filename,
                           # outputSRS="+proj=longlat +ellps=WGS84",
                           # format='PNG',
                           height=browse_image_height,
                           width=browse_image_width,
                           resampleAlg=resamp_algorithm)

            gdal_ds = gdal.Open(browse_temp_file, gdal.GA_ReadOnly)

        gdal_band = gdal_ds.GetRasterBand(1)
        band_image = np.asarray(gdal_band.ReadAsArray(), dtype=np.float32)
        is_valid = np.isfinite(band_image)
        if alpha_channel is None:
            alpha_channel = np.asarray(is_valid,
                                       dtype=np.float32)
        vmin = np.nanpercentile(band_image, BROWSE_IMAGE_MIN_PERCENTILE)
        vmax = np.nanpercentile(band_image, BROWSE_IMAGE_MAX_PERCENTILE)
        logger.info(f'        min ({BROWSE_IMAGE_MIN_PERCENTILE}% percentile):'
                    f' {vmin}')
        logger.info(f'        max ({BROWSE_IMAGE_MAX_PERCENTILE}% percentile):'
                    f' {vmax}')

        # gamma correction: 0.5
        is_positive = np.logical_and(is_valid, band_image - vmin > 0)
        band_image[is_positive] = np.sqrt((band_image[is_positive] - vmin) /
                                          (vmax - vmin))
        band_image = np.clip(band_image, 0, 1)
        band_list_index = expected_pol_order.index(pol)
        band_list[band_list_index] = band_image

    if n_images == 1:
        image = np.dstack((band_list[0],
                           band_list[0],
                           band_list[0],
                           alpha_channel))
    elif n_images == 2:
        image = np.dstack((band_list[0],
                           band_list[1],
                           band_list[0],
                           alpha_channel))
    else:
        image = np.dstack((band_list[0],
                           band_list[1],
                           band_list[2],
                           alpha_channel))
    mpimg.imsave(browse_image_filename, image, format='png')
    logger.info(f'file saved: {browse_image_filename}')


def append_metadata_to_geotiff_file(input_file, metadata_dict, product_id):
    '''Append metadata to GeoTIFF file

       Parameters
       ----------
       input_file : str
           Input GeoTIFF file
       metadata_dict : dict
           Metadata dictionary
       product_id : str
           Product ID
    '''
    input_file_basename = os.path.basename(input_file)
    logger.info('    appending metadata to the GeoTIFF file:'
                f' {input_file_basename}')
    gdal_ds = gdal.Open(input_file, gdal.GA_Update)
    existing_metadata = gdal_ds.GetMetadata()

    # Update existing metadata with RTC-S1 metadata
    existing_metadata.update(metadata_dict)
    layer_id = input_file_basename.replace(f'{product_id}_', '').split('.')[0]

    # Update metadata file name
    existing_metadata['FILENAME'] = input_file_basename

    # Update metadata layer name (short description)
    if layer_id in layer_names_dict.keys():
        layer_name = layer_names_dict[layer_id]
        existing_metadata['LAYER_NAME'] = layer_name

        # Save layer name using SetDescription()
        gdal_ds.SetDescription(layer_name)

        band_out = gdal_ds.GetRasterBand(1)
        band_out.SetDescription(layer_name)
        band_out.FlushCache()

        del band_out

    # Update metadata layer description (long description)
    if layer_id in layer_description_dict.keys():
        layer_description = layer_description_dict[layer_id]
        existing_metadata['LAYER_DESCRIPTION'] = layer_description

    # Write metadata
    gdal_ds.SetMetadata(existing_metadata)

    # Check NoDataValue
    for band in range(gdal_ds.RasterCount):
        band_ds = gdal_ds.GetRasterBand(band + 1)
        dtype = band_ds.DataType
        dtype_name = gdal.GetDataTypeName(dtype)
        if ('float' in dtype_name.lower() and
                band_ds.GetNoDataValue() is None):
            band_ds.SetNoDataValue(np.nan)
        del band_ds

    # Close GDAL dataset
    del gdal_ds


def _separate_pol_channels(multi_band_file, output_file_list,
                           output_raster_format, logger):
    """Save a multi-band raster file as individual single-band files

       Parameters
       ----------
       multi_band_file : str
           Multi-band raster file
       output_file_list : list(str)
           Output file list
       output_raster_format : str
           Output raster format
       logger : loggin.Logger
    """
    gdal_ds = gdal.Open(multi_band_file, gdal.GA_ReadOnly)
    projection = gdal_ds.GetProjectionRef()
    geotransform = gdal_ds.GetGeoTransform()

    num_bands = gdal_ds.RasterCount
    if num_bands != len(output_file_list):
        error_str = (f'ERROR number of output files ({len(output_file_list)})'
                     f' does not match with the number'
                     f' of input bursts` bands ({num_bands})')
        raise ValueError(error_str)

    for b, output_file in enumerate(output_file_list):
        gdal_band = gdal_ds.GetRasterBand(b + 1)
        gdal_dtype = gdal_band.DataType
        band_image = gdal_band.ReadAsArray()

        # Save the corrected image
        driver_out = gdal.GetDriverByName(output_raster_format)
        raster_out = driver_out.Create(
            output_file, band_image.shape[1],
            band_image.shape[0], 1, gdal_dtype)

        raster_out.SetProjection(projection)
        raster_out.SetGeoTransform(geotransform)

        band_out = raster_out.GetRasterBand(1)
        band_out.WriteArray(band_image)
        band_out.FlushCache()
        del band_out
        logger.info(f'file saved: {output_file}')


def _create_raster_obj(output_dir, product_id, layer_name, dtype, shape,
                       radar_grid_file_dict, output_obj_list,
                       flag_create_raster_obj, extension):
    """Create an ISCE3 raster object (GTiff) for a radar geometry layer.

       Parameters
       ----------
       output_dir: str
              Output directory
       layer_name: str
              Layer name
       product_id: str
              Product ID
       ds_hdf5: str
              HDF5 dataset name
       dtype:: gdal.DataType
              GDAL data type
       shape: list
              Shape of the output raster
       radar_grid_file_dict: dict
              Dictionary that will hold the name of the output file
              referenced by the contents of `ds_hdf5` (dict key)
       output_obj_list: list
              Mutable list of output raster objects
       flag_create_raster_obj: bool
              Flag indicating if raster object should be created

       Returns
       -------
       raster_obj : isce3.io.Raster
              ISCE3 raster object
    """
    if flag_create_raster_obj is not True:
        return None

    ds_name = f'{product_id}_{layer_name}'

    output_file = os.path.join(output_dir, ds_name) + '.' + extension
    raster_obj = isce3.io.Raster(
        output_file,
        shape[2],
        shape[1],
        shape[0],
        dtype,
        "GTiff")
    output_obj_list.append(raster_obj)
    radar_grid_file_dict[layer_name] = output_file
    return raster_obj


def add_output_to_output_metadata_dict(flag, key, output_dir, 
                                       output_metadata_dict, product_id,
                                       extension):
    if not flag:
        return
    output_image_list = []
    output_metadata_dict[key] = \
        [os.path.join(output_dir, f'{product_id}_{key}.{extension}'),
         output_image_list]


def apply_slc_corrections(burst_in: Sentinel1BurstSlc,
                          path_slc_vrt: str,
                          path_slc_out: str,
                          flag_output_complex: bool = False,
                          flag_thermal_correction: bool = True,
                          flag_apply_abs_rad_correction: bool = True):
    '''Apply thermal correction stored in burst_in. Save the corrected signal
    back to ENVI format. Preserves the phase when the output is complex

    Parameters
    ----------
    burst_in: Sentinel1BurstSlc
        Input burst to apply the correction
    path_slc_vrt: str
        Path to the input burst to apply correction
    path_slc_out: str
        Path to the output SLC which the corrections are applied
    flag_output_complex: bool
        `path_slc_out` will be in complex number when this is `True`
        Otherwise, the output will be amplitude only.
    flag_thermal_correction: bool
        flag whether or not to apple the thermal correction.
    flag_apply_abs_rad_correction: bool
        Flag to apply radiometric calibration
    '''

    # Load the SLC of the burst
    burst_in.slc_to_vrt_file(path_slc_vrt)
    slc_gdal_ds = gdal.Open(path_slc_vrt)
    arr_slc_from = slc_gdal_ds.ReadAsArray()

    # Apply thermal noise correction
    if flag_thermal_correction:
        logger.info('    applying thermal noise correction to burst SLC')
        corrected_image = (np.abs(arr_slc_from) ** 2 -
                           burst_in.thermal_noise_lut)
        min_backscatter = 0
        max_backscatter = None
        corrected_image = np.clip(corrected_image, min_backscatter,
                                  max_backscatter)
    else:
        corrected_image = np.abs(arr_slc_from) ** 2

    # Apply absolute radiometric correction
    if flag_apply_abs_rad_correction:
        logger.info('    applying absolute radiometric correction to burst'
                    ' SLC')
        corrected_image = \
            corrected_image / burst_in.burst_calibration.beta_naught ** 2

    # Output as complex
    if flag_output_complex:
        factor_mag = np.sqrt(corrected_image) / np.abs(arr_slc_from)
        factor_mag[np.isnan(factor_mag)] = 0.0
        corrected_image = arr_slc_from * factor_mag
        dtype = gdal.GDT_CFloat32
    else:
        dtype = gdal.GDT_Float32

    # Save the corrected image
    drvout = gdal.GetDriverByName('GTiff')
    raster_out = drvout.Create(path_slc_out, burst_in.shape[1],
                               burst_in.shape[0], 1, dtype)
    band_out = raster_out.GetRasterBand(1)
    band_out.WriteArray(corrected_image)
    band_out.FlushCache()
    del band_out


def _test_valid_gdal_ref(gdal_ref):
    '''
    Test if the input string contains a valid GDAL reference.

    Parameters
    -----------
    gdal_ref: str
        Input string

    Returns
    -------
    _ : bool
        Boolean value indicating if the input string is a valid GDAL reference 
    '''
    try:
        gdal_ds = gdal.Open(gdal_ref, gdal.GA_ReadOnly)
    except:
        return False
    return gdal_ds is not None


def set_mask_fill_value_and_ctable(mask_file, reference_file):
    '''
    Update color table and fill values of the layover shadow mask using
    another file as reference for invalid samples

    Parameters
    -----------
    mask_file: str
        Layover/shadow mask file
    reference_file: str
        File to be used as reference for invalid samples

    '''
    logger.info('    updating layover/shadow mask with fill value and color'
                ' table')
    ref_gdal_ds = gdal.Open(reference_file, gdal.GA_ReadOnly)
    ref_gdal_band = ref_gdal_ds.GetRasterBand(1)
    ref_array = ref_gdal_band.ReadAsArray()

    mask_gdal_ds = gdal.Open(mask_file, gdal.GA_Update)
    mask_ctable = gdal.ColorTable()

    # Light gray - Not masked
    mask_ctable.SetColorEntry(0, (175, 175, 175))

    # Shadow - Dark gray
    mask_ctable.SetColorEntry(1, (64, 64, 64))

    # White - Layover
    mask_ctable.SetColorEntry(2, (255, 255, 255))

    # Cyan - Layover and shadow
    mask_ctable.SetColorEntry(3, (0, 255, 255))

    # No data
    mask_gdal_band = mask_gdal_ds.GetRasterBand(1)
    mask_array = mask_gdal_band.ReadAsArray()
    mask_array[(np.isnan(ref_array)) & (mask_array == 0)] = 255
    mask_gdal_band.WriteArray(mask_array)
    mask_gdal_band.SetNoDataValue(255)

    mask_ctable.SetColorEntry(255, (0, 0, 0, 0))
    mask_gdal_band.SetRasterColorTable(mask_ctable)
    mask_gdal_band.SetRasterColorInterpretation(
        gdal.GCI_PaletteIndex)

    del mask_gdal_band
    del mask_gdal_ds


def compute_layover_shadow_mask(radar_grid: isce3.product.RadarGridParameters,
                                orbit: isce3.core.Orbit,
                                geogrid_in: isce3.product.GeoGridParameters,
                                burst_in: Sentinel1BurstSlc,
                                dem_raster: isce3.io.Raster,
                                filename_out: str,
                                output_raster_format: str,
                                scratch_dir: str,
                                shadow_dilation_size: int,
                                threshold_rdr2geo: float = 1.0e-7,
                                numiter_rdr2geo: int = 25,
                                extraiter_rdr2geo: int = 10,
                                lines_per_block_rdr2geo: int = 1000,
                                threshold_geo2rdr: float = 1.0e-7,
                                numiter_geo2rdr: int = 25,
                                memory_mode: isce3.core.GeocodeMemoryMode =
                                None,
                                geocode_options=None):
    '''
    Compute the layover/shadow mask and geocode it

    Parameters
    -----------
    radar_grid: isce3.product.RadarGridParameters
        Radar grid
    orbit: isce3.core.Orbit
        Orbit defining radar motion on input path
    geogrid_in: isce3.product.GeoGridParameters
        Geogrid to geocode the layover/shadow mask in radar grid
    burst_in: Sentinel1BurstSlc
        Input burst
    geogrid_in: isce3.product.GeoGridParameters
        Geogrid to geocode the layover/shadow mask in radar grid
    dem_raster: isce3.io.Raster
        DEM raster
    filename_out: str
        Path to the geocoded layover/shadow mask
    output_raster_format: str
        File format of the layover/shadow mask
    scratch_dir: str
        Temporary Directory
    shadow_dilation_size: int
        Layover/shadow mask dilation size of shadow pixels
    threshold_rdr2geo: float
        Iteration threshold for rdr2geo
    numiter_rdr2geo: int
        Number of max. iteration for rdr2geo object
    extraiter_rdr2geo: int
        Extra number of iteration for rdr2geo object
    lines_per_block_rdr2geo: int
        Lines per block for rdr2geo
    threshold_geo2rdr: float
        Iteration threshold for geo2rdr
    numiter_geo2rdr: int
        Number of max. iteration for geo2rdr object
    memory_mode: isce3.core.GeocodeMemoryMode
        Geocoding memory mode
    geocode_options: dict
        Keyword arguments to be passed to the geocode() function
        when map projection the layover/shadow mask

    Returns
    -------
    slantrange_layover_shadow_mask_raster: isce3.io.Raster
        Layover/shadow-mask ISCE3 raster object in radar coordinates
    '''

    # determine the output filename
    str_datetime = burst_in.sensing_start.strftime('%Y%m%d_%H%M%S.%f')

    # Run topo to get layover/shadow mask
    ellipsoid = isce3.core.Ellipsoid()

    Rdr2Geo = isce3.geometry.Rdr2Geo

    grid_doppler = isce3.core.LUT2d()

    rdr2geo_obj = Rdr2Geo(radar_grid,
                          orbit,
                          ellipsoid,
                          grid_doppler,
                          threshold=threshold_rdr2geo,
                          numiter=numiter_rdr2geo,
                          extraiter=extraiter_rdr2geo,
                          lines_per_block=lines_per_block_rdr2geo)

    if shadow_dilation_size > 0:
        path_layover_shadow_mask_file = os.path.join(
            scratch_dir, 'layover_shadow_mask_slant_range.tif')
        slantrange_layover_shadow_mask_raster = isce3.io.Raster(
            path_layover_shadow_mask_file, radar_grid.width, radar_grid.length,
            1, gdal.GDT_Byte, 'GTiff')
    else:
        path_layover_shadow_mask = (f'layover_shadow_mask_{burst_in.burst_id}_'
                                    f'{burst_in.polarization}_{str_datetime}')
        slantrange_layover_shadow_mask_raster = isce3.io.Raster(
            path_layover_shadow_mask, radar_grid.width, radar_grid.length,
            1, gdal.GDT_Byte, 'MEM')

    rdr2geo_obj.topo(
        dem_raster,
        layover_shadow_raster=slantrange_layover_shadow_mask_raster)

    if shadow_dilation_size > 1:
        '''
        constants from ISCE3:
            SHADOW_VALUE = 1;
            LAYOVER_VALUE = 2;
            LAYOVER_AND_SHADOW_VALUE = 3;
        We only want to dilate values 1 and 3
        '''

        # flush raster data to the disk
        slantrange_layover_shadow_mask_raster.close_dataset()
        del slantrange_layover_shadow_mask_raster

        # read layover/shadow mask
        gdal_ds = gdal.Open(path_layover_shadow_mask_file,
                            gdal.GA_Update)
        gdal_band = gdal_ds.GetRasterBand(1)
        slantrange_layover_shadow_mask = gdal_band.ReadAsArray()

        # save layover pixels and substitute them with 0
        ind = np.where(slantrange_layover_shadow_mask == 2)
        slantrange_layover_shadow_mask[ind] = 0

        # perform grey dilation
        slantrange_layover_shadow_mask = \
            ndimage.grey_dilation(slantrange_layover_shadow_mask,
                                  size=(shadow_dilation_size,
                                        shadow_dilation_size))

        # restore layover pixels
        slantrange_layover_shadow_mask[ind] = 2

        # write dilated layover/shadow mask
        gdal_band.WriteArray(slantrange_layover_shadow_mask)

        # flush updates to the disk
        gdal_band.FlushCache()
        gdal_band = None
        gdal_ds = None

        slantrange_layover_shadow_mask_raster = isce3.io.Raster(
            path_layover_shadow_mask_file)

    # geocode the layover/shadow mask
    geo = isce3.geocode.GeocodeFloat32()
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = isce3.core.LUT2d()
    geo.threshold_geo2rdr = threshold_geo2rdr
    geo.numiter_geo2rdr = numiter_geo2rdr
    geo.data_interpolator = 'NEAREST'
    geo.geogrid(float(geogrid_in.start_x),
                float(geogrid_in.start_y),
                float(geogrid_in.spacing_x),
                float(geogrid_in.spacing_y),
                int(geogrid_in.width),
                int(geogrid_in.length),
                int(geogrid_in.epsg))

    geocoded_layover_shadow_mask_raster = isce3.io.Raster(
        filename_out, geogrid_in.width, geogrid_in.length, 1,
        gdal.GDT_Byte, output_raster_format)

    if geocode_options is None:
        geocode_options = {}

    if memory_mode is not None:
        geocode_options['memory_mode'] = memory_mode

    geo.geocode(radar_grid=radar_grid,
                input_raster=slantrange_layover_shadow_mask_raster,
                output_raster=geocoded_layover_shadow_mask_raster,
                dem_raster=dem_raster,
                output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
                **geocode_options)

    # flush data to the disk
    geocoded_layover_shadow_mask_raster.close_dataset()

    return slantrange_layover_shadow_mask_raster


def read_and_validate_rtc_anf_flags(geocode_namespace, flag_apply_rtc,
                                    output_terrain_radiometry, logger):
    '''
    Read and validate radiometric terrain correction (RTC) area
    normalization factor (ANF) flags

    Parameters
    ----------
    geocode_namespace: types.SimpleNamespace
        Runconfig geocode namespace
    flag_apply_rtc: Bool
        Flag apply RTC (radiometric terrain correction)
    output_terrain_radiometry: isce3.geometry.RtcOutputTerrainRadiometry
        Output terrain radiometry (backscatter coefficient convention)
    logger : loggin.Logger
        Logger

    Returns
    -------
    save_rtc_anf: bool
        Flag indicating wheter the radiometric terrain correction (RTC)
        area normalization factor (ANF) layer should be created
    save_rtc_anf_gamma0_to_sigma0: bool
        Flag indicating wheter the radiometric terrain correction (RTC)
        area normalization factor (ANF) gamma0 to sigma0 layer should be
        created
    '''
    save_rtc_anf = geocode_namespace.save_rtc_anf
    save_rtc_anf_gamma0_to_sigma0 = \
        geocode_namespace.save_rtc_anf_gamma0_to_sigma0

    if not flag_apply_rtc and save_rtc_anf:
        logger.warning("WARNING the option `save_rtc_anf` is not available"
                       " with radiometric terrain correction"
                       " disabled (`apply_rtc = False`). Setting"
                       " flag `save_rtc_anf` to `False`.")
        save_rtc_anf = False

    if not flag_apply_rtc and save_rtc_anf_gamma0_to_sigma0:
        logger.warning = ("WARNING the option `save_rtc_anf_gamma0_to_sigma0`"
                          " is not available with radiometric terrain"
                          " correction disabled (`apply_rtc = False`)."
                          " Setting flag `save_rtc_anf_gamma0_to_sigma0` to"
                          " `False`.")
        save_rtc_anf_gamma0_to_sigma0 = False
    elif (save_rtc_anf_gamma0_to_sigma0 and
          output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT):
        logger.warning = ("WARNING the option `save_rtc_anf_gamma0_to_sigma0`"
                          " is not available with output radiometric terrain"
                          " radiometry (`output_type`) set to `sigma0`."
                          " Setting flag `save_rtc_anf_gamma0_to_sigma0` to"
                          " `False`.")
        save_rtc_anf_gamma0_to_sigma0 = False

    return save_rtc_anf, save_rtc_anf_gamma0_to_sigma0


def run_single_job(cfg: RunConfig):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig `cfg`

    Parameters
    ---------
    cfg: RunConfig
        RunConfig object with user runconfig options
    '''

    # Start tracking processing time
    t_start = time.time()
    time_stamp = str(float(time.time()))
    logger.info('OPERA RTC-S1 Science Application Software (SAS)'
                f' v{SOFTWARE_VERSION}')

    # primary executable
    product_type = cfg.groups.primary_executable.product_type
    product_version_float = cfg.groups.product_group.product_version
    rtc_s1_static_validity_start_date = \
        cfg.groups.product_group.rtc_s1_static_validity_start_date
    if product_version_float is None:
        product_version = SOFTWARE_VERSION
    else:
        product_version = f'{product_version_float:.1f}'

    # unpack processing parameters
    processing_namespace = cfg.groups.processing
    dem_interp_method_enum = \
        processing_namespace.dem_interpolation_method_enum
    flag_apply_rtc = processing_namespace.apply_rtc
    flag_apply_thermal_noise_correction = \
        processing_namespace.apply_thermal_noise_correction
    flag_apply_abs_rad_correction = \
        processing_namespace.apply_absolute_radiometric_correction
    check_ancillary_inputs_coverage = \
        processing_namespace.check_ancillary_inputs_coverage

    # read product path group / output format
    runconfig_product_id = cfg.groups.product_group.product_id

    # set processing_datetime
    processing_datetime = datetime.now()

    # get mosaic_product_id
    burst_id = next(iter(cfg.bursts))
    burst_pol_dict = cfg.bursts[burst_id]
    pol_list = list(burst_pol_dict.keys())
    burst_ref = burst_pol_dict[pol_list[0]]
    pixel_spacing_avg = int((cfg.geogrid.spacing_x + cfg.geogrid.spacing_y) /
                            2)
    mosaic_product_id = populate_product_id(
        runconfig_product_id, burst_ref, processing_datetime, product_version,
        pixel_spacing_avg, product_type, rtc_s1_static_validity_start_date, is_mosaic=True)

    scratch_path = os.path.join(
        cfg.groups.product_group.scratch_path, f'temp_{time_stamp}')
    output_dir = cfg.groups.product_group.output_dir

    # populate processing parameters
    save_bursts = cfg.groups.product_group.save_bursts
    save_mosaics = cfg.groups.product_group.save_mosaics
    flag_save_browse = cfg.groups.product_group.save_browse

    if not save_bursts and not save_mosaics:
        err_msg = ("ERROR either `save_bursts` or `save_mosaics` needs to be"
                   " set")
        raise ValueError(err_msg)

    output_imagery_format = \
        cfg.groups.product_group.output_imagery_format
    output_imagery_compression = \
        cfg.groups.product_group.output_imagery_compression
    output_imagery_nbits = \
        cfg.groups.product_group.output_imagery_nbits

    browse_image_burst_height = \
        cfg.groups.processing.browse_image_group.browse_image_burst_height
    browse_image_burst_width = \
        cfg.groups.processing.browse_image_group.browse_image_burst_width
    browse_image_mosaic_height = \
        cfg.groups.processing.browse_image_group.browse_image_mosaic_height
    browse_image_mosaic_width = \
        cfg.groups.processing.browse_image_group.browse_image_mosaic_width

    save_imagery_as_hdf5 = (output_imagery_format == 'HDF5' or
                            output_imagery_format == 'NETCDF')
    save_secondary_layers_as_hdf5 = \
        cfg.groups.product_group.save_secondary_layers_as_hdf5

    save_hdf5_metadata = (cfg.groups.product_group.save_metadata or
                          save_imagery_as_hdf5 or
                          save_secondary_layers_as_hdf5)

    if output_imagery_format == 'NETCDF':
        hdf5_file_extension = 'nc'
    else:
        hdf5_file_extension = 'h5'

    if save_imagery_as_hdf5 or output_imagery_format == 'COG':
        output_raster_format = 'GTiff'
    else:
        output_raster_format = output_imagery_format

    if output_raster_format == 'GTiff':
        imagery_extension = 'tif'
    else:
        imagery_extension = 'bin'

    # unpack geocode run parameters
    geocode_namespace = cfg.groups.processing.geocoding

    apply_valid_samples_sub_swath_masking = \
        cfg.groups.processing.geocoding.apply_valid_samples_sub_swath_masking
    apply_shadow_masking = \
        cfg.groups.processing.geocoding.apply_shadow_masking

    skip_if_output_files_exist = \
        cfg.groups.processing.geocoding.skip_if_output_files_exist

    if cfg.groups.processing.geocoding.algorithm_type == "area_projection":
        geocode_algorithm = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION
    else:
        geocode_algorithm = isce3.geocode.GeocodeOutputMode.INTERP

    az_step_meters = \
        cfg.groups.processing.correction_lut_azimuth_spacing_in_meters
    rg_step_meters = \
        cfg.groups.processing.correction_lut_range_spacing_in_meters

    memory_mode = geocode_namespace.memory_mode
    geogrid_upsampling = geocode_namespace.geogrid_upsampling
    shadow_dilation_size = geocode_namespace.shadow_dilation_size
    abs_cal_factor = geocode_namespace.abs_rad_cal
    clip_max = geocode_namespace.clip_max
    clip_min = geocode_namespace.clip_min
    flag_upsample_radar_grid = geocode_namespace.upsample_radargrid
    save_incidence_angle = geocode_namespace.save_incidence_angle
    save_local_inc_angle = geocode_namespace.save_local_inc_angle
    save_projection_angle = geocode_namespace.save_projection_angle
    save_rtc_anf_projection_angle = \
        geocode_namespace.save_rtc_anf_projection_angle
    save_range_slope = geocode_namespace.save_range_slope
    save_nlooks = geocode_namespace.save_nlooks

    save_dem = geocode_namespace.save_dem
    save_layover_shadow_mask = geocode_namespace.save_layover_shadow_mask

    # unpack mosaicking run parameters
    mosaicking_namespace = cfg.groups.processing.mosaicking
    mosaic_mode = mosaicking_namespace.mosaic_mode

    flag_call_radar_grid = (save_incidence_angle or
                            save_local_inc_angle or save_projection_angle or
                            save_rtc_anf_projection_angle or save_dem or
                            save_range_slope)

    # unpack RTC run parameters
    rtc_namespace = cfg.groups.processing.rtc

    # only 2 RTC algorithms supported: area_projection (default) &
    # bilinear_distribution
    if rtc_namespace.algorithm_type == "bilinear_distribution":
        rtc_algorithm = isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION
    else:
        rtc_algorithm = isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION

    input_terrain_radiometry = rtc_namespace.input_terrain_radiometry
    input_terrain_radiometry_enum = rtc_namespace.input_terrain_radiometry_enum
    output_terrain_radiometry = rtc_namespace.output_type
    output_terrain_radiometry_enum = rtc_namespace.output_type_enum
    output_radiometry_str = f"radar backscatter {output_terrain_radiometry}"
    if flag_apply_rtc:
        layer_name_rtc_anf = (f"rtc_anf_{output_terrain_radiometry}_to_"
                              f"{input_terrain_radiometry}")
    else:
        layer_name_rtc_anf = ''

    save_rtc_anf, save_rtc_anf_gamma0_to_sigma0 = \
        read_and_validate_rtc_anf_flags(geocode_namespace, flag_apply_rtc,
                                        output_terrain_radiometry_enum, logger)

    rtc_min_value_db = rtc_namespace.rtc_min_value_db
    rtc_upsampling = rtc_namespace.dem_upsampling
    rtc_area_beta_mode = rtc_namespace.area_beta_mode

    if rtc_area_beta_mode == 'pixel_area':
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.PIXEL_AREA
    elif rtc_area_beta_mode == 'projection_angle':
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.PROJECTION_ANGLE
    elif (rtc_area_beta_mode == 'auto' or
            rtc_area_beta_mode is None):
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.AUTO
    else:
        err_msg = ('ERROR invalid area beta mode:'
                   f' {rtc_area_beta_mode}')
        raise ValueError(err_msg)

    logger.info('Identification:')
    logger.info(f'    product type: {product_type}')
    logger.info(f'    product version: {product_version}')
    if save_mosaics:
        logger.info(f'    mosaic product ID: {mosaic_product_id}')
    logger.info('Ancillary input(s):')
    logger.info(f'    DEM file: {cfg.dem}')
    logger.info('Processing parameters:')
    logger.info(f'    apply RTC: {flag_apply_rtc}')
    logger.info(f'    apply thermal noise correction:'
                f' {flag_apply_thermal_noise_correction}')
    logger.info(f'    apply absolute radiometric correction:'
                f' {flag_apply_abs_rad_correction}')
    logger.info(f'    apply valid samples sub-swath masking:'
                f' {apply_valid_samples_sub_swath_masking}')
    logger.info(f'    apply shadow masking:'
                f' {apply_shadow_masking}')
    logger.info(f'    skip if already processed:'
                f' {skip_if_output_files_exist}')
    logger.info(f'    scratch dir: {scratch_path}')
    logger.info(f'    output dir: {output_dir}')
    logger.info(f'    save bursts: {save_bursts}')
    logger.info(f'    save mosaics: {save_mosaics}')
    logger.info(f'    save browse: {flag_save_browse}')
    logger.info(f'    output imagery format: {output_imagery_format}')
    logger.info(f'    output imagery compression:'
                f' {output_imagery_compression}')
    logger.info(f'    output imagery nbits: {output_imagery_nbits}')
    logger.info(f'    save secondary layers as HDF5 files:'
                f' {save_secondary_layers_as_hdf5}')
    logger.info(f'    check ancillary coverage:'
                f' {check_ancillary_inputs_coverage}')

    logger.info('Save layers:')
    logger.info(f'    {layer_names_dict[LAYER_NAME_LAYOVER_SHADOW_MASK]}:'
                f' {save_rtc_anf}')
    logger.info(f'    RTC area normalization factor: {save_rtc_anf}')
    logger.info(f'    RTC area normalization factor Gamma0 to Beta0:'
                f' {save_rtc_anf_gamma0_to_sigma0}')
    logger.info(f'    {layer_names_dict[LAYER_NAME_NUMBER_OF_LOOKS]}:'
                f' {save_nlooks}')
    logger.info(f'    {layer_names_dict[LAYER_NAME_INCIDENCE_ANGLE]}:'
                f' {save_incidence_angle}')
    logger.info(f'    {layer_names_dict[LAYER_NAME_LOCAL_INCIDENCE_ANGLE]}:'
                f' {save_local_inc_angle}')
    logger.info(f'    {layer_names_dict[LAYER_NAME_PROJECTION_ANGLE]}:'
                f' {save_projection_angle}')
    logger.info(f'    {layer_names_dict[LAYER_NAME_RTC_ANF_PROJECTION_ANGLE]}:'
                f' {save_rtc_anf_projection_angle}')
    logger.info(f'    {layer_names_dict[LAYER_NAME_RANGE_SLOPE]}:'
                f' {save_range_slope}')
    logger.info(f'    {layer_names_dict[LAYER_NAME_DEM]}:'
                f' {save_dem}')

    logger.info('Browse images:')
    logger.info(f'    burst height: {browse_image_burst_height}')
    logger.info(f'    burst width: {browse_image_burst_width}')

    if save_mosaics:
        logger.info(f'    mosaic height: {browse_image_mosaic_height}')
        logger.info(f'    mosaic width: {browse_image_mosaic_width}')
        logger.info('Mosaic geogrid:')
        for line in str(cfg.geogrid).split('\n'):
            if not line:
                continue
            logger.info(f'    {line}')

    # check ancillary input (DEM)
    metadata_dict = {}
    check_ancillary_inputs(check_ancillary_inputs_coverage,
                           cfg.dem, cfg.geogrid,
                           metadata_dict, logger=logger)

    # Common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    maxiter = cfg.geo2rdr_params.numiter
    exponent = 1 if (flag_apply_thermal_noise_correction or
                     flag_apply_abs_rad_correction) else 2

    # output mosaics variables
    geo_filename = f'{output_dir}/'f'{mosaic_product_id}.{imagery_extension}'
    output_imagery_list = []
    output_file_list = []
    mosaic_output_file_list = []
    output_metadata_dict = {}

    # output dir (imagery mosaics)
    if save_imagery_as_hdf5:
        output_dir_mosaic_raster = scratch_path
    else:
        output_dir_mosaic_raster = output_dir

    # output dir (secondary layers mosaics)
    if save_secondary_layers_as_hdf5:
        output_dir_sec_mosaic_raster = scratch_path
    else:
        output_dir_sec_mosaic_raster = output_dir

    add_output_to_output_metadata_dict(
        save_layover_shadow_mask, LAYER_NAME_LAYOVER_SHADOW_MASK,
        output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)
    add_output_to_output_metadata_dict(
        save_nlooks, LAYER_NAME_NUMBER_OF_LOOKS, output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)
    add_output_to_output_metadata_dict(
        save_rtc_anf, layer_name_rtc_anf,
        output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)
    add_output_to_output_metadata_dict(
        save_rtc_anf_gamma0_to_sigma0, LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0,
        output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)

    temp_files_list = []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)
    vrt_options_mosaic = gdal.BuildVRTOptions(separate=True)

    n_bursts = len(cfg.bursts.items())
    logger.info(f'Number of bursts to process: {n_bursts}')

    hdf5_mosaic_obj = None
    output_hdf5_file = os.path.join(
        output_dir, f'{mosaic_product_id}.{hdf5_file_extension}')

    lookside = None
    wavelength = None
    orbit = None
    mosaic_geotiff_metadata_dict = None

    # iterate over sub-burts
    for burst_index, (burst_id, burst_pol_dict) in \
            enumerate(cfg.bursts.items()):

        # ===========================================================
        # start burst processing

        t_burst_start = time.time()
        logger.info(f'Processing burst: {burst_id} ({burst_index+1}/'
                    f'{n_bursts})')

        geogrid = cfg.geogrids[burst_id]
        pol_list = list(burst_pol_dict.keys())
        burst = burst_pol_dict[pol_list[0]]

        # populate burst_product_id
        pixel_spacing_avg = int((geogrid.spacing_x + geogrid.spacing_y) / 2)
        burst_product_id = populate_product_id(
            runconfig_product_id, burst, processing_datetime, product_version,
            pixel_spacing_avg, product_type, rtc_s1_static_validity_start_date,
            is_mosaic=True)

        logger.info(f'    product ID: {burst_product_id}')

        if product_type == STATIC_LAYERS_PRODUCT_TYPE:
            # for static layers, we just use the first polarization as
            # reference
            pol_list = [pol_list[0]]
            burst_pol_dict = {pol_list[0]: burst}

        flag_bursts_files_are_temporary = (not save_bursts or
                                           save_imagery_as_hdf5)
        flag_bursts_secondary_files_are_temporary = (
            not save_bursts or save_secondary_layers_as_hdf5)

        burst_scratch_path = f'{scratch_path}/{burst_id}/'
        os.makedirs(burst_scratch_path, exist_ok=True)

        output_dir_bursts = os.path.join(output_dir, burst_id)
        os.makedirs(output_dir_bursts, exist_ok=True)

        if not save_bursts or save_secondary_layers_as_hdf5:
            # burst files are saved in scratch dir
            output_dir_sec_bursts = burst_scratch_path
        else:
            # burst files (individual or HDF5) are saved in burst_id dir
            output_dir_sec_bursts = output_dir_bursts

        logger.info('    burst geogrid:')
        for line in str(geogrid).split('\n'):
            if not line:
                continue
            logger.info(f'        {line}')

        # snap coordinates
        x_snap = geogrid.spacing_x
        y_snap = geogrid.spacing_y
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)

        # Create burst HDF5
        if (save_hdf5_metadata and save_bursts):
            hdf5_file_output_dir = os.path.join(output_dir, burst_id)
            os.makedirs(hdf5_file_output_dir, exist_ok=True)
            output_hdf5_file_burst = os.path.join(
                hdf5_file_output_dir,
                f'{burst_product_id}.{hdf5_file_extension}')

        # If burst imagery is not temporary, separate polarization channels
        output_burst_imagery_list = []
        if not flag_bursts_files_are_temporary:
            for pol in pol_list:
                geo_burst_pol_filename = \
                    os.path.join(output_dir_bursts,
                                 f'{burst_product_id}_{pol}.' +
                                 f'{imagery_extension}')
                output_burst_imagery_list.append(geo_burst_pol_filename)

        else:
            for pol in pol_list:
                geo_burst_pol_filename = (f'NETCDF:{output_hdf5_file_burst}:'
                                          f'{DATA_BASE_GROUP}/'
                                          f'{pol}')
                output_burst_imagery_list.append(geo_burst_pol_filename)

        # skip geocoding if output files exist
        flag_process = (not skip_if_output_files_exist or
                        not all([_test_valid_gdal_ref(f)
                                for f in output_burst_imagery_list]))
        if not flag_process:
            logger.info(f'    found geocoded files for burst {burst_id}!'
                        ' Skipping burst process.')
        else:
            logger.info('    reading burst SLCs')

        radar_grid = burst.as_isce3_radargrid()
        if product_type == STATIC_LAYERS_PRODUCT_TYPE:
            radar_grid = radar_grid.offset_and_resize(
                - int((STATIC_LAYERS_AZ_MARGIN) * radar_grid.length),
                - int((STATIC_LAYERS_RG_MARGIN) * radar_grid.width),
                int((1 + 2 * STATIC_LAYERS_AZ_MARGIN) * radar_grid.length),
                int((1 + 2 * STATIC_LAYERS_RG_MARGIN) * radar_grid.width))
        # native_doppler = burst.doppler.lut2d
        orbit = burst.orbit
        wavelength = burst.wavelength
        lookside = radar_grid.lookside

        input_file_list = []
        for pol, burst_pol in burst_pol_dict.items():
            temp_slc_path = \
                os.path.join(burst_scratch_path, f'slc_{pol}.vrt')
            temp_slc_corrected_path = \
                os.path.join(burst_scratch_path,
                             f'slc_{pol}_corrected.{imagery_extension}')

            if (flag_process and (flag_apply_thermal_noise_correction or
                flag_apply_abs_rad_correction) and 
                    product_type == STATIC_LAYERS_PRODUCT_TYPE):
                fill_value = 1
                build_empty_vrt(temp_slc_path, radar_grid.length,
                                radar_grid.width, fill_value)
                input_burst_filename = temp_slc_path
                temp_files_list.append(temp_slc_path)
 
            elif (flag_process and (flag_apply_thermal_noise_correction or
                  flag_apply_abs_rad_correction)):

                apply_slc_corrections(
                    burst_pol,
                    temp_slc_path,
                    temp_slc_corrected_path,
                    flag_output_complex=False,
                    flag_thermal_correction=flag_apply_thermal_noise_correction,
                    flag_apply_abs_rad_correction=True)
                input_burst_filename = temp_slc_corrected_path
                temp_files_list.append(temp_slc_corrected_path)
            else:
                input_burst_filename = temp_slc_path

            temp_files_list.append(temp_slc_path)
            input_file_list.append(input_burst_filename)

        # At this point, burst imagery files are always temporary
        geo_burst_filename = \
            f'{burst_scratch_path}/{burst_product_id}.{imagery_extension}'
        temp_files_list.append(geo_burst_filename)

        out_geo_nlooks_obj = None
        if save_nlooks:
            nlooks_file = (f'{output_dir_sec_bursts}/{burst_product_id}'
                           f'_{LAYER_NAME_NUMBER_OF_LOOKS}.{imagery_extension}')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(nlooks_file)
            else:
                output_file_list.append(nlooks_file)
        else:
            nlooks_file = None

        out_geo_rtc_obj = None
        if save_rtc_anf:
            rtc_anf_file = (f'{output_dir_sec_bursts}/{burst_product_id}'
                            f'_{layer_name_rtc_anf}.{imagery_extension}')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(rtc_anf_file)
            else:
                output_file_list.append(rtc_anf_file)

        else:
            rtc_anf_file = None

        out_geo_rtc_gamma0_to_sigma0_obj = None
        if save_rtc_anf_gamma0_to_sigma0:
            rtc_anf_gamma0_to_sigma0_file = \
                (f'{output_dir_sec_bursts}/{burst_product_id}'
                 f'_{LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0}.{imagery_extension}')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(rtc_anf_gamma0_to_sigma0_file)
            else:
                output_file_list.append(rtc_anf_gamma0_to_sigma0_file)

        else:
            rtc_anf_gamma0_to_sigma0_file = None

        # geocoding optional arguments
        geocode_kwargs = {}
        layover_shadow_mask_geocode_kwargs = {}

        # get sub_swaths metadata
        if (flag_process and apply_valid_samples_sub_swath_masking and
                not product_type == STATIC_LAYERS_PRODUCT_TYPE):
            # Extract burst boundaries and create sub_swaths object to mask
            # invalid radar samples
            n_subswaths = 1
            sub_swaths = isce3.product.SubSwaths(radar_grid.length,
                                                 radar_grid.width,
                                                 n_subswaths)
            last_range_sample = min([burst.last_valid_sample,
                                     radar_grid.width])
            valid_samples_sub_swath = np.repeat(
                [[burst.first_valid_sample, last_range_sample + 1]],
                radar_grid.length, axis=0)
            for i in range(burst.first_valid_line):
                valid_samples_sub_swath[i, :] = 0
            for i in range(burst.last_valid_line, radar_grid.length):
                valid_samples_sub_swath[i, :] = 0

            sub_swaths.set_valid_samples_array(1, valid_samples_sub_swath)
            geocode_kwargs['sub_swaths'] = sub_swaths
            layover_shadow_mask_geocode_kwargs['sub_swaths'] = sub_swaths

        # Calculate geolocation correction LUT
        if flag_process and cfg.groups.processing.apply_correction_luts:

            # Calculates the LUTs for one polarization in `burst_pol_dict`
            pol_burst_for_lut = next(iter(burst_pol_dict))
            burst_for_lut = burst_pol_dict[pol_burst_for_lut]
            rg_lut, az_lut = compute_correction_lut(burst_for_lut,
                                                    dem_raster,
                                                    burst_scratch_path,
                                                    rg_step_meters,
                                                    az_step_meters)

            geocode_kwargs['az_time_correction'] = az_lut
            geocode_kwargs['slant_range_correction'] = rg_lut

        # Calculate layover/shadow mask when requested
        if save_layover_shadow_mask or apply_shadow_masking:
            flag_layover_shadow_mask_is_temporary = \
                (flag_bursts_secondary_files_are_temporary or
                    (apply_shadow_masking and not save_layover_shadow_mask))

            if flag_layover_shadow_mask_is_temporary:
                # layover/shadow mask is temporary
                layover_shadow_mask_file = \
                    (f'{burst_scratch_path}/{burst_product_id}'
                     f'_{LAYER_NAME_LAYOVER_SHADOW_MASK}.{imagery_extension}')
            else:
                # layover/shadow mask is saved in `output_dir_sec_bursts`
                layover_shadow_mask_file = \
                    (f'{output_dir_sec_bursts}/{burst_product_id}'
                     f'_{LAYER_NAME_LAYOVER_SHADOW_MASK}.{imagery_extension}')
            if flag_process:

                logger.info('    computing layover shadow mask for'
                            f' {burst_id}')

                if product_type == STATIC_LAYERS_PRODUCT_TYPE:
                    radar_grid_layover_shadow_mask = radar_grid.multilook(
                        STATIC_LAYERS_LAYOVER_SHADOW_MASK_MULTILOOK_FACTOR,
                        STATIC_LAYERS_LAYOVER_SHADOW_MASK_MULTILOOK_FACTOR)
                else:
                    radar_grid_layover_shadow_mask = radar_grid

                slantrange_layover_shadow_mask_raster = \
                    compute_layover_shadow_mask(
                        radar_grid_layover_shadow_mask,
                        orbit,
                        geogrid,
                        burst,
                        dem_raster,
                        layover_shadow_mask_file,
                        output_raster_format,
                        burst_scratch_path,
                        shadow_dilation_size=shadow_dilation_size,
                        threshold_rdr2geo=cfg.rdr2geo_params.threshold,
                        numiter_rdr2geo=cfg.rdr2geo_params.numiter,
                        threshold_geo2rdr=cfg.geo2rdr_params.threshold,
                        numiter_geo2rdr=cfg.geo2rdr_params.numiter,
                        memory_mode=geocode_namespace.memory_mode,
                        geocode_options=layover_shadow_mask_geocode_kwargs)

            if flag_layover_shadow_mask_is_temporary:
                temp_files_list.append(layover_shadow_mask_file)
            else:
                output_file_list.append(layover_shadow_mask_file)
                logger.info(f'file saved: {layover_shadow_mask_file}')
                if save_layover_shadow_mask:
                    output_metadata_dict[LAYER_NAME_LAYOVER_SHADOW_MASK][1].append(
                        layover_shadow_mask_file)

            if not save_layover_shadow_mask:
                layover_shadow_mask_file = None

            # The radar grid for static layers is multilooked by a factor of
            # STATIC_LAYERS_LAYOVER_SHADOW_MASK_MULTILOOK_FACTOR. If that
            # number is not unitary, the layover shadow mask cannot be used
            # with geocoding
            if (apply_shadow_masking and flag_process and
                    product_type != STATIC_LAYERS_PRODUCT_TYPE or
                    STATIC_LAYERS_LAYOVER_SHADOW_MASK_MULTILOOK_FACTOR == 1):
                geocode_kwargs['input_layover_shadow_mask_raster'] = \
                    slantrange_layover_shadow_mask_raster
        else:
            layover_shadow_mask_file = None

        if flag_process:

            if save_nlooks:
                out_geo_nlooks_obj = isce3.io.Raster(
                    nlooks_file, geogrid.width, geogrid.length, 1,
                    gdal.GDT_Float32, output_raster_format)

            if save_rtc_anf:
                out_geo_rtc_obj = isce3.io.Raster(
                    rtc_anf_file,
                    geogrid.width, geogrid.length, 1,
                    gdal.GDT_Float32, output_raster_format)

            if save_rtc_anf_gamma0_to_sigma0:
                out_geo_rtc_gamma0_to_sigma0_obj = isce3.io.Raster(
                    rtc_anf_gamma0_to_sigma0_file,
                    geogrid.width, geogrid.length, 1,
                    gdal.GDT_Float32, output_raster_format)
                geocode_kwargs['out_geo_rtc_gamma0_to_sigma0'] = \
                    out_geo_rtc_gamma0_to_sigma0_obj

            # create multi-band VRT
            if len(input_file_list) == 1:
                rdr_burst_raster = isce3.io.Raster(input_file_list[0])
            else:
                temp_vrt_path = f'{burst_scratch_path}/slc.vrt'
                gdal.BuildVRT(temp_vrt_path, input_file_list,
                              options=vrt_options_mosaic)
                rdr_burst_raster = isce3.io.Raster(temp_vrt_path)
                temp_files_list.append(temp_vrt_path)

            # Generate output geocoded burst raster
            geo_burst_raster = isce3.io.Raster(
                geo_burst_filename,
                geogrid.width, geogrid.length,
                rdr_burst_raster.num_bands, gdal.GDT_Float32,
                output_raster_format)

            # init Geocode object depending on raster type
            if rdr_burst_raster.datatype() == gdal.GDT_Float32:
                geo_obj = isce3.geocode.GeocodeFloat32()
            elif rdr_burst_raster.datatype() == gdal.GDT_Float64:
                geo_obj = isce3.geocode.GeocodeFloat64()
            elif rdr_burst_raster.datatype() == gdal.GDT_CFloat32:
                geo_obj = isce3.geocode.GeocodeCFloat32()
            elif rdr_burst_raster.datatype() == gdal.GDT_CFloat64:
                geo_obj = isce3.geocode.GeocodeCFloat64()
            else:
                err_str = 'Unsupported raster type for geocoding'
                raise NotImplementedError(err_str)

            # init geocode members
            geo_obj.orbit = orbit
            geo_obj.ellipsoid = ellipsoid
            geo_obj.doppler = zero_doppler
            geo_obj.threshold_geo2rdr = threshold
            geo_obj.numiter_geo2rdr = maxiter

            # set data interpolator based on the geocode algorithm
            if geocode_algorithm == isce3.geocode.GeocodeOutputMode.INTERP:
                geo_obj.data_interpolator = geocode_algorithm

            geo_obj.geogrid(geogrid.start_x, geogrid.start_y,
                            geogrid.spacing_x, geogrid.spacing_y,
                            geogrid.width, geogrid.length, geogrid.epsg)

            geo_obj.geocode(radar_grid=radar_grid,
                            input_raster=rdr_burst_raster,
                            output_raster=geo_burst_raster,
                            dem_raster=dem_raster,
                            output_mode=geocode_algorithm,
                            geogrid_upsampling=geogrid_upsampling,
                            flag_apply_rtc=flag_apply_rtc,
                            input_terrain_radiometry=input_terrain_radiometry_enum,
                            output_terrain_radiometry=output_terrain_radiometry_enum,
                            exponent=exponent,
                            rtc_min_value_db=rtc_min_value_db,
                            rtc_upsampling=rtc_upsampling,
                            rtc_algorithm=rtc_algorithm,
                            abs_cal_factor=abs_cal_factor,
                            flag_upsample_radar_grid=flag_upsample_radar_grid,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            out_geo_nlooks=out_geo_nlooks_obj,
                            out_geo_rtc=out_geo_rtc_obj,
                            rtc_area_beta_mode=rtc_area_beta_mode_enum,
                            # out_geo_rtc_gamma0_to_sigma0=out_geo_rtc_gamma0_to_sigma0_obj,
                            input_rtc=None,
                            output_rtc=None,
                            dem_interp_method=dem_interp_method_enum,
                            memory_mode=memory_mode,
                            **geocode_kwargs)

            del geo_burst_raster

            # Output imagery list contains multi-band files that
            # will be used for mosaicking
            if product_type != STATIC_LAYERS_PRODUCT_TYPE:
                output_imagery_list.append(geo_burst_filename)

        else:
            # Bundle the single-pol geo burst files into .vrt
            geo_burst_filename = geo_burst_filename.replace(
                f'.{imagery_extension}', '.vrt')
            os.makedirs(os.path.dirname(geo_burst_filename), exist_ok=True)
            gdal.BuildVRT(geo_burst_filename, output_burst_imagery_list,
                          options=vrt_options_mosaic)
            if product_type != STATIC_LAYERS_PRODUCT_TYPE:
                output_imagery_list.append(geo_burst_filename)

        if (flag_process and save_layover_shadow_mask and
                not save_secondary_layers_as_hdf5):
            set_mask_fill_value_and_ctable(layover_shadow_mask_file,
                                           geo_burst_filename)

        # If burst imagery is not temporary, separate polarization channels
        if (flag_process and not flag_bursts_files_are_temporary and
                product_type != STATIC_LAYERS_PRODUCT_TYPE):
            _separate_pol_channels(geo_burst_filename,
                                   output_burst_imagery_list,
                                   output_raster_format, logger)

            output_file_list += output_burst_imagery_list

        if save_nlooks:
            if flag_process:
                out_geo_nlooks_obj.close_dataset()
                del out_geo_nlooks_obj

                if not flag_bursts_secondary_files_are_temporary:
                    logger.info(f'file saved: {nlooks_file}')
            output_metadata_dict[
                LAYER_NAME_NUMBER_OF_LOOKS][1].append(nlooks_file)

        if save_rtc_anf:
            if flag_process:
                out_geo_rtc_obj.close_dataset()
                del out_geo_rtc_obj

                if not flag_bursts_secondary_files_are_temporary:
                    logger.info(f'file saved: {rtc_anf_file}')
            output_metadata_dict[layer_name_rtc_anf][1].append(
                rtc_anf_file)

        if save_rtc_anf_gamma0_to_sigma0:
            if flag_process:
                out_geo_rtc_gamma0_to_sigma0_obj.close_dataset()
                del out_geo_rtc_gamma0_to_sigma0_obj

                if not flag_bursts_secondary_files_are_temporary:
                    logger.info(f'file saved: {rtc_anf_gamma0_to_sigma0_file}')
            output_metadata_dict[
                LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0][1].append(
                    rtc_anf_gamma0_to_sigma0_file)

        radar_grid_file_dict = {}

        if flag_process and flag_call_radar_grid and save_bursts:
            get_radar_grid(
                geogrid, dem_interp_method_enum, burst_product_id,
                output_dir_sec_bursts, imagery_extension, save_incidence_angle,
                save_local_inc_angle, save_projection_angle,
                save_rtc_anf_projection_angle,
                save_range_slope, save_dem,
                dem_raster, radar_grid_file_dict,
                lookside, wavelength, orbit,
                verbose=not flag_bursts_secondary_files_are_temporary)
            if flag_bursts_secondary_files_are_temporary:
                # files are temporary
                temp_files_list += list(radar_grid_file_dict.values())
            else:
                output_file_list += list(radar_grid_file_dict.values())

        # Create burst HDF5
        if (flag_process and save_hdf5_metadata and save_bursts):
            hdf5_file_output_dir = os.path.join(output_dir, burst_id)
            with create_hdf5_file(burst_product_id,
                                  output_hdf5_file_burst, orbit, burst, cfg,
                                  processing_datetime,
                                  is_mosaic=False) as hdf5_burst_obj:
                save_hdf5_file(
                    hdf5_burst_obj, output_hdf5_file_burst,
                    clip_max, clip_min, output_radiometry_str,
                    geogrid, pol_list, geo_burst_filename, nlooks_file,
                    rtc_anf_file, layer_name_rtc_anf,
                    rtc_anf_gamma0_to_sigma0_file,
                    layover_shadow_mask_file, radar_grid_file_dict,
                    save_imagery=save_imagery_as_hdf5,
                    save_secondary_layers=save_secondary_layers_as_hdf5)
            output_file_list.append(output_hdf5_file_burst)

        # Save browse image (burst)
        if flag_process and flag_save_browse:
            browse_image_filename = \
                os.path.join(output_dir_bursts, f'{burst_product_id}.png')
            save_browse(output_burst_imagery_list, browse_image_filename,
                        pol_list, browse_image_burst_height,
                        browse_image_burst_width, temp_files_list,
                        burst_scratch_path, logger)
            output_file_list.append(browse_image_filename)

        # Append metadata to burst GeoTIFFs
        if (flag_process and (not flag_bursts_files_are_temporary or
                              save_secondary_layers_as_hdf5)):
            metadata_dict = get_metadata_dict(burst_product_id, burst, cfg,
                                              processing_datetime,
                                              is_mosaic=False)
            geotiff_metadata_dict = all_metadata_dict_to_geotiff_metadata_dict(
                metadata_dict)
            for current_file in output_file_list:
                if not current_file.endswith('.tif'):
                    continue
                append_metadata_to_geotiff_file(current_file,
                                                geotiff_metadata_dict,
                                                burst_product_id)

        # Create mosaic HDF5
        if (save_hdf5_metadata and save_mosaics
                and burst_index == 0):
            hdf5_mosaic_obj = create_hdf5_file(
                mosaic_product_id, output_hdf5_file, orbit, burst, cfg,
                processing_datetime, is_mosaic=True)

        # Save mosaic metadata for later use
        if (save_mosaics and burst_index == 0):
            mosaic_metadata_dict = get_metadata_dict(mosaic_product_id, burst,
                                                     cfg, processing_datetime,
                                                     is_mosaic=True)
            mosaic_geotiff_metadata_dict = \
                all_metadata_dict_to_geotiff_metadata_dict(
                    mosaic_metadata_dict)

        t_burst_end = time.time()
        logger.info(
            f'elapsed time (burst): {t_burst_end - t_burst_start}')

        # end burst processing
        # ===========================================================

    if flag_call_radar_grid and save_mosaics:
        radar_grid_file_dict = {}

        if save_secondary_layers_as_hdf5:
            radar_grid_output_dir = scratch_path
        else:
            radar_grid_output_dir = output_dir
        get_radar_grid(
            cfg.geogrid, dem_interp_method_enum, mosaic_product_id,
            radar_grid_output_dir, imagery_extension,
            save_incidence_angle,
            save_local_inc_angle, save_projection_angle,
            save_rtc_anf_projection_angle,
            save_range_slope, save_dem,
            dem_raster, radar_grid_file_dict,
            lookside, wavelength, orbit,
            verbose=not flag_bursts_secondary_files_are_temporary)
        if flag_bursts_secondary_files_are_temporary:
            # files are temporary
            temp_files_list += list(radar_grid_file_dict.values())
        else:
            output_file_list += list(radar_grid_file_dict.values())
            mosaic_output_file_list += list(radar_grid_file_dict.values())

    if save_mosaics:

        if len(output_imagery_list) > 0:
            # Mosaic sub-bursts imagery
            logger.info('mosaicking files:')
            output_imagery_filename_list = []
            for pol in pol_list:
                geo_pol_filename = \
                    (f'{output_dir_mosaic_raster}/{mosaic_product_id}_{pol}.'
                     f'{imagery_extension}')
                logger.info(f'    {geo_pol_filename}')
                output_imagery_filename_list.append(geo_pol_filename)

        if save_nlooks:
            nlooks_list = output_metadata_dict[LAYER_NAME_NUMBER_OF_LOOKS][1]
        else:
            nlooks_list = []

        if len(output_imagery_list) > 0:

            mosaic_multiple_output_files(
                output_imagery_list, nlooks_list,
                output_imagery_filename_list, mosaic_mode,
                scratch_dir=scratch_path, geogrid_in=cfg.geogrid,
                temp_files_list=temp_files_list,
                output_raster_format=output_raster_format)

            if save_imagery_as_hdf5:
                temp_files_list += output_imagery_filename_list
            else:
                output_file_list += output_imagery_filename_list
                mosaic_output_file_list += output_imagery_filename_list

        # Mosaic other layers
        for key, (output_file, input_files) in output_metadata_dict.items():
            logger.info(f'mosaicking file: {output_file}')
            if len(input_files) == 0:
                continue

            mosaic_single_output_file(
                input_files, nlooks_list, output_file,
                mosaic_mode, scratch_dir=scratch_path,
                geogrid_in=cfg.geogrid, temp_files_list=temp_files_list,
                output_raster_format=output_raster_format)

            if save_secondary_layers_as_hdf5:
                temp_files_list.append(output_file)
            else:
                output_file_list.append(output_file)
                mosaic_output_file_list.append(output_file)

        # Save browse image (mosaic)
        if flag_save_browse:
            browse_image_filename = \
                os.path.join(output_dir, f'{mosaic_product_id}.png')
            save_browse(output_imagery_filename_list, browse_image_filename,
                        pol_list, browse_image_mosaic_height,
                        browse_image_mosaic_width, temp_files_list,
                        scratch_path, logger)
            output_file_list.append(browse_image_filename)
            mosaic_output_file_list.append(browse_image_filename)

        # Save HDF5
        if save_hdf5_metadata:
            if save_nlooks:
                nlooks_mosaic_file = output_metadata_dict[
                    LAYER_NAME_NUMBER_OF_LOOKS][0]
            else:
                nlooks_mosaic_file = None
            if save_rtc_anf:
                rtc_anf_mosaic_file = output_metadata_dict[
                    layer_name_rtc_anf][0]
            else:
                rtc_anf_mosaic_file = None
            if save_rtc_anf_gamma0_to_sigma0:
                rtc_anf_gamma0_to_sigma0_mosaic_file = output_metadata_dict[
                    LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0][0]
            else:
                rtc_anf_gamma0_to_sigma0_mosaic_file = None
            if save_layover_shadow_mask:
                layover_shadow_mask_file = output_metadata_dict[
                    LAYER_NAME_LAYOVER_SHADOW_MASK][0]
            else:
                layover_shadow_mask_file = None

            # Update metadata datasets that depend on all bursts
            sensing_start = None
            sensing_stop = None
            for burst_id, burst_pol_dict in cfg.bursts.items():
                pols = list(burst_pol_dict.keys())
                burst = burst_pol_dict[pols[0]]

                if (sensing_start is None or
                        burst.sensing_start < sensing_start):
                    sensing_start = burst.sensing_start

                if sensing_stop is None or burst.sensing_stop > sensing_stop:
                    sensing_stop = burst.sensing_stop

            sensing_start_ds = '/identification/zeroDopplerStartTime'
            sensing_end_ds = '/identification/zeroDopplerEndTime'
            if sensing_start_ds in hdf5_mosaic_obj:
                del hdf5_mosaic_obj[sensing_start_ds]
            if sensing_end_ds in hdf5_mosaic_obj:
                del hdf5_mosaic_obj[sensing_end_ds]
            hdf5_mosaic_obj[sensing_start_ds] = \
                sensing_start.strftime('%Y-%m-%dT%H:%M:%S.%f')
            hdf5_mosaic_obj[sensing_end_ds] = \
                sensing_stop.strftime('%Y-%m-%dT%H:%M:%S.%f')

            # Bundle the mosaicked single-pol rasters
            geo_filename_vrt = f'{geo_filename}.vrt'
            gdal.BuildVRT(geo_filename_vrt, output_imagery_filename_list,
                          options=vrt_options_mosaic)
            temp_files_list.append(geo_filename_vrt)

            save_hdf5_file(
                hdf5_mosaic_obj, output_hdf5_file,
                clip_max, clip_min, output_radiometry_str,
                cfg.geogrid, pol_list, geo_filename_vrt, nlooks_mosaic_file,
                rtc_anf_mosaic_file, layer_name_rtc_anf,
                rtc_anf_gamma0_to_sigma0_mosaic_file,
                layover_shadow_mask_file,
                radar_grid_file_dict, save_imagery=save_imagery_as_hdf5,
                save_secondary_layers=save_secondary_layers_as_hdf5)
            hdf5_mosaic_obj.close()
            output_file_list.append(output_hdf5_file)

        # Append metadata to mosaic GeoTIFFs
        for current_file in mosaic_output_file_list:
            if not current_file.endswith('.tif'):
                continue
            append_metadata_to_geotiff_file(current_file,
                                            mosaic_geotiff_metadata_dict,
                                            mosaic_product_id)

    # Save GeoTIFFs as cloud optimized GeoTIFFs (COGs)
    if output_imagery_format == 'COG':
        logger.info('Saving files as Cloud-Optimized GeoTIFFs (COGs)')
        for filename in output_file_list:
            if not filename.endswith('.tif'):
                continue

            logger.info(f'    processing file: {filename}')

            # if file is backscatter, use the 'AVERAGE' mode to create overlays
            options_save_as_cog = {}
            gdal_ds = gdal.Open(filename, gdal.GA_ReadOnly)
            description = gdal_ds.GetRasterBand(1).GetDescription()
            if description and 'backscatter' in description.lower():
                options_save_as_cog['ovr_resamp_algorithm'] = 'AVERAGE'
            del gdal_ds

            save_as_cog(filename, scratch_path, logger,
                        compression=output_imagery_compression,
                        nbits=output_imagery_nbits,
                        **options_save_as_cog)

    logger.info('removing temporary files:')
    for filename in temp_files_list:
        if not os.path.isfile(filename):
            continue
        os.remove(filename)

        logger.info(f'    {filename}')

    logger.info('output files:')
    for filename in output_file_list:
        logger.info(f'    {filename}')

    t_end = time.time()
    logger.info(f'elapsed time: {t_end - t_start}')

    # Return value to indicate that the processing has completed succesfully
    return 0


def get_radar_grid(geogrid, dem_interp_method_enum, product_id,
                   output_dir, extension, save_incidence_angle,
                   save_local_inc_angle, save_projection_angle,
                   save_rtc_anf_projection_angle,
                   save_range_slope, save_dem, dem_raster,
                   radar_grid_file_dict, lookside, wavelength, orbit,
                   verbose=True):
    output_obj_list = []
    layers_nbands = 1
    shape = [layers_nbands, geogrid.length, geogrid.width]

    incidence_angle_raster = _create_raster_obj(
        output_dir, product_id, LAYER_NAME_INCIDENCE_ANGLE,
        gdal.GDT_Float32, shape, radar_grid_file_dict,
        output_obj_list, save_incidence_angle, extension)
    local_incidence_angle_raster = _create_raster_obj(
        output_dir, product_id, LAYER_NAME_LOCAL_INCIDENCE_ANGLE,
        gdal.GDT_Float32, shape,
        radar_grid_file_dict, output_obj_list, save_local_inc_angle,
        extension)
    projection_angle_raster = _create_raster_obj(
        output_dir, product_id, LAYER_NAME_PROJECTION_ANGLE,
        gdal.GDT_Float32, shape, radar_grid_file_dict,
        output_obj_list, save_projection_angle, extension)
    rtc_anf_projection_angle_raster = _create_raster_obj(
        output_dir, product_id, LAYER_NAME_RTC_ANF_PROJECTION_ANGLE,
        gdal.GDT_Float32, shape,
        radar_grid_file_dict, output_obj_list,
        save_rtc_anf_projection_angle, extension)
    range_slope_raster = _create_raster_obj(
        output_dir, product_id, LAYER_NAME_RANGE_SLOPE,
        gdal.GDT_Float32, shape, radar_grid_file_dict,
        output_obj_list, save_range_slope, extension)
    interpolated_dem_raster = _create_raster_obj(
        output_dir, product_id, LAYER_NAME_DEM,
        gdal.GDT_Float32, shape, radar_grid_file_dict,
        output_obj_list, save_dem, extension)

    # TODO review this (Doppler)!!!
    # native_doppler = burst.doppler.lut2d
    native_doppler = isce3.core.LUT2d()
    native_doppler.bounds_error = False
    grid_doppler = isce3.core.LUT2d()
    grid_doppler.bounds_error = False

    # TODO: update code below
    # Computation of range slope is not merged to ISCE yet
    kwargs_get_radar_grid = {}
    if range_slope_raster:
        kwargs_get_radar_grid['range_slope_angle_raster'] = \
            range_slope_raster

    # call get_radar_grid()
    isce3.geogrid.get_radar_grid(
        lookside,
        wavelength,
        dem_raster,
        geogrid,
        orbit,
        native_doppler,
        grid_doppler,
        incidence_angle_raster=incidence_angle_raster,
        local_incidence_angle_raster=local_incidence_angle_raster,
        projection_angle_raster=projection_angle_raster,
        simulated_radar_brightness_raster=rtc_anf_projection_angle_raster,
        interpolated_dem_raster=interpolated_dem_raster,
        dem_interp_method=dem_interp_method_enum,
        **kwargs_get_radar_grid)

    # Flush data
    for obj in output_obj_list:
        del obj

    if not verbose:
        return


def get_rtc_s1_parser():
    '''Initialize YamlArgparse class and parse CLI arguments for OPERA RTC.
    '''
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_config_path',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Path to run config file')

    parser.add_argument('--log',
                        '--log-file',
                        dest='log_file',
                        type=str,
                        help='Log file')

    parser.add_argument('--full-log-format',
                        dest='full_log_formatting',
                        action='store_true',
                        default=False,
                        help='Enable full formatting of log messages')

    return parser
