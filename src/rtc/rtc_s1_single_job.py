'''
RTC-S1 Science Application Software (single job)
'''

import datetime
import os
import time
import tempfile

import logging
import numpy as np
from osgeo import gdal
import argparse

import isce3

from s1reader.s1_burst_slc import Sentinel1BurstSlc

from rtc.geogrid import snap_coord
from rtc.runconfig import RunConfig
from rtc.mosaic_geobursts import (compute_weighted_mosaic_raster,
                                  compute_weighted_mosaic_raster_single_band)
from rtc.core import create_logger, save_as_cog
from rtc.h5_prep import (save_hdf5_file, create_hdf5_file, BASE_HDF5_DATASET,
                         get_metadata_dict,
                         all_metadata_dict_to_geotiff_metadata_dict)
from rtc.version import VERSION as SOFTWARE_VERSION
import matplotlib.image as mpimg

logger = logging.getLogger('rtc_s1')


def compute_correction_lut(burst_in, dem_raster, scratch_path,
                           rg_step=120,
                           az_step=120):
    '''
    Compute lookup table for geolocation correction

    Parameters
    ----------
    burst_in: Sentinel1BurstSlc
        Input burst SLC
    dem_raster: isce3.io.raster
        DEM to run rdr2geo
    scratch_path: str
        Scratch path where the radargrid rasters will be saved
    rg_step: float
        LUT spacing in slant range. Unit: meters
    az_step: float
        LUT spacing in azimth direction. Unit: meters

    Returns
    -------
    rg_lut, az_lut: isce3.core.LUT2d
        LUT2d for geolocation correction in slant range and azimuth direction
    '''

    # convert az_step in meters into seconds
    numrow_orbit = burst_in.orbit.position.shape[0]
    vel_mid = burst_in.orbit.velocity[numrow_orbit // 2, :]
    spd_mid = np.linalg.norm(vel_mid)
    pos_mid = burst_in.orbit.position[numrow_orbit // 2, :]
    alt_mid = np.linalg.norm(pos_mid)

    r = 6371000.0 # geometric mean of WGS84 ellipsoid

    az_step_sec = (az_step * alt_mid) / (spd_mid * r)
    


    # Bistatic - azimuth direction
    bistatic_delay = burst_in.bistatic_delay(range_step=rg_step, az_step=az_step_sec)

    # Calculate rdr2geo rasters
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    rdr_grid = burst_in.as_isce3_radargrid(az_step=az_step_sec,
                                           rg_step=rg_step)

    grid_doppler = isce3.core.LUT2d()

    # Initialize the rdr2geo object
    rdr2geo_obj = isce3.geometry.Rdr2Geo(rdr_grid, burst_in.orbit,
                                         ellipsoid, grid_doppler,
                                         threshold=1.0e-8)

    # Get the rdr2geo raster needed for SET computation
    topo_output = {f'{scratch_path}/height.rdr': gdal.GDT_Float64,
                   f'{scratch_path}/incidence_angle.rdr': gdal.GDT_Float32}
    raster_list = [
        isce3.io.Raster(fname, rdr_grid.width,
                        rdr_grid.length, 1, dtype, 'ENVI')
        for fname, dtype in topo_output.items()]
    height_raster, incidence_raster = raster_list

    rdr2geo_obj.topo(dem_raster, x_raster=None, y_raster=None,
                     height_raster=height_raster,
                     incidence_angle_raster=incidence_raster)

    height_raster.close_dataset()
    incidence_raster.close_dataset()

    # Load the lon / lat / hgt value from the raster
    height_arr =\
        gdal.Open(f'{scratch_path}/height.rdr', gdal.GA_ReadOnly).ReadAsArray()
    incidence_angle_arr =\
        gdal.Open(f'{scratch_path}/incidence_angle.rdr', gdal.GA_ReadOnly).ReadAsArray()

    # static troposphere - range direction
    ZPD = 2.3
    H = 6000.0
    tropo = ZPD / np.cos(np.deg2rad(incidence_angle_arr)) * np.exp(-1 * height_arr / H)

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
                browse_image_width =  int(np.round(
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
        if alpha_channel is None:
            alpha_channel = np.asarray(np.isfinite(band_image),
                                       dtype=np.float32)
        vmax = np.nanpercentile(band_image, BROWSE_IMAGE_MAX_PERCENTILE)
        logger.info('        min: 0')
        logger.info(f'        max ({BROWSE_IMAGE_MAX_PERCENTILE}% percentile):'
                    f' {vmax}')
        band_image /= vmax
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


def append_metadata_to_geotiff_file(input_file, metadata_dict):
    '''Append metadata to GeoTIFF file

       Parameters
       ----------
       input_file : str
           Input GeoTIFF file
       metadata_dict : dict
           Metadata dictionary
    '''
    logger.info(f'    appending metadata to GeoTIFF file: {input_file}')
    gdal_ds = gdal.Open(input_file, gdal.GA_Update)
    existing_metadata = gdal_ds.GetMetadata()
    existing_metadata.update(metadata_dict)
    gdal_ds.SetMetadata(existing_metadata)
    del gdal_ds


def _separate_pol_channels(multi_band_file, output_file_list,
                           pol_list, output_radiometry_str,
                           output_raster_format, logger):
    """Save a multi-band raster file as individual single-band files

       Parameters
       ----------
       multi_band_file : str
           Multi-band raster file
       pol_list : list(str)
           List of polarization channels
       output_radiometry_str: str
           Output radiometry
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

        description = f'RTC-S1 {output_radiometry_str} ({pol_list[b]})'
        band_out = raster_out.GetRasterBand(1)
        band_out.SetDescription(description)
        band_out.WriteArray(band_image)
        band_out.FlushCache()
        del band_out
        logger.info(f'file saved: {output_file}')


def _create_raster_obj(output_dir, ds_name, ds_hdf5, dtype, shape,
                       radar_grid_file_dict, output_obj_list, flag_create_raster_obj,
                       extension):
    """Create an ISCE3 raster object (GTiff) for a radar geometry layer.

       Parameters
       ----------
       output_dir: str
              Output directory
       ds_name: str
              Dataset (geometry layer) name
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

    output_file = os.path.join(output_dir, ds_name) + '.' + extension
    raster_obj = isce3.io.Raster(
        output_file,
        shape[2],
        shape[1],
        shape[0],
        dtype,
        "GTiff")
    output_obj_list.append(raster_obj)
    radar_grid_file_dict[ds_hdf5] = output_file
    return raster_obj


def add_output_to_output_metadata_dict(flag, key, output_dir,
        output_metadata_dict, product_id, extension):
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
        Flag to apply radiometirc calibration
    '''

    # Load the SLC of the burst
    burst_in.slc_to_vrt_file(path_slc_vrt)
    slc_gdal_ds = gdal.Open(path_slc_vrt)
    arr_slc_from = slc_gdal_ds.ReadAsArray()

    # Apply thermal noise correction
    if flag_thermal_correction:
        logger.info(f'    applying thermal noise correction to burst SLC')
        corrected_image = np.abs(arr_slc_from) ** 2 - burst_in.thermal_noise_lut
        min_backscatter = 0
        max_backscatter = None
        corrected_image = np.clip(corrected_image, min_backscatter,
                                  max_backscatter)
    else:
        corrected_image = np.abs(arr_slc_from) ** 2

    # Apply absolute radiometric correction
    if flag_apply_abs_rad_correction:
        logger.info(f'    applying absolute radiometric correction to burst SLC')
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


def compute_layover_shadow_mask(radar_grid: isce3.product.RadarGridParameters,
                                orbit: isce3.core.Orbit,
                                geogrid_in: isce3.product.GeoGridParameters,
                                burst_in: Sentinel1BurstSlc,
                                dem_raster: isce3.io.Raster,
                                filename_out: str,
                                output_raster_format: str,
                                threshold_rdr2geo: float=1.0e-7,
                                numiter_rdr2geo: int=25,
                                extraiter_rdr2geo: int=10,
                                lines_per_block_rdr2geo: int=1000,
                                threshold_geo2rdr: float=1.0e-7,
                                numiter_geo2rdr: int=25,
                                memory_mode: isce3.core.GeocodeMemoryMode=None):
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

    Returns
    -------
    slantrange_layover_shadow_mask_raster: isce3.io.Raster
        Layover/shadow-mask ISCE3 raster object in radar coordinates
    '''

    # determine the output filename
    str_datetime = burst_in.sensing_start.strftime('%Y%m%d_%H%M%S.%f')

    path_layover_shadow_mask = (f'layover_shadow_mask_{burst_in.burst_id}_'
                                f'{burst_in.polarization}_{str_datetime}')

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

    slantrange_layover_shadow_mask_raster = isce3.io.Raster(path_layover_shadow_mask,
        radar_grid.width, radar_grid.length, 1, gdal.GDT_Byte, 'MEM')

    rdr2geo_obj.topo(dem_raster, None, None, None,
                     layover_shadow_raster=slantrange_layover_shadow_mask_raster)

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

    geocoded_layover_shadow_mask_raster = isce3.io.Raster(filename_out, 
                                      geogrid_in.width, geogrid_in.length, 1,
                                      gdal.GDT_Byte, output_raster_format)

    geocode_options = {}
    if memory_mode is not None:
        geocode_options['memory_mode'] = memory_mode

    geo.geocode(radar_grid=radar_grid,
                input_raster=slantrange_layover_shadow_mask_raster,
                output_raster=geocoded_layover_shadow_mask_raster,
                dem_raster=dem_raster,
                output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
                **geocode_options)

    return slantrange_layover_shadow_mask_raster


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
    processing_type = cfg.groups.product_group.processing_type
    product_version_float = cfg.groups.product_group.product_version
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

    # read product path group / output format
    runconfig_product_id = cfg.groups.product_group.product_id
    if runconfig_product_id is None:
        runconfig_product_id = 'RTC-S1'
    product_id = f'{runconfig_product_id}_v{product_version}'
    mosaic_product_id = product_id.replace('_{burst_id}', '')

    scratch_path = os.path.join(
        cfg.groups.product_group.scratch_path, f'temp_{time_stamp}')
    output_dir = cfg.groups.product_group.output_dir

    # RTC-S1 imagery
    save_bursts = cfg.groups.product_group.save_bursts
    save_mosaics = cfg.groups.product_group.save_mosaics
    flag_save_browse = cfg.groups.product_group.save_browse

    if not save_bursts and not save_mosaics:
        err_msg = (f"ERROR either `save_bursts` or `save_mosaics` needs to be"
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

    if cfg.groups.processing.geocoding.algorithm_type == "area_projection":
        geocode_algorithm = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION
    else:
        geocode_algorithm = isce3.geocode.GeocodeOutputMode.INTERP

    memory_mode = geocode_namespace.memory_mode
    geogrid_upsampling = geocode_namespace.geogrid_upsampling
    abs_cal_factor = geocode_namespace.abs_rad_cal
    clip_max = geocode_namespace.clip_max
    clip_min = geocode_namespace.clip_min
    flag_upsample_radar_grid = geocode_namespace.upsample_radargrid
    save_incidence_angle = geocode_namespace.save_incidence_angle
    save_local_inc_angle = geocode_namespace.save_local_inc_angle
    save_projection_angle = geocode_namespace.save_projection_angle
    save_rtc_anf_psi = geocode_namespace.save_rtc_anf_psi
    save_range_slope = geocode_namespace.save_range_slope
    save_nlooks = geocode_namespace.save_nlooks





    # TODO remove the lines below:
    if save_mosaics:
        save_nlooks = True




    save_rtc_anf = geocode_namespace.save_rtc_anf
    save_dem = geocode_namespace.save_dem
    save_layover_shadow_mask = geocode_namespace.save_layover_shadow_mask

    flag_call_radar_grid = (save_incidence_angle or
                            save_local_inc_angle or save_projection_angle or
                            save_rtc_anf_psi or save_dem or
                            save_range_slope)

    # unpack RTC run parameters
    rtc_namespace = cfg.groups.processing.rtc

    # only 2 RTC algorithms supported: area_projection (default) &
    # bilinear_distribution
    if rtc_namespace.algorithm_type == "bilinear_distribution":
        rtc_algorithm = isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION
    else:
        rtc_algorithm = isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION

    output_terrain_radiometry = rtc_namespace.output_type
    input_terrain_radiometry = rtc_namespace.input_terrain_radiometry
    rtc_min_value_db = rtc_namespace.rtc_min_value_db
    rtc_upsampling = rtc_namespace.dem_upsampling
    if (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT):
        output_radiometry_str = "radar backscatter sigma0"
    elif (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT):
        output_radiometry_str = 'radar backscatter gamma0'
    elif input_terrain_radiometry == isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT:
        output_radiometry_str = 'radar backscatter beta0'
    else:
        output_radiometry_str = 'radar backscatter sigma0'

    logger.info(f'Identification:')
    logger.info(f'    processing type: {processing_type}')
    logger.info(f'    product version: {product_version}')
    if save_mosaics:
        logger.info(f'    mosaic product ID: {mosaic_product_id}')
    logger.info(f'Processing parameters:')
    logger.info(f'    apply RTC: {flag_apply_rtc}')
    logger.info(f'    apply thermal noise correction:'
                f' {flag_apply_thermal_noise_correction}')
    logger.info(f'    apply absolute radiometric correction:'
                f' {flag_apply_abs_rad_correction}')
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
    logger.info(f'Browse images:')
    logger.info(f'    burst height: {browse_image_burst_height}')
    logger.info(f'    burst width: {browse_image_burst_width}')
    logger.info(f'    mosaic height: {browse_image_mosaic_height}')
    logger.info(f'    mosaic width: {browse_image_mosaic_width}')

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

    # configure mosaic secondary layers
    add_output_to_output_metadata_dict(
        save_layover_shadow_mask, 'layover_shadow_mask',
        output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)
    add_output_to_output_metadata_dict(
        save_nlooks, 'nlooks', output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)
    add_output_to_output_metadata_dict(
        save_rtc_anf, 'rtc_area_normalization_factor', output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)

    temp_files_list = []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)
    vrt_options_mosaic = gdal.BuildVRTOptions(separate=True)

    n_bursts = len(cfg.bursts.items())
    logger.info(f'Number of bursts to process: {n_bursts}')

    hdf5_mosaic_obj = None
    output_hdf5_file = os.path.join(output_dir,
                                    f'{mosaic_product_id}.{hdf5_file_extension}')

    lookside = None
    wavelength = None
    orbit = None
    mosaic_geotiff_metadata_dict = None

    # iterate over sub-burts
    for burst_index, (burst_id, burst_pol_dict) in enumerate(cfg.bursts.items()):

        # ===========================================================
        # start burst processing

        t_burst_start = time.time()
        logger.info(f'Processing burst: {burst_id} ({burst_index+1}/'
                    f'{n_bursts})')

        burst_id_file_name = burst_id[1:].upper().replace('_', '-')
        burst_product_id = \
            product_id.replace('{burst_id}', burst_id_file_name)

        logger.info(f'    product ID: {burst_product_id}')

        pol_list = list(burst_pol_dict.keys())
        burst = burst_pol_dict[pol_list[0]]

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

        geogrid = cfg.geogrids[burst_id]

        # snap coordinates
        x_snap = geogrid.spacing_x
        y_snap = geogrid.spacing_y
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)

        logger.info(f'    reading burst SLCs')
        radar_grid = burst.as_isce3_radargrid()

        # native_doppler = burst.doppler.lut2d
        orbit = burst.orbit
        wavelength = burst.wavelength
        lookside = radar_grid.lookside

        input_file_list = []

        for pol, burst_pol in burst_pol_dict.items():
            temp_slc_path = \
                f'{burst_scratch_path}/rslc_{pol}.vrt'
            temp_slc_corrected_path = (
                f'{burst_scratch_path}/rslc_{pol}_corrected.{imagery_extension}')

            burst_pol.slc_to_vrt_file(temp_slc_path)

            if (flag_apply_thermal_noise_correction or
                    flag_apply_abs_rad_correction):
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

        # create multi-band VRT
        if len(input_file_list) == 1:
            rdr_burst_raster = isce3.io.Raster(input_file_list[0])
        else:
            temp_vrt_path = f'{burst_scratch_path}/rslc.vrt'
            gdal.BuildVRT(temp_vrt_path, input_file_list,
                          options=vrt_options_mosaic)
            rdr_burst_raster = isce3.io.Raster(temp_vrt_path)
            temp_files_list.append(temp_vrt_path)

        # At this point, burst imagery files are always temporary
        geo_burst_filename = \
            f'{burst_scratch_path}/{burst_product_id}.{imagery_extension}'
        temp_files_list.append(geo_burst_filename)

        # Calculate geolocation correction LUT
        if cfg.groups.processing.apply_correction_luts:
            az_step = cfg.groups.processing.correction_lut_azimuth_spacing
            rg_step = cfg.groups.processing.correction_lut_range_spacing
            # Calculates the LUTs for one polarization in `burst_pol_dict`
            pol_burst_for_lut = next(iter(burst_pol_dict))
            burst_for_lut = burst_pol_dict[pol_burst_for_lut]
            rg_lut, az_lut = compute_correction_lut(burst_for_lut,
                                                    dem_raster,
                                                    burst_scratch_path,
                                                    rg_step, az_step)
        else:
            rg_lut = isce3.core.LUT2d()
            az_lut = isce3.core.LUT2d()

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

        if save_nlooks:
            nlooks_file = (f'{output_dir_sec_bursts}/{burst_product_id}'
                           f'_nlooks.{imagery_extension}')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(nlooks_file)
            else:
                output_file_list.append(nlooks_file)

            out_geo_nlooks_obj = isce3.io.Raster(
                nlooks_file, geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, output_raster_format)
        else:
            nlooks_file = None
            out_geo_nlooks_obj = None

        if save_rtc_anf:
            rtc_anf_file = (f'{output_dir_sec_bursts}/{burst_product_id}'
                            f'_rtc_anf.{imagery_extension}')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(rtc_anf_file)
            else:
                output_file_list.append(rtc_anf_file)

            out_geo_rtc_obj = isce3.io.Raster(
                rtc_anf_file,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, output_raster_format)
        else:
            rtc_anf_file = None
            out_geo_rtc_obj = None

        # geocoding optional arguments (new ISCE3 with unmerged code)
        geocode_new_isce3_kwargs = {}

        # geocoding optional arguments
        geocode_kwargs = {}

        # Calculate layover/shadow mask when requested
        if save_layover_shadow_mask or apply_shadow_masking:
            flag_layover_shadow_mask_is_temporary = \
                (flag_bursts_secondary_files_are_temporary or
                    (apply_shadow_masking and not save_layover_shadow_mask))

            if flag_layover_shadow_mask_is_temporary:
                # layover/shadow mask is temporary
                layover_shadow_mask_file = \
                    (f'{burst_scratch_path}/{burst_product_id}'
                     f'_layover_shadow_mask.{imagery_extension}')
            else:
                # layover/shadow mask is saved in `output_dir_sec_bursts`
                layover_shadow_mask_file = \
                    (f'{output_dir_sec_bursts}/{burst_product_id}'
                     f'_layover_shadow_mask.{imagery_extension}')
            logger.info(f'Computing layover shadow mask for {burst_id}')
            slantrange_layover_shadow_mask_raster = compute_layover_shadow_mask(
                radar_grid,
                orbit,
                geogrid,
                burst,
                dem_raster,
                layover_shadow_mask_file,
                output_raster_format,
                threshold_rdr2geo=cfg.rdr2geo_params.threshold,
                numiter_rdr2geo=cfg.rdr2geo_params.numiter,
                threshold_geo2rdr=cfg.geo2rdr_params.threshold,
                numiter_geo2rdr=cfg.geo2rdr_params.numiter,
                memory_mode=geocode_namespace.memory_mode)

            if flag_layover_shadow_mask_is_temporary:
                temp_files_list.append(layover_shadow_mask_file)
            else:
                output_file_list.append(layover_shadow_mask_file)
                logger.info(f'file saved: {layover_shadow_mask_file}')
                if save_layover_shadow_mask:
                    output_metadata_dict['layover_shadow_mask'][1].append(
                        layover_shadow_mask_file)

            if not save_layover_shadow_mask:
                layover_shadow_mask_file = None

            if apply_shadow_masking:
                geocode_new_isce3_kwargs['input_layover_shadow_mask_raster'] = \
                    slantrange_layover_shadow_mask_raster
        else:
            layover_shadow_mask_file = None

        # flag to run geocoding without shadow masking
        flag_geocoding_without_shadow_masking = False

        # flag to inform the user that there was an error using
        # sub-swath masking
        flag_inform_user_about_isce3_version_error = False

        # get sub_swaths metadata
        if apply_valid_samples_sub_swath_masking:
            # Extract burst boundaries and create sub_swaths object to mask
            # invalid radar samples
            n_subswaths = 1
            sub_swaths = isce3.product.SubSwaths(radar_grid.length,
                                                 radar_grid.width,
                                                 n_subswaths)
            last_range_sample = min([burst.last_valid_sample, radar_grid.width])
            valid_samples_sub_swath = np.repeat(
                [[burst.first_valid_sample, last_range_sample + 1]],
                radar_grid.length, axis=0)
            for i in range(burst.first_valid_line):
                valid_samples_sub_swath[i, :] = 0
            for i in range(burst.last_valid_line, radar_grid.length):
                valid_samples_sub_swath[i, :] = 0

            sub_swaths.set_valid_samples_array(1, valid_samples_sub_swath)
            geocode_new_isce3_kwargs['sub_swaths'] = sub_swaths
            geocode_kwargs['sub_swaths'] = sub_swaths

        if apply_shadow_masking:
            # run ISCE3 geocoding
            try:
                geo_obj.geocode(radar_grid=radar_grid,
                                input_raster=rdr_burst_raster,
                                output_raster=geo_burst_raster,
                                dem_raster=dem_raster,
                                output_mode=geocode_algorithm,
                                geogrid_upsampling=geogrid_upsampling,
                                flag_apply_rtc=flag_apply_rtc,
                                input_terrain_radiometry=input_terrain_radiometry,
                                output_terrain_radiometry=output_terrain_radiometry,
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
                                input_rtc=None,
                                output_rtc=None,
                                dem_interp_method=dem_interp_method_enum,
                                memory_mode=memory_mode,
                                **geocode_new_isce3_kwargs)
            except TypeError:
                flag_geocoding_without_shadow_masking = True
                flag_inform_user_about_isce3_version_error = True
                logger.warning('WARNING there was an error executing geocode().'
                               ' Retrying it with less parameters')

        else:
            flag_geocoding_without_shadow_masking = True

        if flag_geocoding_without_shadow_masking:
            # run ISCE3 geocoding (without shadow masking)
            geo_obj.geocode(radar_grid=radar_grid,
                            input_raster=rdr_burst_raster,
                            output_raster=geo_burst_raster,
                            dem_raster=dem_raster,
                            output_mode=geocode_algorithm,
                            geogrid_upsampling=geogrid_upsampling,
                            flag_apply_rtc=flag_apply_rtc,
                            input_terrain_radiometry=input_terrain_radiometry,
                            output_terrain_radiometry=output_terrain_radiometry,
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
                            input_rtc=None,
                            output_rtc=None,
                            dem_interp_method=dem_interp_method_enum,
                            memory_mode=memory_mode,
                            **geocode_kwargs)

            if flag_inform_user_about_isce3_version_error:
                logger.warning('WARNING shadow masking is not available'
                               f' from installed version of ISCE3. ({isce3.__version__})')

        del geo_burst_raster

        # Output imagery list contains multi-band files that
        # will be used for mosaicking
        output_imagery_list.append(geo_burst_filename)

        # Create burst HDF5
        if (save_hdf5_metadata and save_bursts):
            hdf5_file_output_dir = os.path.join(output_dir, burst_id)
            os.makedirs(hdf5_file_output_dir, exist_ok=True)
            output_hdf5_file_burst = os.path.join(
                hdf5_file_output_dir, f'{burst_product_id}.{hdf5_file_extension}')

        # If burst imagery is not temporary, separate polarization channels
        output_burst_imagery_list = []
        if not flag_bursts_files_are_temporary:
            for pol in pol_list:
                geo_burst_pol_filename = \
                    os.path.join(output_dir_bursts,
                                 f'{burst_product_id}_{pol}.' +
                                 f'{imagery_extension}')
                output_burst_imagery_list.append(geo_burst_pol_filename)

            _separate_pol_channels(geo_burst_filename,
                                   output_burst_imagery_list,
                                   pol_list, output_radiometry_str,
                                   output_raster_format, logger)

            output_file_list += output_burst_imagery_list


        else:
            for pol in pol_list:
                geo_burst_pol_filename = (f'NETCDF:{output_hdf5_file_burst}:'
                                          '/science/SENTINEL1/RTC/grids/'
                                          f'frequencyA/{pol}')
            output_burst_imagery_list.append(geo_burst_pol_filename)


        if save_nlooks:
            del out_geo_nlooks_obj

            if not flag_bursts_secondary_files_are_temporary:
                logger.info(f'file saved: {nlooks_file}')
            output_metadata_dict['nlooks'][1].append(nlooks_file)

        if save_rtc_anf:
            del out_geo_rtc_obj

            if not flag_bursts_secondary_files_are_temporary:
                logger.info(f'file saved: {rtc_anf_file}')
            output_metadata_dict['rtc_area_normalization_factor'][1].append(rtc_anf_file)

        radar_grid_file_dict = {}

        if flag_call_radar_grid and save_bursts:
            get_radar_grid(
                geogrid, dem_interp_method_enum, burst_product_id,
                output_dir_sec_bursts, imagery_extension, save_incidence_angle,
                save_local_inc_angle, save_projection_angle,
                save_rtc_anf_psi,
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
        if (save_hdf5_metadata and save_bursts):
            hdf5_file_output_dir = os.path.join(output_dir, burst_id)
            with create_hdf5_file(burst_product_id,
                    output_hdf5_file_burst, orbit, burst, cfg,
                    is_mosaic=False) as hdf5_burst_obj:
                save_hdf5_file(
                    hdf5_burst_obj, output_hdf5_file_burst, flag_apply_rtc,
                    clip_max, clip_min, output_radiometry_str,
                    geogrid, pol_list, geo_burst_filename, nlooks_file,
                    rtc_anf_file, layover_shadow_mask_file,
                    radar_grid_file_dict,
                    save_imagery=save_imagery_as_hdf5,
                    save_secondary_layers=save_secondary_layers_as_hdf5)
            output_file_list.append(output_hdf5_file_burst)

        # save browse image (burst)
        if flag_save_browse:
            browse_image_filename = \
                os.path.join(output_dir_bursts, f'{burst_product_id}.png')
            save_browse(output_burst_imagery_list, browse_image_filename,
                        pol_list, browse_image_burst_height,
                        browse_image_burst_width, temp_files_list,
                        burst_scratch_path, logger)
            output_file_list.append(browse_image_filename)

        if (not flag_bursts_files_are_temporary or
                save_secondary_layers_as_hdf5):
            metadata_dict = get_metadata_dict(burst_product_id, burst, cfg,
                                              is_mosaic=False)
            geotiff_metadata_dict = all_metadata_dict_to_geotiff_metadata_dict(
                metadata_dict)
            for current_file in output_file_list:
                if not current_file.endswith('.tif'):
                    continue
                append_metadata_to_geotiff_file(current_file,
                                                geotiff_metadata_dict)

        # Create mosaic HDF5
        if (save_hdf5_metadata and save_mosaics
                and burst_index == 0):
            hdf5_mosaic_obj = create_hdf5_file(mosaic_product_id,
                output_hdf5_file, orbit, burst, cfg, is_mosaic=True)
        
        if (save_mosaics and burst_index == 0):
            mosaic_metadata_dict = get_metadata_dict(mosaic_product_id, burst, cfg,
                is_mosaic=True)
            mosaic_geotiff_metadata_dict = all_metadata_dict_to_geotiff_metadata_dict(
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
        get_radar_grid(cfg.geogrid, dem_interp_method_enum, mosaic_product_id,
                       radar_grid_output_dir, imagery_extension, save_incidence_angle,
                       save_local_inc_angle, save_projection_angle,
                       save_rtc_anf_psi,
                       save_range_slope, save_dem,
                       dem_raster, radar_grid_file_dict,
                       lookside, wavelength,
                       orbit, verbose=not flag_bursts_secondary_files_are_temporary)
        if flag_bursts_secondary_files_are_temporary:
            # files are temporary
            temp_files_list += list(radar_grid_file_dict.values())
        else:
            output_file_list += list(radar_grid_file_dict.values())
            mosaic_output_file_list += list(radar_grid_file_dict.values())

    if save_mosaics:

        # Mosaic sub-bursts imagery
        logger.info(f'mosaicking files:')
        output_imagery_filename_list = []
        for pol in pol_list:
            geo_pol_filename = \
                (f'{output_dir_mosaic_raster}/{mosaic_product_id}_{pol}.'
                 f'{imagery_extension}')
            logger.info(f'    {geo_pol_filename}')
            output_imagery_filename_list.append(geo_pol_filename)

        nlooks_list = output_metadata_dict['nlooks'][1]

        if len(output_imagery_list) > 0:

            compute_weighted_mosaic_raster_single_band(
                output_imagery_list, nlooks_list,
                output_imagery_filename_list, cfg.geogrid,
                verbose=False)

        if save_imagery_as_hdf5:
            temp_files_list += output_imagery_filename_list
        else:
            output_file_list += output_imagery_filename_list
            mosaic_output_file_list += output_imagery_filename_list

        # Mosaic other bands
        for key in output_metadata_dict.keys():
            output_file, input_files = output_metadata_dict[key]
            logger.info(f'mosaicking file: {output_file}')
            if len(input_files) == 0:
                continue
            compute_weighted_mosaic_raster(
                input_files, nlooks_list, output_file, cfg.geogrid,
                verbose=False)


            # TODO: Remove nlooks exception below
            if (save_secondary_layers_as_hdf5 or
                    (key == 'nlooks' and not save_nlooks)):
                temp_files_list.append(output_file)
            else:
                output_file_list.append(output_file)
                mosaic_output_file_list.append(output_file)
                


        # save browse image (mosaic)
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
                nlooks_mosaic_file = output_metadata_dict['nlooks'][0]
            else:
                nlooks_mosaic_file = None
            if save_rtc_anf:
                rtc_anf_mosaic_file = output_metadata_dict[
                    'rtc_area_normalization_factor'][0]
            else:
                rtc_anf_mosaic_file = None
            if save_layover_shadow_mask:
                layover_shadow_mask_file = output_metadata_dict[
                    'layover_shadow_mask'][0]
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

            sensing_start_ds = f'{BASE_HDF5_DATASET}/identification/zeroDopplerStartTime'
            sensing_end_ds = f'{BASE_HDF5_DATASET}/identification/zeroDopplerEndTime'
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
                hdf5_mosaic_obj, output_hdf5_file, flag_apply_rtc,
                clip_max, clip_min, output_radiometry_str,
                cfg.geogrid, pol_list, geo_filename_vrt, nlooks_mosaic_file,
                rtc_anf_mosaic_file, layover_shadow_mask_file,
                radar_grid_file_dict, save_imagery=save_imagery_as_hdf5,
                save_secondary_layers=save_secondary_layers_as_hdf5)
            hdf5_mosaic_obj.close()
            output_file_list.append(output_hdf5_file)

    if save_mosaics:
        for current_file in mosaic_output_file_list:
            if not current_file.endswith('.tif'):
                continue
            append_metadata_to_geotiff_file(current_file,
                                            mosaic_geotiff_metadata_dict)

    if output_imagery_format == 'COG':
        logger.info(f'Saving files as Cloud-Optimized GeoTIFFs (COGs)')
        for filename in output_file_list:
            if not filename.endswith('.tif'):
                continue

            logger.info(f'    processing file: {filename}')

            # if file is backscatter, use the 'AVERAGE' mode to create overlays
            options_save_as_cog = {}
            gdal_ds = gdal.Open(filename, gdal.GA_ReadOnly)
            description = gdal_ds.GetRasterBand(1).GetDescription()
            if  description and 'backscatter' in description.lower():
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
                   save_rtc_anf_psi,
                   save_range_slope, save_dem, dem_raster,
                   radar_grid_file_dict, lookside, wavelength, orbit,
                   verbose=True):
    output_obj_list = []
    layers_nbands = 1
    shape = [layers_nbands, geogrid.length, geogrid.width]

    incidence_angle_raster = _create_raster_obj(
        output_dir, f'{product_id}_incidence_angle',
        'incidenceAngle', gdal.GDT_Float32, shape, radar_grid_file_dict,
        output_obj_list, save_incidence_angle, extension)
    local_incidence_angle_raster = _create_raster_obj(
        output_dir, f'{product_id}_local_incidence_angle',
        'localIncidenceAngle', gdal.GDT_Float32, shape,
        radar_grid_file_dict, output_obj_list, save_local_inc_angle,
        extension)
    projection_angle_raster = _create_raster_obj(
        output_dir, f'{product_id}_projection_angle',
        'projectionAngle', gdal.GDT_Float32, shape, radar_grid_file_dict,
        output_obj_list, save_projection_angle, extension)
    rtc_anf_psi_raster = _create_raster_obj(
        output_dir, f'{product_id}_rtc_anf_psi',
        'RTCAreaNormalizationFactorPsi', gdal.GDT_Float32, shape,
        radar_grid_file_dict, output_obj_list,
        save_rtc_anf_psi, extension)
    range_slope_raster = _create_raster_obj(
        output_dir, f'{product_id}_range_slope',
        'rangeSlope', gdal.GDT_Float32, shape, radar_grid_file_dict,
        output_obj_list, save_range_slope, extension)
    interpolated_dem_raster = _create_raster_obj(
        output_dir, f'{product_id}_interpolated_dem',
        'interpolatedDem', gdal.GDT_Float32, shape, radar_grid_file_dict,
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
    isce3.geogrid.get_radar_grid(lookside,
                                 wavelength,
                                 dem_raster,
                                 geogrid,
                                 orbit,
                                 native_doppler,
                                 grid_doppler,
                                 incidence_angle_raster =
                                    incidence_angle_raster,
                                 local_incidence_angle_raster =
                                    local_incidence_angle_raster,
                                 projection_angle_raster =
                                    projection_angle_raster,
                                 simulated_radar_brightness_raster =
                                    rtc_anf_psi_raster,
                                 interpolated_dem_raster =
                                    interpolated_dem_raster,
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
    parser = argparse.ArgumentParser(description='',
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
