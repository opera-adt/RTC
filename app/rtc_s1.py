#!/usr/bin/env python

'''
RTC Workflow
'''

import datetime
import os
import time

import logging
import numpy as np
from osgeo import gdal
import argparse

import isce3

from s1reader.s1_burst_slc import Sentinel1BurstSlc

from rtc.geogrid import snap_coord
from rtc.runconfig import RunConfig
from rtc.mosaic_geobursts import compute_weighted_mosaic_raster, compute_weighted_mosaic_raster_single_band
from rtc.core import create_logger, save_as_cog
from rtc.h5_prep import save_hdf5_file, create_hdf5_file, \
    save_hdf5_dataset, BASE_DS

logger = logging.getLogger('rtc_s1')


def _update_mosaic_boundaries(mosaic_geogrid_dict, geogrid):
    """Updates mosaic boundaries and check if pixel spacing
       and EPSG code are consistent between burst
       and mosaic geogrid

       Parameters
       ----------
       mosaic_geogrid_dict: dict
              Dictionary containing mosaic geogrid parameters
       geogrid : isce3.product.GeoGridParameters
              Burst geogrid

    """
    xf = geogrid.start_x + geogrid.spacing_x * geogrid.width
    yf = geogrid.start_y + geogrid.spacing_y * geogrid.length
    if ('x0' not in mosaic_geogrid_dict.keys() or
            geogrid.start_x < mosaic_geogrid_dict['x0']):
        mosaic_geogrid_dict['x0'] = geogrid.start_x
    if ('xf' not in mosaic_geogrid_dict.keys() or
            xf > mosaic_geogrid_dict['xf']):
        mosaic_geogrid_dict['xf'] = xf
    if ('y0' not in mosaic_geogrid_dict.keys() or
            geogrid.start_y > mosaic_geogrid_dict['y0']):
        mosaic_geogrid_dict['y0'] = geogrid.start_y
    if ('yf' not in mosaic_geogrid_dict.keys() or
            yf < mosaic_geogrid_dict['yf']):
        mosaic_geogrid_dict['yf'] = yf
    if 'dx' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['dx'] = geogrid.spacing_x
    else:
        assert(mosaic_geogrid_dict['dx'] == geogrid.spacing_x)
    if 'dy' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['dy'] = geogrid.spacing_y
    else:
        assert(mosaic_geogrid_dict['dy'] == geogrid.spacing_y)
    if 'epsg' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['epsg'] = geogrid.epsg
    else:
        assert(mosaic_geogrid_dict['epsg'] == geogrid.epsg)


def _separate_pol_channels(multi_band_file, output_file_list, logger,
                           output_raster_format):
    """Save a multi-band raster file as individual single-band files

       Parameters
       ----------
       multi_band_file : str
              Multi-band raster file
       output_file_list : list
              Output file list
       logger : loggin.Logger
              Logger
    """
    gdal_ds = gdal.Open(multi_band_file, gdal.GA_ReadOnly)
    description = gdal_ds.GetDescription()
    projection = gdal_ds.GetProjectionRef()
    geotransform = gdal_ds.GetGeoTransform()
    metadata = gdal_ds.GetMetadata()

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

        raster_out.SetDescription(description)
        raster_out.SetProjection(projection)
        raster_out.SetGeoTransform(geotransform)
        raster_out.SetMetadata(metadata)

        band_out = raster_out.GetRasterBand(1)
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


def _add_output_to_output_metadata_dict(flag, key, output_dir,
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
    back to ENVI format. Preserves the phase.'''

    # Load the SLC of the burst
    burst_in.slc_to_vrt_file(path_slc_vrt)
    slc_gdal_ds = gdal.Open(path_slc_vrt)
    arr_slc_from = slc_gdal_ds.ReadAsArray()

    # Apply the correction
    if flag_thermal_correction:
        logger.info(f'    applying thermal noise correction to burst SLC')
        corrected_image = np.abs(arr_slc_from) ** 2 - burst_in.thermal_noise_lut
        min_backscatter = 0
        max_backscatter = None
        corrected_image = np.clip(corrected_image, min_backscatter,
                                  max_backscatter)
    else:
        corrected_image=np.abs(arr_slc_from) ** 2

    if flag_apply_abs_rad_correction:
        logger.info(f'    applying absolute radiometric correction to burst SLC')
    if flag_output_complex:
        factor_mag = np.sqrt(corrected_image) / np.abs(arr_slc_from)
        factor_mag[np.isnan(factor_mag)] = 0.0
        corrected_image = arr_slc_from * factor_mag
        dtype = gdal.GDT_CFloat32
        if flag_apply_abs_rad_correction:
            corrected_image = \
                corrected_image / burst_in.burst_calibration.beta_naught
    else:
        dtype = gdal.GDT_Float32
        if flag_apply_abs_rad_correction:
            corrected_image = \
                corrected_image / burst_in.burst_calibration.beta_naught ** 2

    # Save the corrected image
    drvout = gdal.GetDriverByName('GTiff')
    raster_out = drvout.Create(path_slc_out, burst_in.shape[1],
                               burst_in.shape[0], 1, dtype)
    band_out = raster_out.GetRasterBand(1)
    band_out.WriteArray(corrected_image)
    band_out.FlushCache()
    del band_out


def calculate_layover_shadow_mask(burst_in: Sentinel1BurstSlc,
                                  geogrid_in: isce3.product.GeoGridParameters,
                                  path_dem: str,
                                  filename_out: str,
                                  output_raster_format: str,
                                  threshold_rdr2geo: float=1.0e-7,
                                  numiter_rdr2geo: int=25,
                                  extraiter_rdr2geo: int=10,
                                  lines_per_block_rdr2geo: int=1000,
                                  threshold_geo2rdr: float=1.0e-8,
                                  numiter_geo2rdr: int=25,
                                  nlooks_az: int=1, nlooks_rg: int=1):
    '''
    Generate the layover shadow mask and geocode the mask

    Parameters:
    -----------
    burst_in: Sentinel1BurstSlc
        Input burst
    geogrid_in: isce3.product.GeoGridParameters
        Geogrid to geocode the layover shadow mask in radar grid
    path_dem: str
        Path to the DEM
    filename_out: str
        Path to the geocoded layover shadow mask
    output_raster_format: str
        File format of the layover shadow mask
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
    nlooks_az: int
        Number of looks in azimuth direction. For the calculation in coarse grid
    nlooks_rg: int
        Number of looks in range direction. For the calculation in coarse grid

    '''

    # determine the output filename
    str_datetime = burst_in.sensing_start.strftime('%Y%m%d_%H%M%S.%f')

    path_layover_shadow_mask = (f'layover_shadow_mask_{burst_in.burst_id}_'
                                f'{burst_in.polarization}_{str_datetime}')
    
    # Run topo to get layover shadow mask
    dem_raster = isce3.io.Raster(path_dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    Rdr2Geo = isce3.geometry.Rdr2Geo

    rdr_grid = burst_in.as_isce3_radargrid()

    # when requested, apply mulitilooking on radar grid for the computation in coarse resolution
    if nlooks_az > 1 or nlooks_rg > 1:
        rdr_grid = rdr_grid.multilook(nlooks_az, nlooks_rg)
    
    isce3_orbit = burst_in.orbit
    grid_doppler = isce3.core.LUT2d()

    rdr2geo_obj = Rdr2Geo(rdr_grid,
                          isce3_orbit,
                          ellipsoid,
                          grid_doppler,
                          threshold=threshold_rdr2geo,
                          numiter=numiter_rdr2geo,
                          extraiter=extraiter_rdr2geo,
                          lines_per_block=lines_per_block_rdr2geo)

    mask_raster = isce3.io.Raster(path_layover_shadow_mask, rdr_grid.width,
                                  rdr_grid.length, 1, gdal.GDT_Byte, 'MEM')

    rdr2geo_obj.topo(dem_raster, None, None, None,
                     layover_shadow_raster=mask_raster)
    
    # geocode the layover shadow mask
    geo = isce3.geocode.GeocodeFloat32()
    geo.orbit = isce3_orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = grid_doppler
    geo.threshold_geo2rdr = threshold_geo2rdr
    geo.numiter_geo2rdr = numiter_geo2rdr
    geo.lines_per_block = lines_per_block_rdr2geo
    geo.data_interpolator = 'NEAREST'
    geo.geogrid(float(geogrid_in.start_x),
                float(geogrid_in.start_y),
                float(geogrid_in.spacing_x),
                float(geogrid_in.spacing_y),
                int(geogrid_in.width),
                int(geogrid_in.length),
                int(geogrid_in.epsg))

    geocoded_raster = isce3.io.Raster(filename_out, 
                                      geogrid_in.width, geogrid_in.length, 1,
                                      gdal.GDT_Byte, output_raster_format)

    geo.geocode(radar_grid=rdr_grid,
                input_raster=mask_raster,
                output_raster=geocoded_raster,
                dem_raster=dem_raster,
                output_mode=isce3.geocode.GeocodeOutputMode.INTERP)


def run(cfg: RunConfig):
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
    logger.info("Starting the RTC-S1 Science Application Software (SAS)")

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
    product_id = cfg.groups.product_path_group.product_id
    if product_id is None:
        product_id = 'rtc_product'

    scratch_path = os.path.join(
        cfg.groups.product_path_group.scratch_path, f'temp_{time_stamp}')
    output_dir = cfg.groups.product_path_group.output_dir

    # RTC-S1 imagery
    save_bursts = cfg.groups.product_path_group.save_bursts
    save_mosaics = cfg.groups.product_path_group.save_mosaics

    if not save_bursts and not save_mosaics:
        err_msg = (f"ERROR either `save_bursts` or `save_mosaics` needs to be"
                   " set")
        raise ValueError(err_msg)

    output_imagery_format = \
        cfg.groups.product_path_group.output_imagery_format
    output_imagery_compression = \
        cfg.groups.product_path_group.output_imagery_compression
    output_imagery_nbits = \
        cfg.groups.product_path_group.output_imagery_nbits

    logger.info(f'Processing parameters:')
    logger.info(f'    apply RTC: {flag_apply_rtc}')
    logger.info(f'    apply thermal noise correction:'
                f' {flag_apply_thermal_noise_correction}')
    logger.info(f'    apply absolute radiometric correction:'
                f' {flag_apply_abs_rad_correction}')
    logger.info(f'    product ID: {product_id}')
    logger.info(f'    scratch dir: {scratch_path}')
    logger.info(f'    output dir: {output_dir}')
    logger.info(f'    save bursts: {save_bursts}')
    logger.info(f'    save mosaics: {save_mosaics}')
    logger.info(f'    output imagery format: {output_imagery_format}')
    logger.info(f'    output imagery compression: {output_imagery_compression}')
    logger.info(f'    output imagery nbits: {output_imagery_nbits}')

    save_imagery_as_hdf5 = (output_imagery_format == 'HDF5' or
                            output_imagery_format == 'NETCDF')
    save_metadata = cfg.groups.product_path_group.save_metadata

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
    save_range_slope = \
        geocode_namespace.save_range_slope
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

    # Common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    zero_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    maxiter = cfg.geo2rdr_params.numiter
    exponent = 1 if (flag_apply_thermal_noise_correction or
                     flag_apply_abs_rad_correction) else 2

    # output mosaics
    geo_filename = f'{output_dir}/'f'{product_id}.{imagery_extension}'
    output_imagery_list = []
    output_file_list = []
    output_metadata_dict = {}

    if save_imagery_as_hdf5:
        output_dir_mosaic_raster = scratch_path
    else:
        output_dir_mosaic_raster = output_dir

    _add_output_to_output_metadata_dict(
        save_layover_shadow_mask, 'layover_shadow_mask',
        output_dir_mosaic_raster,
        output_metadata_dict, product_id, imagery_extension)
    _add_output_to_output_metadata_dict(
        save_nlooks, 'nlooks', output_dir_mosaic_raster,
        output_metadata_dict, product_id, imagery_extension)
    _add_output_to_output_metadata_dict(
        save_rtc_anf, 'rtc', output_dir_mosaic_raster,
        output_metadata_dict, product_id, imagery_extension)

    mosaic_geogrid_dict = {}
    temp_files_list = []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)
    vrt_options_mosaic = gdal.BuildVRTOptions(separate=True)

    n_bursts = len(cfg.bursts.items())
    print('Number of bursts to process:', n_bursts)

    hdf5_obj = None
    output_hdf5_file = os.path.join(output_dir,
                                    f'{product_id}.{hdf5_file_extension}')
    # iterate over sub-burts
    for burst_index, (burst_id, burst_pol_dict) in enumerate(cfg.bursts.items()):
        
        # ===========================================================
        # start burst processing

        t_burst_start = time.time()
        logger.info(f'Processing burst: {burst_id} ({burst_index+1}/'
                    f'{n_bursts})')

        pol_list = list(burst_pol_dict.keys())
        burst = burst_pol_dict[pol_list[0]]

        flag_bursts_files_are_temporary = (not save_bursts or
                                           save_imagery_as_hdf5)

        burst_scratch_path = f'{scratch_path}/{burst_id}/'
        os.makedirs(burst_scratch_path, exist_ok=True)

        if not save_bursts:
            # burst files are saved in scratch dir
            bursts_output_dir = burst_scratch_path
        else:
            # burst files (individual or HDF5) are saved in burst_id dir 
            bursts_output_dir = os.path.join(output_dir, burst_id)
            os.makedirs(bursts_output_dir, exist_ok=True)
        
        geogrid = cfg.geogrids[burst_id]

        # snap coordinates
        x_snap = geogrid.spacing_x
        y_snap = geogrid.spacing_y
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)

        # update mosaic boundaries
        _update_mosaic_boundaries(mosaic_geogrid_dict, geogrid)

        logger.info(f'    reading burst SLCs')
        radar_grid = burst.as_isce3_radargrid()
        # native_doppler = burst.doppler.lut2d
        orbit = burst.orbit
        if 'orbit' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['orbit'] = orbit
        if 'wavelength' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['wavelength'] = burst.wavelength
        if 'lookside' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['lookside'] = radar_grid.lookside

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
                    flag_thermal_correction =
                        flag_apply_thermal_noise_correction,
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
            f'{burst_scratch_path}/{product_id}.{imagery_extension}'
        temp_files_list.append(geo_burst_filename)

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
            nlooks_file = (f'{bursts_output_dir}/{product_id}'
                           f'_nlooks.{imagery_extension}')
            if flag_bursts_files_are_temporary:
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
            rtc_anf_file = (f'{bursts_output_dir}/{product_id}'
               f'_rtc_anf.{imagery_extension}')
            if flag_bursts_files_are_temporary:
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

        # geocode
        flag_error_sub_swaths = False
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
                            clip_min = clip_min,
                            clip_max = clip_max,
                            # out_off_diag_terms=out_off_diag_terms_obj,
                            out_geo_nlooks=out_geo_nlooks_obj,
                            out_geo_rtc=out_geo_rtc_obj,
                            input_rtc=None,
                            output_rtc=None,
                            dem_interp_method=dem_interp_method_enum,
                            memory_mode=memory_mode,
                            sub_swaths=sub_swaths)
        except TypeError:
            flag_error_sub_swaths = True
            logger.warning('WARNING there was an error executing geocode().'
                           ' Retrying it with less parameters')

            # geocode (without sub_swaths)
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
                            clip_min = clip_min,
                            clip_max = clip_max,
                            # out_off_diag_terms=out_off_diag_terms_obj,
                            out_geo_nlooks=out_geo_nlooks_obj,
                            out_geo_rtc=out_geo_rtc_obj,
                            input_rtc=None,
                            output_rtc=None,
                            dem_interp_method=dem_interp_method_enum,
                            memory_mode=memory_mode)

        if flag_error_sub_swaths:
            logger.warning('WARNING the sub-swath masking is not available'
                           ' from this ISCE3 version. The sub-swath masking'
                           ' was disabled.')

        # Calculate layover shadow mask when requested
        if save_layover_shadow_mask:
            layover_shadow_mask_file = (f'{bursts_output_dir}/{product_id}'
               f'_layover_shadow_mask.{imagery_extension}')
            calculate_layover_shadow_mask(burst,
                                geogrid,
                                cfg.dem,
                                layover_shadow_mask_file,
                                output_raster_format,
                                threshold_rdr2geo=cfg.rdr2geo_params.threshold,
                                numiter_rdr2geo=cfg.rdr2geo_params.numiter,
                                threshold_geo2rdr=cfg.geo2rdr_params.threshold,
                                numiter_geo2rdr=cfg.geo2rdr_params.numiter)
            
            if flag_bursts_files_are_temporary:
                temp_files_list.append(layover_shadow_mask_file)
            else:
                output_file_list.append(layover_shadow_mask_file)
                logger.info(f'file saved: {layover_shadow_mask_file}')
            output_metadata_dict['layover_shadow_mask'][1].append(
                layover_shadow_mask_file)

        else:
            layover_shadow_mask_file = None

        del geo_burst_raster

        # Output imagery list contains multi-band files that
        # will be used for mosaicking
        output_imagery_list.append(geo_burst_filename)

        # If burst imagery is not temporary, separate polarization channels
        if not flag_bursts_files_are_temporary:
            output_burst_imagery_list = []
            for pol in pol_list:
                geo_burst_pol_filename = \
                    (f'{output_dir}/{burst_id}/{product_id}_{pol}.'
                     f'{imagery_extension}')
                output_burst_imagery_list.append(geo_burst_pol_filename)

            _separate_pol_channels(geo_burst_filename,
                                   output_burst_imagery_list,
                                   logger, output_raster_format)

            output_file_list += output_burst_imagery_list

        if save_nlooks:
            del out_geo_nlooks_obj
            if not flag_bursts_files_are_temporary:
                logger.info(f'file saved: {nlooks_file}')
            output_metadata_dict['nlooks'][1].append(nlooks_file)
    
        if save_rtc_anf:
            del out_geo_rtc_obj
            if not flag_bursts_files_are_temporary:
                logger.info(f'file saved: {rtc_anf_file}')
            output_metadata_dict['rtc'][1].append(rtc_anf_file)

        radar_grid_file_dict = {}
        if flag_call_radar_grid and save_bursts:
            get_radar_grid(
                geogrid, dem_interp_method_enum, product_id,
                bursts_output_dir, imagery_extension, save_incidence_angle,
                save_local_inc_angle, save_projection_angle,
                save_rtc_anf_psi,
                save_range_slope, save_dem,
                dem_raster, radar_grid_file_dict,
                mosaic_geogrid_dict, orbit,
                verbose = not flag_bursts_files_are_temporary)
            if save_imagery_as_hdf5:
                # files are temporary
                temp_files_list += list(radar_grid_file_dict.values())
            else:
                output_file_list += list(radar_grid_file_dict.values())

        # Create burst HDF5
        if ((save_imagery_as_hdf5 or save_metadata) and save_bursts):
            hdf5_file_output_dir = os.path.join(output_dir, burst_id)
            os.makedirs(hdf5_file_output_dir, exist_ok=True)
            output_hdf5_file_burst =  os.path.join(
                hdf5_file_output_dir, f'{product_id}.{hdf5_file_extension}')
            hdf5_obj = create_hdf5_file(output_hdf5_file_burst, orbit, burst, cfg)
            save_hdf5_file(
                hdf5_obj, output_hdf5_file_burst, flag_apply_rtc,
                clip_max, clip_min, output_radiometry_str, output_file_list,
                geogrid, pol_list, geo_burst_filename, nlooks_file,
                rtc_anf_file, layover_shadow_mask_file,
                radar_grid_file_dict,
                save_imagery = save_imagery_as_hdf5)
            output_file_list.append(output_hdf5_file_burst)

        # Create mosaic HDF5 
        if ((save_imagery_as_hdf5 or save_metadata) and save_mosaics
                and burst_index == 0):
            hdf5_obj = create_hdf5_file(output_hdf5_file, orbit, burst, cfg)

        t_burst_end = time.time()
        logger.info(
            f'elapsed time (burst): {t_burst_end - t_burst_start}')

        # end burst processing
        # ===========================================================

    if flag_call_radar_grid and save_mosaics:
        radar_grid_file_dict = {}

        if save_imagery_as_hdf5:
            radar_grid_output_dir = scratch_path
        else:
            radar_grid_output_dir = output_dir
        get_radar_grid(cfg.geogrid, dem_interp_method_enum, product_id,
                       radar_grid_output_dir, imagery_extension, save_incidence_angle,
                       save_local_inc_angle, save_projection_angle,
                       save_rtc_anf_psi,
                       save_range_slope, save_dem,
                       dem_raster, radar_grid_file_dict,
                       mosaic_geogrid_dict,
                       orbit, verbose = not save_imagery_as_hdf5)
        if save_imagery_as_hdf5:
            # files are temporary
            temp_files_list += list(radar_grid_file_dict.values())
        else:
            output_file_list += list(radar_grid_file_dict.values())

    if save_mosaics:

        # Mosaic sub-bursts imagery
        logger.info(f'mosaicking files:')
        output_imagery_filename_list = []
        for pol in pol_list:
            geo_pol_filename = \
                (f'{output_dir_mosaic_raster}/{product_id}_{pol}.'
                 f'{imagery_extension}')
            logger.info(f'    {geo_pol_filename}')
            output_imagery_filename_list.append(geo_pol_filename)

        # geo_filename = f'{output_dir_mosaic_raster}/{product_id}.{imagery_extension}'

        nlooks_list = output_metadata_dict['nlooks'][1]
        compute_weighted_mosaic_raster_single_band(
            output_imagery_list, nlooks_list,
            output_imagery_filename_list, cfg.geogrid, verbose=False)

        if save_imagery_as_hdf5:
            temp_files_list += output_imagery_filename_list
        else:
            output_file_list += output_imagery_filename_list

        # Mosaic other bands
        for key in output_metadata_dict.keys():
            output_file, input_files = output_metadata_dict[key]
            logger.info(f'mosaicking file: {output_file}')
            compute_weighted_mosaic_raster(input_files, nlooks_list, output_file,
                            cfg.geogrid, verbose=False)
            if save_imagery_as_hdf5:
                temp_files_list.append(output_file)
            else:
                output_file_list.append(output_file)

        # Save HDF5
        if save_imagery_as_hdf5 or save_metadata:
            if save_nlooks:
                nlooks_mosaic_file = output_metadata_dict['nlooks'][0]
            else:
                nlooks_mosaic_file = None
            if save_rtc_anf:
                rtc_anf_mosaic_file = output_metadata_dict['rtc'][0]
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

            sensing_start_ds = f'{BASE_DS}/identification/zeroDopplerStartTime'
            sensing_end_ds = f'{BASE_DS}/identification/zeroDopplerEndTime'
            del hdf5_obj[sensing_start_ds]
            del hdf5_obj[sensing_end_ds]
            hdf5_obj[sensing_start_ds] = \
                sensing_start.strftime('%Y-%m-%dT%H:%M:%S.%f')
            hdf5_obj[sensing_end_ds] = \
                sensing_stop.strftime('%Y-%m-%dT%H:%M:%S.%f')

            save_hdf5_file(hdf5_obj, output_hdf5_file, flag_apply_rtc,
                           clip_max, clip_min, output_radiometry_str,
                           output_file_list, cfg.geogrid, pol_list,
                           geo_filename, nlooks_mosaic_file,
                           rtc_anf_mosaic_file, layover_shadow_mask_file,
                           radar_grid_file_dict,
                           save_imagery = save_imagery_as_hdf5)
            output_file_list.append(output_hdf5_file_burst)

    if output_imagery_format == 'COG':
        logger.info(f'Saving files as Cloud-Optimized GeoTIFFs (COGs)')
        for filename in output_file_list:
            if not filename.endswith('.tif'):
                continue
            logger.info(f'    processing file: {filename}')
            save_as_cog(filename, scratch_path, logger,
                        compression=output_imagery_compression,
                        nbits=output_imagery_nbits)

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



def get_radar_grid(geogrid, dem_interp_method_enum, product_id,
                   output_dir, extension, save_incidence_angle,
                   save_local_inc_angle, save_projection_angle,
                   save_rtc_anf_psi,
                   save_range_slope, save_dem, dem_raster,
                   radar_grid_file_dict, mosaic_geogrid_dict, orbit,
                   verbose = True):
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
            'areaNormalizationFactorPsi', gdal.GDT_Float32, shape,
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
        kwargs_get_radar_grid['directional_slope_angle_raster'] = \
            range_slope_raster

    # call get_radar_grid()
    isce3.geogrid.get_radar_grid(mosaic_geogrid_dict['lookside'],
                                 mosaic_geogrid_dict['wavelength'],
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


def _load_parameters(cfg):
    '''
    Load GCOV specific parameters.
    '''

    geocode_namespace = cfg.groups.processing.geocoding
    rtc_namespace = cfg.groups.processing.rtc

    if geocode_namespace.clip_max is None:
        geocode_namespace.clip_max = np.nan

    if geocode_namespace.clip_min is None:
        geocode_namespace.clip_min = np.nan

    if geocode_namespace.geogrid_upsampling is None:
        geocode_namespace.geogrid_upsampling = 1.0

    if geocode_namespace.memory_mode == 'single_block':
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.SingleBlock
    elif geocode_namespace.memory_mode == 'geogrid':
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.BlocksGeogrid
    elif geocode_namespace.memory_mode == 'geogrid_and_radargrid':
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.BlocksGeogridAndRadarGrid
    elif (geocode_namespace.memory_mode == 'auto' or
          geocode_namespace.memory_mode is None):
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.Auto
    else:
        err_msg = f"ERROR memory_mode: {geocode_namespace.memory_mode}"
        raise ValueError(err_msg)

    rtc_output_type = rtc_namespace.output_type
    if rtc_output_type == 'sigma0':
        rtc_namespace.output_type = \
            isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
    else:
        rtc_namespace.output_type = \
            isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT

    if rtc_namespace.input_terrain_radiometry == "sigma0":
        rtc_namespace.input_terrain_radiometry = \
            isce3.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
    else:
        rtc_namespace.input_terrain_radiometry = \
            isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

    if rtc_namespace.rtc_min_value_db is None:
        rtc_namespace.rtc_min_value_db = np.nan

    # Update the DEM interpolation method
    dem_interp_method = \
        cfg.groups.processing.dem_interpolation_method

    if dem_interp_method == 'biquintic':
        dem_interp_method_enum = isce3.core.DataInterpMethod.BIQUINTIC
    elif (dem_interp_method == 'sinc'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.SINC
    elif (dem_interp_method == 'bilinear'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.BILINEAR
    elif (dem_interp_method == 'bicubic'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.BICUBIC
    elif (dem_interp_method == 'nearest'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.NEAREST
    else:
        err_msg = ('ERROR invalid DEM interpolation method:'
                   f' {dem_interp_method}')
        raise ValueError(err_msg)

    cfg.groups.processing.dem_interpolation_method_enum = \
        dem_interp_method_enum


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


if __name__ == "__main__":
    '''Run geocode rtc workflow from command line'''
    # load arguments from command line
    parser  = get_rtc_s1_parser()
    
    # parse arguments
    args = parser.parse_args()

    # create logger
    create_logger(args.log_file, args.full_log_formatting)

    # Get a runconfig dict from command line argumens
    cfg = RunConfig.load_from_yaml(args.run_config_path, 'rtc_s1')

    _load_parameters(cfg)

    # Run geocode burst workflow
    run(cfg)
