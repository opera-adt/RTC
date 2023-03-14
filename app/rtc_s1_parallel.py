#!/usr/bin/env python

'''
Parallel execution of RTC Workflow
'''

from itertools import repeat
import logging
import multiprocessing
import os
import time
import yaml

import isce3
import numpy as np
from osgeo import gdal
import rtc_s1

from rtc.runconfig import RunConfig


logger = logging.getLogger('rtc_s1')


def split_runconfig(cfg_in, output_dir_child, scratch_path_child=None):
    '''
    Split the input runconfig into single burst runconfigs.
    Writes out the runconfigs.
    Return the list of the burst runconfigs.

    Parameters
    ----------
    path_runconfig_in: str
        Path to the original runconfig
    path_log_in: str
        Path to the original logfile

    Returns
    -------
    list_runconfig_burst: list(str)
        List of the burst runconfigs
    list_logfile_burst: list(str)
        List of the burst logfiles,
        which corresponds to `list_runconfig_burst`
    path_output: str
        Path to the output directory,
        which will be the temporary dir. of the following process

    # TODO revise docstring

    '''

    with open(cfg_in.run_config_path, 'r+', encoding='utf8') as fin:
        runconfig_dict_in = yaml.safe_load(fin.read())

    list_runconfig_burst = []

    # determine the bursts to process
    list_burst_id = cfg_in.bursts.keys()

    # determine the scratch path for the child process
    
    if not scratch_path_child:
        scratch_path_child = \
            os.path.join(cfg_in.groups.product_group.scratch_path,
                        f'{os.path.basename(output_dir_child)}_child_scratch')

    # determine the output directory for child process
    for burst_id in list_burst_id:
        path_temp_runconfig = os.path.join(cfg_in.scratch_path,
                                           f'burst_runconfig_{burst_id}.yaml')

        runconfig_dict_out = runconfig_dict_in.copy()
        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'input_file_group',
                                 'burst_id'],
                                [burst_id])

        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'processing',
                                 'geocoding',
                                 'memory_mode'],
                                'single_block')

        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'output_dir'],
                                output_dir_child)

        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'scratch_path'],
                                scratch_path_child)

        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'save_mosaics'],
                                False)
        
        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'output_imagery_format'],
                                'GTiff')

        # NOTE: This should not be forced off in order to make the burst HDF5 writeout to happen in child process level.
        #set_dict_item_recursive(runconfig_dict_out,
        #                        ['runconfig',
        #                         'groups',
        #                         'product_group',
        #                         'save_secondary_layers_as_hdf5'],
        #                        False)


        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'save_bursts'],
                                True)

        # TODO: Remove the code below once the mosaicking algorithm does not take nlooks as the weight input
        if cfg_in.groups.product_group.save_mosaics:
            set_dict_item_recursive(runconfig_dict_out,
                                    ['runconfig',
                                     'groups',
                                     'processing',
                                     'geocoding',
                                     'save_nlooks'],
                                    True)

        list_runconfig_burst.append(path_temp_runconfig)

        with open(path_temp_runconfig, 'w+', encoding='utf8') as fout:
            yaml.dump(runconfig_dict_out, fout)

    return list_runconfig_burst


def set_dict_item_recursive(dict_in, list_path, val):
    '''
    - Recursively locate the dict item in the multiple layer of dict,
      whose path is provided in the list of keys.
    - Add or update the value of the located item
    - Create the key with empty dict when the key does not exist

    Parameters
    ----------
    dict_in: dict
        Dict to set the value
    list_path:
        Path to the item in the multiple layer of dict
    val:
        Value to add or set
    '''

    if len(list_path) == 1:
        key_in = list_path[0]
        dict_in[key_in] = val
        return

    key_next = list_path[0]
    if not key_next in dict_in.keys():
        dict_in[key_next] = {}
    set_dict_item_recursive(dict_in[key_next], list_path[1:], val)


def process_child_runconfig(path_runconfig_burst,
                            path_logfile = None,
                            full_logfile_format = None,
                            keep_burst_runconfig=False):
    '''
    single worker to process runconfig from terminal using `subprocess`

    Parameters
    ----------
    path_runconfig_burst: str
        Path to the burst runconfig
    path_logfile_burst: str
        Path to the burst logfile
    full_log_format: bool
        Enable full formatting of log messages.
        See `get_rtc_s1_parser()`

    '''
    # Get a runconfig dict from command line argumens
    cfg = RunConfig.load_from_yaml(path_runconfig_burst, 'rtc_s1')

    rtc_s1._load_parameters(cfg)

    rtc_s1.create_logger(path_logfile, full_logfile_format)

    # Run geocode burst workflow
    rtc_s1.run(cfg)

    # Remove or keep (for debugging purpose) the processed runconfig
    if keep_burst_runconfig:
        os.remove(path_runconfig_burst)


def run_parallel(cfg: RunConfig):
    '''
    Parallel version of `rtc_s1.run()`
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
                f' v{rtc_s1.SOFTWARE_VERSION}')

    # primary executable
    processing_type = cfg.groups.product_group.processing_type
    product_version_float = cfg.groups.product_group.product_version
    if product_version_float is None:
        product_version = rtc_s1.SOFTWARE_VERSION
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
    product_id = cfg.groups.product_group.product_id
    if product_id is None:
        product_id = 'rtc_product'
    product_prefix = f'{product_id}_v{product_version}'

    scratch_path = os.path.join(
        cfg.groups.product_group.scratch_path, f'temp_{time_stamp}')

    output_dir = cfg.groups.product_group.output_dir

    # RTC-S1 imagery
    save_bursts = cfg.groups.product_group.save_bursts
    save_mosaics = cfg.groups.product_group.save_mosaics

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

    logger.info(f'Identification:')
    logger.info(f'    product ID: {product_id}')
    logger.info(f'    processing type: {processing_type}')
    logger.info(f'    product version: {product_version}')
    logger.info(f'    product prefix: {product_prefix}')
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
    logger.info(f'    output imagery format: {output_imagery_format}')
    logger.info(f'    output imagery compression: {output_imagery_compression}')
    logger.info(f'    output imagery nbits: {output_imagery_nbits}')

    save_imagery_as_hdf5 = (output_imagery_format == 'HDF5' or
                            output_imagery_format == 'NETCDF')
    save_secondary_layers_as_hdf5 = \
        cfg.groups.product_group.save_secondary_layers_as_hdf5

    save_metadata = (cfg.groups.product_group.save_metadata or
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

    # TODO remove the lines below when the mosaic
    #       does not take in nlooks anymore
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

    # output mosaics variables
    geo_filename = f'{output_dir}/'f'{product_prefix}.{imagery_extension}'
    output_imagery_list = []
    output_file_list = []
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
    rtc_s1._add_output_to_output_metadata_dict(
        save_layover_shadow_mask, 'layover_shadow_mask',
        output_dir_sec_mosaic_raster,
        output_metadata_dict, product_prefix, imagery_extension)
    rtc_s1._add_output_to_output_metadata_dict(
        save_nlooks, 'nlooks', output_dir_sec_mosaic_raster,
        output_metadata_dict, product_prefix, imagery_extension)
    rtc_s1._add_output_to_output_metadata_dict(
        save_rtc_anf, 'rtc', output_dir_sec_mosaic_raster,
        output_metadata_dict, product_prefix, imagery_extension)

    mosaic_geogrid_dict = {}
    temp_files_list = []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)
    vrt_options_mosaic = gdal.BuildVRTOptions(separate=True)

    n_bursts = len(cfg.bursts.items())
    print('Number of bursts to process:', n_bursts)

    hdf5_mosaic_obj = None
    output_hdf5_file = os.path.join(output_dir,
                                    f'{product_prefix}.{hdf5_file_extension}')



    # ------ Start parallelized burst processing ------



    #list_burst_runconfig = split_runconfig(cfg, scratch_path)
    if not save_bursts:
        # burst files are saved in scratch dir
        #bursts_output_dir = scratch_path
        #bursts_output_dir = burst_scratch_path
        list_burst_runconfig = split_runconfig(cfg, scratch_path, None)
    else:
        # burst files (individual or HDF5) are saved in burst_id dir
        #bursts_output_dir = os.path.join(output_dir, burst_id)
        #os.makedirs(bursts_output_dir, exist_ok=True)
        
        list_burst_runconfig = split_runconfig(cfg, output_dir, scratch_path)

    #list_burst_runconfig = split_runconfig(cfg, bursts_output_dir, scratch_path)




    # extract the logger setting from the logger
    path_logger_parent, flag_logger_full_format = get_parent_logger_setting(logger)

    # determine the number of the processors here
    num_workers = cfg.groups.processing.num_workers

    if num_workers == 0:
        # Decide the number of workers automatically
        ncpu_system = os.cpu_count()
        omp_num_threads = os.getenv('OMP_NUM_THREADS')
        if omp_num_threads:
            num_workers = min(ncpu_system,
                              omp_num_threads,
                              len(list_burst_runconfig))
        else:
            num_workers = min(ncpu_system,
                              len(list_burst_runconfig))

    # Execute the single burst processes using multiprocessing
    with multiprocessing.Pool(num_workers) as p:
        p.starmap(process_child_runconfig,
                  zip(list_burst_runconfig,
                      repeat(f'{path_logger_parent}.child'),
                      repeat(flag_logger_full_format)
                      )
                  )

    # ------  End of parallelized burst processing ------


    # NOTE: The following routines assume that the
    #       parallelized burst processing was successful.
    















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
        geogrid.start_x = rtc_s1.snap_coord(geogrid.start_x, x_snap, np.floor)
        geogrid.start_y = rtc_s1.snap_coord(geogrid.start_y, y_snap, np.ceil)

        # update mosaic boundaries
        rtc_s1._update_mosaic_boundaries(mosaic_geogrid_dict, geogrid)

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

            #burst_pol.slc_to_vrt_file(temp_slc_path)

            if (flag_apply_thermal_noise_correction or
                    flag_apply_abs_rad_correction):
            #    apply_slc_corrections(
            #        burst_pol,
            #        temp_slc_path,
            #        temp_slc_corrected_path,
            #        flag_output_complex=False,
            #        flag_thermal_correction =
            #            flag_apply_thermal_noise_correction,
            #        flag_apply_abs_rad_correction=True)
                input_burst_filename = temp_slc_corrected_path
                temp_files_list.append(temp_slc_corrected_path)
            else:
                input_burst_filename = temp_slc_path

            temp_files_list.append(temp_slc_path)
            input_file_list.append(input_burst_filename)

        # create multi-band VRT
        #if len(input_file_list) == 1:
        #    rdr_burst_raster = isce3.io.Raster(input_file_list[0])
        #else:
        #    temp_vrt_path = f'{burst_scratch_path}/rslc.vrt'
        #    gdal.BuildVRT(temp_vrt_path, input_file_list,
        #                options=vrt_options_mosaic)
        #    rdr_burst_raster = isce3.io.Raster(temp_vrt_path)
        #    temp_files_list.append(temp_vrt_path)

        # At this point, burst imagery files are always temporary
        geo_burst_filename = \
            f'{burst_scratch_path}/{product_prefix}.{imagery_extension}'
        temp_files_list.append(geo_burst_filename)

        # Generate output geocoded burst raster        
        #geo_burst_raster = isce3.io.Raster(
        #    geo_burst_filename,
        #    geogrid.width, geogrid.length,
        #    rdr_burst_raster.num_bands, gdal.GDT_Float32,
        #    output_raster_format)

        # init Geocode object depending on raster type
        #if rdr_burst_raster.datatype() == gdal.GDT_Float32:
        #    geo_obj = isce3.geocode.GeocodeFloat32()
        #elif rdr_burst_raster.datatype() == gdal.GDT_Float64:
        #    geo_obj = isce3.geocode.GeocodeFloat64()
        #elif rdr_burst_raster.datatype() == gdal.GDT_CFloat32:
        #    geo_obj = isce3.geocode.GeocodeCFloat32()
        #elif rdr_burst_raster.datatype() == gdal.GDT_CFloat64:
        #    geo_obj = isce3.geocode.GeocodeCFloat64()
        #else:
        #    err_str = 'Unsupported raster type for geocoding'
        #    raise NotImplementedError(err_str)

        # init geocode members
        #geo_obj.orbit = orbit
        #geo_obj.ellipsoid = ellipsoid
        #geo_obj.doppler = zero_doppler
        #geo_obj.threshold_geo2rdr = threshold
        #geo_obj.numiter_geo2rdr = maxiter

        # set data interpolator based on the geocode algorithm
        #if geocode_algorithm == isce3.geocode.GeocodeOutputMode.INTERP:
        #    geo_obj.data_interpolator = geocode_algorithm

        #geo_obj.geogrid(geogrid.start_x, geogrid.start_y,
        #                geogrid.spacing_x, geogrid.spacing_y,
        #                geogrid.width, geogrid.length, geogrid.epsg)

        burst_hdf5_in_scratch = os.path.join(scratch_path,
                                             burst_id,
                                             f'{product_prefix}.{hdf5_file_extension}')

        burst_hdf5_in_output = burst_hdf5_in_scratch.replace(scratch_path,
                                                             output_dir)

        if save_nlooks:
            nlooks_file = (f'{output_dir_sec_bursts}/{product_prefix}'
                           f'_nlooks.{imagery_extension}')

            if save_secondary_layers_as_hdf5:
                nlooks_file = (f'NETCDF:"{burst_hdf5_in_output}":'
                                '/science/SENTINEL1/RTC/grids/frequencyA/'
                                'numberOfLooks')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(nlooks_file)
            else:
                output_file_list.append(nlooks_file)

            #out_geo_nlooks_obj = isce3.io.Raster(
            #    nlooks_file, geogrid.width, geogrid.length, 1,
            #    gdal.GDT_Float32, output_raster_format)
        else:
            nlooks_file = None
            #out_geo_nlooks_obj = None

        if save_rtc_anf:
            rtc_anf_file = (f'{output_dir_sec_bursts}/{product_prefix}'
               f'_rtc_anf.{imagery_extension}')
            
            if save_secondary_layers_as_hdf5:
                rtc_anf_file = (f'NETCDF:"{burst_hdf5_in_output}":'
                                '/science/SENTINEL1/RTC/grids/frequencyA/'
                                #'areaNormalizationFactorPsi') # TODO discuss about the correct path tothe ANF
                                'areaNormalizationFactor')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(rtc_anf_file)
            else:
                output_file_list.append(rtc_anf_file)

            #out_geo_rtc_obj = isce3.io.Raster(
            #    rtc_anf_file,
            #    geogrid.width, geogrid.length, 1,
            #    gdal.GDT_Float32, output_raster_format)
        else:
            rtc_anf_file = None
            #out_geo_rtc_obj = None

        # geocoding optional arguments (new ISCE3 with unmerged code)
        #geocode_new_isce3_kwargs = {}

        # geocoding optional arguments
        #geocode_kwargs = {}

        # Calculate layover/shadow mask when requested
        if save_layover_shadow_mask or apply_shadow_masking:
            flag_layover_shadow_mask_is_temporary = \
                (flag_bursts_secondary_files_are_temporary or
                    (apply_shadow_masking and not save_layover_shadow_mask))

            if flag_layover_shadow_mask_is_temporary:
                # layover/shadow mask is temporary
                layover_shadow_mask_file = \
                    (f'{burst_scratch_path}/{product_prefix}'
                     f'_layover_shadow_mask.{imagery_extension}')
            else:
                # layover/shadow mask is saved in `output_dir_sec_bursts`
                layover_shadow_mask_file = \
                    (f'{output_dir_sec_bursts}/{product_prefix}'
                     f'_layover_shadow_mask.{imagery_extension}')

            #layover_shadow_mask_raster = compute_layover_shadow_mask(
            #    radar_grid,
            #    orbit,
            #    geogrid,
            #    burst,
            #    dem_raster,
            #    layover_shadow_mask_file,
            #    output_raster_format,
            #    threshold_rdr2geo=cfg.rdr2geo_params.threshold,
            #    numiter_rdr2geo=cfg.rdr2geo_params.numiter,
            #    threshold_geo2rdr=cfg.geo2rdr_params.threshold,
            #    numiter_geo2rdr=cfg.geo2rdr_params.numiter)
            
            if flag_layover_shadow_mask_is_temporary:
                temp_files_list.append(layover_shadow_mask_file)
            else:
                output_file_list.append(layover_shadow_mask_file)
                logger.info(f'file saved: {layover_shadow_mask_file}')

            # Take the layover shadow mask from HDF5 file if not exists
            if save_secondary_layers_as_hdf5:
                layover_shadow_mask_file = (f'NETCDF:{burst_hdf5_in_output}:'
                                            '/science/SENTINEL1/RTC/grids/'
                                            'frequencyA/layoverShadowMask')


            output_metadata_dict['layover_shadow_mask'][1].append(
                layover_shadow_mask_file)

            #if apply_shadow_masking:
            #    geocode_new_isce3_kwargs['input_layover_shadow_mask_raster'] = \
            #        layover_shadow_mask_raster
        else:
            layover_shadow_mask_file = None

        # flag to run geocoding without shadow masking
        flag_geocoding_without_shadow_masking = False
        
        # flag to inform the user that there was an error using
        # sub-swath masking
        flag_inform_user_about_isce3_version_error = False

        # get sub_swaths metadata
#        if apply_valid_samples_sub_swath_masking:
        #    # Extract burst boundaries and create sub_swaths object to mask
        #    # invalid radar samples
        #    n_subswaths = 1
        #    sub_swaths = isce3.product.SubSwaths(radar_grid.length,
        #                                         radar_grid.width,
        #                                         n_subswaths)
        #    last_range_sample = min([burst.last_valid_sample, radar_grid.width])
        #    valid_samples_sub_swath = np.repeat(
        #        [[burst.first_valid_sample, last_range_sample + 1]],
        #        radar_grid.length, axis=0)
        #    for i in range(burst.first_valid_line):
        #        valid_samples_sub_swath[i, :] = 0
        #    for i in range(burst.last_valid_line, radar_grid.length):
        #        valid_samples_sub_swath[i, :] = 0
#
#            sub_swaths.set_valid_samples_array(1, valid_samples_sub_swath)
#            geocode_new_isce3_kwargs['sub_swaths'] = sub_swaths
#            geocode_kwargs['sub_swaths'] = sub_swaths

#          if apply_shadow_masking:
 #           # run ISCE3 geocoding
 #           try:
 #               geo_obj.geocode(radar_grid=radar_grid,
 #                               input_raster=rdr_burst_raster,
 #                               output_raster=geo_burst_raster,
 #                               dem_raster=dem_raster,
 #                               output_mode=geocode_algorithm,
 #                               geogrid_upsampling=geogrid_upsampling,
 #                               flag_apply_rtc=flag_apply_rtc,
 #                               input_terrain_radiometry=input_terrain_radiometry,
 #                               output_terrain_radiometry=output_terrain_radiometry,
 #                               exponent=exponent,
 #                               rtc_min_value_db=rtc_min_value_db,
 #                               rtc_upsampling=rtc_upsampling,
 #                               rtc_algorithm=rtc_algorithm,
 #                               abs_cal_factor=abs_cal_factor,
 #                               flag_upsample_radar_grid=flag_upsample_radar_grid,
 #                               clip_min = clip_min,
 #                               clip_max = clip_max,
 #                               out_geo_nlooks=out_geo_nlooks_obj,
 #                               out_geo_rtc=out_geo_rtc_obj,
 #                               input_rtc=None,
 #                               output_rtc=None,
 #                               dem_interp_method=dem_interp_method_enum,
 #                               memory_mode=memory_mode,
 #                               **geocode_new_isce3_kwargs)
 #           except TypeError:
 #               flag_geocoding_without_shadow_masking = True
 #               flag_inform_user_about_isce3_version_error = True
 #               logger.warning('WARNING there was an error executing geocode().'
 #                           ' Retrying it with less parameters')
#
#        else:
#            flag_geocoding_without_shadow_masking = True
#
#        if flag_geocoding_without_shadow_masking:
#            # run ISCE3 geocoding (without shadow masking)
#            geo_obj.geocode(radar_grid=radar_grid,
#                            input_raster=rdr_burst_raster,
#                            output_raster=geo_burst_raster,
#                            dem_raster=dem_raster,
#                            output_mode=geocode_algorithm,
#                            geogrid_upsampling=geogrid_upsampling,
#                            flag_apply_rtc=flag_apply_rtc,
#                            input_terrain_radiometry=input_terrain_radiometry,
#                            output_terrain_radiometry=output_terrain_radiometry,
#                            exponent=exponent,
#                            rtc_min_value_db=rtc_min_value_db,
#                            rtc_upsampling=rtc_upsampling,
#                            rtc_algorithm=rtc_algorithm,
#                            abs_cal_factor=abs_cal_factor,
#                            flag_upsample_radar_grid=flag_upsample_radar_grid,
#                            clip_min = clip_min,
#                            clip_max = clip_max,
       #                     out_geo_nlooks=out_geo_nlooks_obj,
      #                      out_geo_rtc=out_geo_rtc_obj,
     #                       input_rtc=None,
    #                        output_rtc=None,
   #                         dem_interp_method=dem_interp_method_enum,
  #                          memory_mode=memory_mode,
 #                           **geocode_kwargs)
#
  #          if flag_inform_user_about_isce3_version_error:
  #              logger.warning('WARNING shadow masking is not available'
 #                              ' from installed version of ISCE3.')
#
#        del geo_burst_raster

        # Output imagery list contains multi-band files that
        # will be used for mosaicking
        output_imagery_list.append(geo_burst_filename)

        # If burst imagery is not temporary, separate polarization channels
        if not flag_bursts_files_are_temporary:
            output_burst_imagery_list = []
            for pol in pol_list:
                geo_burst_pol_filename = \
                    os.path.join(output_dir, burst_id,
                        f'{product_prefix}_{pol}.' +
                        f'{imagery_extension}')
                output_burst_imagery_list.append(geo_burst_pol_filename)

            #_separate_pol_channels(geo_burst_filename,
            #                        output_burst_imagery_list,
            #                        logger, output_raster_format)

            # create a vrt from the separated pol tiffs
            geo_burst_filename +='.vrt'
            os.makedirs(os.path.dirname(geo_burst_filename), exist_ok=True)
            gdal.BuildVRT(geo_burst_filename, output_burst_imagery_list,
                            options=vrt_options_mosaic)
            output_imagery_list[-1] = geo_burst_filename

            output_file_list += output_burst_imagery_list

        if save_nlooks:
            #del out_geo_nlooks_obj

            if not flag_bursts_secondary_files_are_temporary:
                logger.info(f'file saved: {nlooks_file}')
            output_metadata_dict['nlooks'][1].append(nlooks_file)

        if save_rtc_anf:
            #del out_geo_rtc_obj

            if not flag_bursts_secondary_files_are_temporary:
                logger.info(f'file saved: {rtc_anf_file}')
            output_metadata_dict['rtc'][1].append(rtc_anf_file)

        radar_grid_file_dict = {}
        if flag_call_radar_grid and save_bursts:
            if save_incidence_angle:
                radar_grid_file_dict['incidenceAngle'] =\
                    os.path.join(output_dir,
                                f'{product_prefix}_incidence_angle.', imagery_extension)
            if save_local_inc_angle:
                radar_grid_file_dict['localIncidenceAngle'] =\
                    os.path.join(output_dir,
                                f'{product_prefix}_local_incidence_angle.', imagery_extension)
            if save_projection_angle:
                radar_grid_file_dict['projectionAngle'] =\
                    os.path.join(output_dir,
                                f'{product_prefix}_projection_angle.', imagery_extension)
            if save_rtc_anf_psi:
                radar_grid_file_dict['areaNormalizationFactorPsi'] =\
                    os.path.join(output_dir,
                                f'{product_prefix}_rtc_anf_psi.', imagery_extension)
            if save_range_slope:
                radar_grid_file_dict['rangeSlope'] =\
                    os.path.join(output_dir,
                                f'{product_prefix}_range_slope.', imagery_extension)
            if save_dem:
                radar_grid_file_dict['interpolatedDem'] =\
                    os.path.join(output_dir,
                                f'{product_prefix}_interpolated_dem.', imagery_extension)

            if flag_bursts_secondary_files_are_temporary:
                # files are temporary
                temp_files_list += list(radar_grid_file_dict.values())
            else:
                output_file_list += list(radar_grid_file_dict.values())

        # Create burst HDF5
        if ((save_imagery_as_hdf5 or save_metadata) and save_bursts):
            hdf5_file_output_dir = os.path.join(output_dir, burst_id)
            os.makedirs(hdf5_file_output_dir, exist_ok=True)
            output_hdf5_file_burst =  os.path.join(
                hdf5_file_output_dir, f'{product_prefix}.{hdf5_file_extension}')
            #with create_hdf5_file(output_hdf5_file_burst, orbit, burst, cfg) as hdf5_burst_obj:
            #    save_hdf5_file(
            #        hdf5_burst_obj, output_hdf5_file_burst, flag_apply_rtc,
            #        clip_max, clip_min, output_radiometry_str,
            #        geogrid, pol_list, geo_burst_filename, nlooks_file,
            #        rtc_anf_file, layover_shadow_mask_file,
            #        radar_grid_file_dict,
            #        save_imagery = save_imagery_as_hdf5,
            #        save_secondary_layers = save_secondary_layers_as_hdf5)
            output_file_list.append(output_hdf5_file_burst)

        # Create mosaic HDF5
        #if ((save_imagery_as_hdf5 or save_metadata) and save_mosaics
        #        and burst_index == 0):
        #    hdf5_obj = create_hdf5_file(output_hdf5_file, orbit, burst, cfg)
        if ((save_imagery_as_hdf5 or save_metadata) and save_mosaics
                and burst_index == 0):
            hdf5_mosaic_obj = rtc_s1.create_hdf5_file(output_hdf5_file, orbit, burst, cfg)


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
        rtc_s1.get_radar_grid(cfg.geogrid, dem_interp_method_enum, product_prefix,
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
                (f'{output_dir_mosaic_raster}/{product_prefix}_{pol}.'
                 f'{imagery_extension}')
            logger.info(f'    {geo_pol_filename}')
            output_imagery_filename_list.append(geo_pol_filename)

        nlooks_list = output_metadata_dict['nlooks'][1]
        rtc_s1.compute_weighted_mosaic_raster_single_band(
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
            rtc_s1.compute_weighted_mosaic_raster(input_files, nlooks_list, output_file,
                            cfg.geogrid, verbose=False)




            # TODO: Remove nlooks exception below
            if (save_secondary_layers_as_hdf5 or
                    (key == 'nlooks' and not save_nlooks)):
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

            sensing_start_ds = f'{rtc_s1.BASE_HDF5_DATASET}/identification/zeroDopplerStartTime'
            sensing_end_ds = f'{rtc_s1.BASE_HDF5_DATASET}/identification/zeroDopplerEndTime'
            del hdf5_mosaic_obj[sensing_start_ds]
            del hdf5_mosaic_obj[sensing_end_ds]
            hdf5_mosaic_obj[sensing_start_ds] = \
                sensing_start.strftime('%Y-%m-%dT%H:%M:%S.%f')
            hdf5_mosaic_obj[sensing_end_ds] = \
                sensing_stop.strftime('%Y-%m-%dT%H:%M:%S.%f')

            rtc_s1.save_hdf5_file(
                hdf5_mosaic_obj, output_hdf5_file, flag_apply_rtc,
                clip_max, clip_min, output_radiometry_str,
                cfg.geogrid, pol_list, geo_filename, nlooks_mosaic_file,
                rtc_anf_mosaic_file, layover_shadow_mask_file,
                radar_grid_file_dict, save_imagery = save_imagery_as_hdf5,
                save_secondary_layers = save_secondary_layers_as_hdf5)
            hdf5_mosaic_obj.close()
            output_file_list.append(output_hdf5_file)

    if output_imagery_format == 'COG':
        logger.info(f'Saving files as Cloud-Optimized GeoTIFFs (COGs)')
        for filename in output_file_list:
            if not filename.endswith('.tif'):
                continue
            logger.info(f'    processing file: {filename}')
            rtc_s1.save_as_cog(filename, scratch_path, logger,
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



def get_parent_logger_setting(logger_in):
    path_logger = ''

    flag_full_format = logger_in.handlers[0].formatter._fmt != '%(message)s'

    for handler_logger in logger_in.handlers:
        if isinstance(handler_logger, logging.FileHandler):
            path_logger = handler_logger.baseFilename
            continue

    return path_logger, flag_full_format



def main():
    '''
    Main entrypoint of the script
    '''
    parser  = rtc_s1.get_rtc_s1_parser()
    # parse arguments
    args = parser.parse_args()

    # Spawn multiple processes for parallelization
    rtc_s1.create_logger(args.log_file, args.full_log_formatting)

    # Get a runconfig dict from command line argumens
    cfg = RunConfig.load_from_yaml(args.run_config_path, 'rtc_s1')

    rtc_s1._load_parameters(cfg)

    # Run geocode burst workflow
    run_parallel(cfg)


if __name__ == "__main__":
    # load arguments from command line
    main()