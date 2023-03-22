'''
RTC-S1 Science Application Software
'''

from itertools import repeat
import logging
import multiprocessing
import os
import subprocess
import time
import yaml

import isce3
import numpy as np
from osgeo import gdal
from rtc.rtc_s1_single_job import (add_output_to_output_metadata_dict,
                                   snap_coord,
                                   run_single_job,
                                   get_radar_grid,
                                   save_browse,
                                   append_metadata_to_geotiff_file)
from rtc.mosaic_geobursts import (compute_weighted_mosaic_raster,
                                  compute_weighted_mosaic_raster_single_band)
from rtc.core import create_logger, save_as_cog
from rtc.version import VERSION as SOFTWARE_VERSION
from rtc.h5_prep import (save_hdf5_file, create_hdf5_file, BASE_HDF5_DATASET,
                         get_metadata_dict, all_metadata_dict_to_geotiff_metadata_dict)

from rtc.runconfig import RunConfig, load_parameters


logger = logging.getLogger('rtc_s1')


def _populate_radar_grid_file_dict(radar_grid_file_dict: dict,
                                   key_layer: str, save_as_hdf5: bool,
                                   output_dir: str, product_id: str,
                                   layer_postfix: str, imagery_extension: str,
                                   burst_hdf5_in_output: str):
    '''
    Helper function to populate `radar_grid_file_dict` in the parent process

    Parameters
    ----------
    radar_grid_file_dict: dict
        Radar grid dict to populate
    key_layer: str
        Key of the layer in `radar_grid_file_dict`
    save_as_hdf5: bool
        Flag whether to save the radargrid layers as HDF5
    output_dir: str
        Output directory
    product_id: str
        Prefix string for the radar grid file
    layer_postfix: str
        Postfix for the radar grid file to specify the layer
    imagery_extension: str
        Image extension
    burst_hdf5_in_output: str
        Path to the output burst HDF5
    '''
    if save_as_hdf5:
        radar_grid_file_dict[key_layer] = (f'NETCDF:{burst_hdf5_in_output}:'
                                           '/science/SENTINEL1/RTC/grids/'
                                           f'frequencyA/{key_layer}')
    else:
        radar_grid_file_dict[key_layer] =\
            os.path.join(output_dir,
                         f'{product_id}_{layer_postfix}.', imagery_extension)


def split_runconfig(cfg_in,
                    output_dir_child,
                    scratch_path_child=None,
                    path_parent_logfile=None):
    '''
    Split the input runconfig into single burst runconfigs.
    Writes out the burst runconfigs.
    Return the list of the burst runconfigs.

    Parameters
    ----------
    cfg_in: rtc.runconfig.RunConfig
        Path to the original runconfig
    output_dir_child: str
        Output directory of the child process
    scratch_path_child: str
        Scratch path to of the child process.
        If `None`, the scratch path of the child processes it will be:
         "[scratch path of parent process]_child_scratch"
    path_parent_logfile: str
        Path to the parent processes' logfile

    Returns
    -------
    runconfig_burst_list: list(str)
        List of the burst runconfigs
    logfile_burst_list: list(str)
        List of the burst logfiles
    '''

    with open(cfg_in.run_config_path, 'r+', encoding='utf8') as fin:
        runconfig_dict_in = yaml.safe_load(fin.read())

    runconfig_burst_list = []
    logfile_burst_list = []

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
        if path_parent_logfile:
            path_logfile_child = os.path.join(output_dir_child,
                                              f'{burst_id}.{path_parent_logfile}')
        else:
            path_logfile_child = None

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
                                 'save_bursts'],
                                True)

        # TODO: Remove the code below once the
        # mosaicking algorithm does not take nlooks as the weight input
        if cfg_in.groups.product_group.save_mosaics:
            set_dict_item_recursive(runconfig_dict_out,
                                    ['runconfig',
                                     'groups',
                                     'processing',
                                     'geocoding',
                                     'save_nlooks'],
                                    True)

        runconfig_burst_list.append(path_temp_runconfig)
        logfile_burst_list.append(path_logfile_child)

        with open(path_temp_runconfig, 'w+', encoding='utf8') as fout:
            yaml.dump(runconfig_dict_out, fout)

    return runconfig_burst_list, logfile_burst_list


def set_dict_item_recursive(dict_in, list_path, val):
    '''
    - Recursively locate the key in `dict_in`,
      whose path is provided in the list of keys.
    - Add or update the value of the located key of `dict_in`
    - Create the key in `dict_in` with empty dict when the key does not exist

    Parameters
    ----------
    dict_in: dict
        Dict to update
    list_path: list
        Path to the item in the multiple layer of dict
    val: any
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
                            path_burst_logfile=None,
                            flag_full_logfile_format=None,
                            keep_burst_runconfig=False):
    '''
    single worker to process runconfig using `subprocess`
    Parameters
    ----------
    path_runconfig_burst: str
        Path to the burst runconfig
    path_burst_logfile: str
        Path to the burst logfile
    full_log_format: bool
        Enable full formatting of log messages.
        See `get_rtc_s1_parser()`
    keep_burst_runconfig: bool
        Keep the child runconfig when `True`;
        delete it after done with the processing when `False`
    Returns
    -------
    result_child_process: int
        0 when the child process has completed succesfully
    '''
    os.environ['OMP_NUM_THREADS'] = "1"
    list_arg_subprocess = ['rtc_s1_single_job.py', path_runconfig_burst]

    if path_burst_logfile is not None: 
        list_arg_subprocess += ['--log-file', path_burst_logfile]

    if flag_full_logfile_format:
        list_arg_subprocess.append('--full-log-format')

    child_processing_result = subprocess.run(list_arg_subprocess)

    if not keep_burst_runconfig:
        os.remove(path_runconfig_burst)

    return child_processing_result.returncode


def run_parallel(cfg: RunConfig, logfile_path, flag_logger_full_format):
    '''
    Run RTC workflow with user-defined args
    stored in dictionary runconfig `cfg`

    Parameters
    ---------
    cfg: RunConfig
        RunConfig object with user runconfig options
    logfile_path: str
        Path to the parent processes' logfile
    full_log_formatting: bool
        Flag to enable full log formatting
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
        runconfig_product_id = 'rtc_product'
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

    apply_shadow_masking = \
        cfg.groups.processing.geocoding.apply_shadow_masking

    clip_max = geocode_namespace.clip_max
    clip_min = geocode_namespace.clip_min
    save_incidence_angle = geocode_namespace.save_incidence_angle
    save_local_inc_angle = geocode_namespace.save_local_inc_angle
    save_projection_angle = geocode_namespace.save_projection_angle
    save_rtc_anf_psi = geocode_namespace.save_rtc_anf_psi
    save_range_slope = geocode_namespace.save_range_slope
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
        logger.info(f'    mosaic product ID: {product_id}')
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
    logger.info(f'    output imagery compression: {output_imagery_compression}')
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
    # e.g. layover shadow mask, nlooks, area normalization factor
    add_output_to_output_metadata_dict(
        save_layover_shadow_mask, 'layover_shadow_mask',
        output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)
    add_output_to_output_metadata_dict(
        save_nlooks, 'nlooks', output_dir_sec_mosaic_raster,
        output_metadata_dict, mosaic_product_id, imagery_extension)
    add_output_to_output_metadata_dict(
        save_rtc_anf, 'rtc_area_normalization_factor',
        output_dir_sec_mosaic_raster,
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

    # ------ Start parallelized burst processing ------
    t_start_parallel = time.time()
    logger.info(f'Starting child processes for burst processing')

    if save_bursts:
        output_path_child = output_dir
        scratch_path_child = scratch_path
    else:
        output_path_child = scratch_path
        scratch_path_child = f'{scratch_path}_child_scratch'

    # burst files are saved in scratch dir
    burst_runconfig_list, burst_log_list = split_runconfig(cfg,
                                                           output_path_child,
                                                           scratch_path_child,
                                                           logfile_path)

    # determine the number of the processors here
    num_workers = cfg.groups.processing.num_workers

    if num_workers == 0:
        # Read system variable OMP_NUM_THREADS
        num_workers = os.getenv('OMP_NUM_THREADS')
        if not num_workers:
            # Otherwise, read it from os.cpu_count()
            num_workers = os.cpu_count()
    num_workers = min(int(num_workers), len(burst_runconfig_list))

    # Execute the single burst processes using multiprocessing
    with multiprocessing.Pool(num_workers) as p:
        processing_result_list =\
            p.starmap(process_child_runconfig,
                      zip(burst_runconfig_list,
                          burst_log_list,
                          repeat(flag_logger_full_format)
                          )
                      )
    t_end_parallel = time.time()
    logger.info('Child processes has completed. '
                f'Elapsed time: {t_end_parallel - t_start_parallel} seconds.')
    # ------  End of parallelized burst processing ------

    # Check if there are any failed child processes
    all_child_processes_successful =\
        processing_result_list.count(0) == len(burst_runconfig_list)

    if all_child_processes_successful:
        # delete the log files for child processes
        if not save_bursts and logfile_path:
            temp_files_list += burst_log_list
    else:
        msg_failed_child_proc = (f'Some of the child process(es) from '
                                 'burst runconfig (listed below) '
                                 'did not complete succesfully:\n')
        list_burst_id = list(cfg.bursts.keys())
        for index_child, processing_result in enumerate(processing_result_list):
            if processing_result != 0:
                msg_failed_child_proc += (f'"{burst_runconfig_list[index_child]}"'
                                          ' for burst ID '
                                          f'"{list_burst_id[index_child]}"\n')
        raise RuntimeError(msg_failed_child_proc)

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
        print('burst ID name:', burst_id_file_name)    
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

            if (flag_apply_thermal_noise_correction or
                    flag_apply_abs_rad_correction):
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

        # Generate output geocoded burst raster
        burst_hdf5_in_output = os.path.join(output_path_child,
                                            burst_id,
                                            f'{burst_product_id}.{hdf5_file_extension}')
        if not save_bursts:
            temp_files_list.append(burst_hdf5_in_output)

        if save_nlooks:
            if save_secondary_layers_as_hdf5:
                nlooks_file = (f'NETCDF:"{burst_hdf5_in_output}":'
                               '/science/SENTINEL1/RTC/grids/frequencyA/'
                               'numberOfLooks')
            else:
                nlooks_file = (f'{output_dir_sec_bursts}/{burst_product_id}'
                               f'_nlooks.{imagery_extension}')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(nlooks_file)
            else:
                output_file_list.append(nlooks_file)
        else:
            nlooks_file = None

        if save_rtc_anf:
            if save_secondary_layers_as_hdf5:
                rtc_anf_file = (f'NETCDF:"{burst_hdf5_in_output}":'
                                '/science/SENTINEL1/RTC/grids/frequencyA/'
                                'RTCAreaNormalizationFactor')
            else:
                rtc_anf_file = (f'{output_dir_sec_bursts}/{burst_product_id}'
                                f'_rtc_anf.{imagery_extension}')

            if flag_bursts_secondary_files_are_temporary:
                temp_files_list.append(rtc_anf_file)
            else:
                output_file_list.append(rtc_anf_file)

        else:
            rtc_anf_file = None

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

            if flag_layover_shadow_mask_is_temporary:
                temp_files_list.append(layover_shadow_mask_file)
                layover_shadow_mask_file = None
            else:
                output_file_list.append(layover_shadow_mask_file)
                logger.info(f'file saved: {layover_shadow_mask_file}')

                # Take the layover shadow mask from HDF5 file if not exists
                if save_secondary_layers_as_hdf5:
                    layover_shadow_mask_file = (f'NETCDF:{burst_hdf5_in_output}:'
                                                '/science/SENTINEL1/RTC/grids/'
                                                'frequencyA/layoverShadowMask')

                if save_layover_shadow_mask:
                    output_metadata_dict['layover_shadow_mask'][1].append(
                        layover_shadow_mask_file)

            if not save_layover_shadow_mask:
                layover_shadow_mask_file = None

        else:
            layover_shadow_mask_file = None

        # Output imagery list contains multi-band files that
        # will be used for mosaicking
        output_burst_imagery_list = []
        for pol in pol_list:
            if save_imagery_as_hdf5:
                geo_burst_pol_filename = (f'NETCDF:{burst_hdf5_in_output}:'
                                          '/science/SENTINEL1/RTC/grids/'
                                          f'frequencyA/{pol}')
            else:
                geo_burst_pol_filename = \
                    os.path.join(output_path_child, burst_id,
                                 f'{burst_product_id}_{pol}.' +
                                 f'{imagery_extension}')
            output_burst_imagery_list.append(geo_burst_pol_filename)

        # Bundle the single-pol geo burst files into .vrt
        geo_burst_filename += '.vrt'
        os.makedirs(os.path.dirname(geo_burst_filename), exist_ok=True)
        gdal.BuildVRT(geo_burst_filename, output_burst_imagery_list,
                      options=vrt_options_mosaic)
        output_imagery_list.append(geo_burst_filename)

        # .vrt files (for RTC product in geogrid) will be removed after the process
        temp_files_list.append(geo_burst_filename)

        if not flag_bursts_files_are_temporary:
            output_file_list += output_burst_imagery_list
        else:
            temp_files_list += output_burst_imagery_list

        if save_nlooks:
            output_metadata_dict['nlooks'][1].append(nlooks_file)

        if save_rtc_anf:
            output_metadata_dict['rtc_area_normalization_factor'][1].append(rtc_anf_file)

        radar_grid_file_dict = {}
        if flag_call_radar_grid and save_bursts:
            if save_incidence_angle:
                _populate_radar_grid_file_dict(
                    radar_grid_file_dict, 'incidenceAngle',
                    save_secondary_layers_as_hdf5, output_dir, burst_product_id,
                    'incidence_angle', imagery_extension, burst_hdf5_in_output)

            if save_local_inc_angle:
                _populate_radar_grid_file_dict(
                    radar_grid_file_dict, 'localIncidenceAngle',
                    save_secondary_layers_as_hdf5, output_dir, burst_product_id,
                    'local_incidence_angle', imagery_extension, burst_hdf5_in_output)

            if save_projection_angle:
                _populate_radar_grid_file_dict(
                    radar_grid_file_dict, 'projectionAngle',
                    save_secondary_layers_as_hdf5, output_dir, burst_product_id,
                    'projection_angle', imagery_extension, burst_hdf5_in_output)
            if save_rtc_anf_psi:
                _populate_radar_grid_file_dict(
                    radar_grid_file_dict, 'RTCAreaNormalizationFactorPsi',
                    save_secondary_layers_as_hdf5, output_dir, burst_product_id,
                    'rtc_anf_psi', imagery_extension, burst_hdf5_in_output)

            if save_range_slope:
                _populate_radar_grid_file_dict(
                    radar_grid_file_dict, 'rangeSlope',
                    save_secondary_layers_as_hdf5, output_dir, burst_product_id,
                    'range_slope', imagery_extension, burst_hdf5_in_output)

            if save_dem:
                _populate_radar_grid_file_dict(
                    radar_grid_file_dict, 'interpolatedDem',
                    save_secondary_layers_as_hdf5, output_dir, burst_product_id,
                    'interpolated_dem', imagery_extension, burst_hdf5_in_output)


            if flag_bursts_secondary_files_are_temporary:
                # files are temporary
                temp_files_list += list(radar_grid_file_dict.values())
            else:
                output_file_list += list(radar_grid_file_dict.values())

        # Create burst HDF5
        if (save_hdf5_metadata and save_bursts):
            hdf5_file_output_dir = os.path.join(output_dir, burst_id)
            os.makedirs(hdf5_file_output_dir, exist_ok=True)
            output_hdf5_file_burst = os.path.join(
                hdf5_file_output_dir, f'{burst_product_id}.{hdf5_file_extension}')
            output_file_list.append(output_hdf5_file_burst)

        # Create mosaic HDF5
        if (save_hdf5_metadata and save_mosaics
                and burst_index == 0):
            hdf5_mosaic_obj = create_hdf5_file(
                mosaic_product_id, output_hdf5_file, orbit, burst, cfg,
                is_mosaic=True)

        if (save_mosaics and burst_index == 0):
            mosaic_metadata_dict = get_metadata_dict(burst_product_id, burst, cfg,
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
        get_radar_grid(cfg.geogrid, dem_interp_method_enum, burst_product_id,
                       radar_grid_output_dir, imagery_extension, save_incidence_angle,
                       save_local_inc_angle, save_projection_angle,
                       save_rtc_anf_psi,
                       save_range_slope, save_dem,
                       dem_raster, radar_grid_file_dict,
                       lookside, wavelength,
                       orbit, verbose=not save_imagery_as_hdf5)
        if save_hdf5_metadata:
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
                input_files, nlooks_list, output_file, cfg.geogrid, verbose=False)

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


        # Save the mosaicked layers as HDF5
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

    if save_mosaics:
        for current_file in mosaic_output_file_list:
            if not current_file.endswith('.tif'):
                continue
            append_metadata_to_geotiff_file(current_file,
                                            mosaic_geotiff_metadata_dict)

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
