#!/usr/bin/env python3

import os
import requests
import tarfile
from osgeo import gdal

from rtc.runconfig import RunConfig, load_parameters
from rtc.core import create_logger
from rtc.rtc_s1_single_job import run_single_job
from rtc.rtc_s1 import run_parallel
from rtc.h5_prep import DATA_BASE_GROUP

FLAG_ALWAYS_DOWNLOAD = False


def _load_cfg_parameters(cfg):

    load_parameters(cfg)

    # Load parameters
    output_dir = cfg.groups.product_group.output_dir
    product_id = cfg.groups.product_group.product_id
    if product_id is None:
        product_id = 'OPERA_L2_RTC-S1_{burst_id}'
    product_prefix = product_id

    output_imagery_format = \
        cfg.groups.product_group.output_imagery_format
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

    return output_dir, product_prefix, save_imagery_as_hdf5, \
        save_secondary_layers_as_hdf5, save_metadata, \
        hdf5_file_extension, imagery_extension


def _is_valid_gdal_reference(gdal_reference):
    try:
        gdal_ds = gdal.Open(gdal_reference, gdal.GA_ReadOnly)
        return gdal_ds is not None
    except RuntimeError:
        return False
    return False


def _check_results(output_dir, product_prefix, save_imagery_as_hdf5,
                   save_secondary_layers_as_hdf5, save_metadata,
                   hdf5_file_extension, imagery_extension):

    # Check RTC-S1 HDF5 metadata
    assert save_metadata
    geo_h5_filename = os.path.join(
        output_dir, f'{product_prefix}.{hdf5_file_extension}')

    # Check RTC-S1 imagery
    if save_imagery_as_hdf5:

        # assert that VV image is present
        geo_vv_file = (f'NETCDF:"{geo_h5_filename}":'
                       f'{DATA_BASE_GROUP}/VV')
        assert _is_valid_gdal_reference(geo_vv_file)

        # assert that HH image is not present
        geo_hh_file = (f'NETCDF:"{geo_h5_filename}":'
                       f'{DATA_BASE_GROUP}/HH')
        assert not(_is_valid_gdal_reference(geo_hh_file))

    else:
    
        # assert that VV image is present
        geo_vv_filename = os.path.join(
            output_dir, f'{product_prefix}_VV.{imagery_extension}')
        assert os.path.isfile(geo_vv_filename)

        # assert that HH image is not present
        geo_hh_filename = os.path.join(
            output_dir, f'{product_prefix}_HH.{imagery_extension}')
        assert not(os.path.isfile(geo_hh_filename))
    
    # Check RTC-S1 secondary layers
    if save_secondary_layers_as_hdf5:

        # assert that the following secondary layers are present:
        ds_list = ['numberOfLooks',
                   'rtcAnfGamma0ToBeta0',
                   'rtcAnfGamma0ToSigma0',
                   'localIncidenceAngle']
        for ds_name in ds_list:
            current_file = (f'NETCDF:"{geo_h5_filename}":'
                            f'{DATA_BASE_GROUP}/'
                            f'{ds_name}')
            assert _is_valid_gdal_reference(current_file)

        # assert that the following secondary layers are not present:
        ds_list = ['incidenceAngle', 'projectionAngle']
        for ds_name in ds_list:
            current_file = (f'NETCDF:"{geo_h5_filename}":'
                            f'{DATA_BASE_GROUP}/'
                            f'{ds_name}')
            assert not(_is_valid_gdal_reference(current_file))

    else:
        # assert that the following secondary layers are present:
        ds_list = ['number_of_looks', 'rtc_area_normalization_factor',
                   'rtc_area_normalization_factor_gamma0_to_sigma0',
                   'local_incidence_angle']
        for ds_name in ds_list:
            current_file = os.path.join(
                output_dir, f'{product_prefix}_'
                f'{ds_name}.{imagery_extension}')
            assert os.path.isfile(current_file)

        # assert that the following secondary layers are not present:
        ds_list = ['incidence_angle', 'projectionAngle']
        for ds_name in ds_list:
            current_file = os.path.join(
                output_dir, f'{product_prefix}_'
                f'{ds_name}.{imagery_extension}')
            assert not(os.path.isfile(current_file))


def test_workflow():

    test_data_directory = 'data'

    if not os.path.isdir(test_data_directory):
        os.makedirs(test_data_directory, exist_ok=True)

    dataset_name = 's1b_los_angeles'
    dataset_url = ('https://zenodo.org/record/7753472/files/'
                   's1b_los_angeles.tar.gz?download=1')

    tests_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(test_data_directory, dataset_name)
    user_runconfig_file = os.path.join(tests_dir, 'runconfigs',
                                       's1b_los_angeles.yaml')

    if (FLAG_ALWAYS_DOWNLOAD or not os.path.isdir(dataset_dir) or
            not os.path.isfile(user_runconfig_file)):

        print(f'Test dataset {dataset_name} not found. Downloading'
              f' file {dataset_url}.')
        response = requests.get(dataset_url)
        response.raise_for_status()

        compressed_filename = os.path.join(test_data_directory,
                                           os.path.basename(dataset_url))

        open(compressed_filename, 'wb').write(response.content)

        print(f'Extracting downloaded file {compressed_filename}')
        with tarfile.open(compressed_filename) as compressed_file:
            compressed_file.extractall(test_data_directory)

    # create logger
    log_file = os.path.join('data', 'log.txt')
    full_log_formatting = False
    create_logger(log_file, full_log_formatting)

    # Get a runconfig dict from command line argumens
    runconfig_path = os.path.join(tests_dir, 'runconfigs',
                                  's1b_los_angeles.yaml')

    # for output_imagery_format in ['COG', 'HDF5']:
    for output_imagery_format in ['COG']:

        cfg = RunConfig.load_from_yaml(runconfig_path)
        cfg.groups.product_group.output_imagery_format = output_imagery_format

        output_dir_single_job, product_prefix, save_imagery_as_hdf5, \
            save_secondary_layers_as_hdf5, save_metadata, \
            hdf5_file_extension, imagery_extension = _load_cfg_parameters(cfg)

        # Testing creation of secondary layers:
        #
        # The YAML file above (`runconfig_path`) is only set to create the
        # `numberOfLooks`, `rtcAnfGamma0ToBeta0`,
        # and `rtcAnfGamma0ToSigma0`. Here,
        # we also force the creation of `layoverShadowMask` and
        # `localIncidenceAngle` and will test if they are present in the
        # output files. We also assert that layers not set to be created,
        # such as `incidenceAngle`, are indeed not created

        cfg.groups.processing.geocoding.save_layover_shadow_mask = True
        cfg.groups.processing.geocoding.save_local_inc_angle = True

        # Run geocode burst workflow (single job)
        run_single_job(cfg)

        _check_results(output_dir_single_job, product_prefix,
                       save_imagery_as_hdf5, save_secondary_layers_as_hdf5,
                       save_metadata, hdf5_file_extension,
                       imagery_extension)

        # Run geocode burst workflow (parallel)
        output_dir_parallel = os.path.join('data', 's1b_los_angeles',
                                           'output_dir_parallel')

        cfg.groups.product_group.output_dir = output_dir_parallel

        log_file_path = 'log.txt'
        flag_logger_full_format = False
        run_parallel(cfg, log_file_path, flag_logger_full_format)

        _check_results(output_dir_parallel, product_prefix,
                       save_imagery_as_hdf5, save_secondary_layers_as_hdf5,
                       save_metadata, hdf5_file_extension, imagery_extension)

