#!/usr/bin/env python3

import os
import requests
import tarfile
from osgeo import gdal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        assert not _is_valid_gdal_reference(geo_hh_file)

    else:

        # assert that VV image is present
        geo_vv_filename = os.path.join(
            output_dir, f'{product_prefix}_VV.{imagery_extension}')
        assert os.path.isfile(geo_vv_filename)

        # assert that HH image is not present
        geo_hh_filename = os.path.join(
            output_dir, f'{product_prefix}_HH.{imagery_extension}')
        assert not os.path.isfile(geo_hh_filename)

    # Check RTC-S1 secondary layers
    if save_secondary_layers_as_hdf5:

        # assert that the following secondary layers are present:
        ds_list = ['numberOfLooks',
                   'rtcAreaNormalizationFactorGamma0ToBeta0',
                   # 'rtcAreaNormalizationFactorGamma0ToSigma0',
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
            assert not _is_valid_gdal_reference(current_file)

    else:
        # assert that the following secondary layers are present:
        ds_list = ['number_of_looks', 'rtc_anf_gamma0_to_beta0',
                   # 'rtc_area_normalization_factor_gamma0_to_sigma0',
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
            assert not os.path.isfile(current_file)


def test_workflow():

    test_data_directory = 'data'

    if not os.path.isdir(test_data_directory):
        os.makedirs(test_data_directory, exist_ok=True)

    dataset_name = 's1b_los_angeles'
    dataset_url = ('https://zenodo.org/records/7753472/files/'
                   's1b_los_angeles.tar.gz?download=1')

    tests_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(test_data_directory, dataset_name)
    if FLAG_ALWAYS_DOWNLOAD or not os.path.isdir(dataset_dir):

        print(f'Test dataset {dataset_name} not found. Downloading'
              f' file {dataset_url}.')
        # To avoid the issue in downloading, try again. 
        session = requests.Session()
        retries = Retry(
            total=5,                    # up to 5 attempts
            backoff_factor=2,           # 2 s, 4 s, 8 s, …
            status_forcelist=[502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        compressed_path = os.path.join(test_data_directory,
                                       f"{dataset_name}.tar.gz")
        with session.get(dataset_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(compressed_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB
                    f.write(chunk)

        print(f"Extracting {compressed_path}")
        with tarfile.open(compressed_path, "r:gz") as tf:
            tf.extractall(test_data_directory)

    # create logger
    log_file = os.path.join('data', 'log.txt')
    full_log_formatting = False
    create_logger(log_file, full_log_formatting)

    for runconfig_mode in ['mask_off', 'mask_on',
                           'mask_off_h5', 'mask_on_h5']:

        # Get a runconfig dict from command line argumens
        runconfig_path = os.path.join(
                tests_dir, 'runconfigs',
                f's1b_los_angeles_{runconfig_mode}.yaml')

        cfg = RunConfig.load_from_yaml(runconfig_path)

        output_dir_single_job, product_prefix, save_imagery_as_hdf5, \
            save_secondary_layers_as_hdf5, save_metadata, \
            hdf5_file_extension, imagery_extension = _load_cfg_parameters(cfg)

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
