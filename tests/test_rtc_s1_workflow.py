#!/usr/bin/env python3

import os
import requests
import glob
import tarfile

from rtc.runconfig import RunConfig, load_parameters
from rtc.core import create_logger
from rtc.rtc_s1_single_job import get_rtc_s1_parser, run_single_job


from rtc.version import VERSION as SOFTWARE_VERSION

FLAG_ALWAYS_DOWNLOAD = False

def test_workflow():

    test_data_directory = 'data'

    if not os.path.isdir(test_data_directory):
        os.makedirs(test_data_directory, exist_ok=True)

    dataset_name = 's1b_los_angeles'
    dataset_url = ('https://zenodo.org/record/7753472/files/'
                   's1b_los_angeles.tar.gz?download=1')

    dataset_dir = os.path.join(test_data_directory, dataset_name)
    user_runconfig_file = os.path.join(dataset_dir, 'rtc_s1.yaml')

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
    runconfig_path = os.path.join('data', 's1b_los_angeles', 'rtc_s1.yaml')
    cfg = RunConfig.load_from_yaml(runconfig_path, 'rtc_s1')

    load_parameters(cfg)

    # Run geocode burst workflow
    run_single_job(cfg)
    product_version_float = cfg.groups.product_group.product_version
    if product_version_float is None:
        product_version = SOFTWARE_VERSION
    else:
        product_version = f'{product_version_float:.1f}'
    output_dir = cfg.groups.product_group.output_dir
    product_id = cfg.groups.product_group.product_id
    if product_id is None:
        product_id = 'rtc_product'
    product_prefix = f'{product_id}_v{product_version}'

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

    if save_metadata:
        geo_h5_filename = f'{output_dir}/'f'{product_prefix}.{hdf5_file_extension}'

    if not save_imagery_as_hdf5:
        geo_vv_filename = f'{output_dir}/'f'{product_prefix}_VV.{imagery_extension}'
        # geo_vh_filename = f'{output_dir}/'f'{product_prefix}_VH.{imagery_extension}'

    assert(os.path.isfile(geo_h5_filename))
    assert(os.path.isfile(geo_vv_filename))
    # assert(os.path.isfile(geo_vh_filename))
