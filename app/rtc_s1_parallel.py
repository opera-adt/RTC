#!/usr/bin/env python

'''
Parallel execution of RTC Workflow
'''

import argparse
import multiprocessing
import os
import subprocess
import time
import yaml

from rtc.runconfig import RunConfig

import logging

from rtc.runconfig import RunConfig
from rtc.core import create_logger
import rtc_s1


logger = logging.getLogger('rtc_s1')


def get_rtc_s1_parser():
    '''Initialize YamlArgparse class and parse CLI arguments for OPERA RTC.
    Modified after copied from `rtc_s1.py`
    '''
    parser = argparse.ArgumentParser(description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_config_path',
                        type=str,
                        default=None,
                        help='Path to run config file')

    # Determine the default # of concurrent workers
    ncpu_default = os.cpu_count()
    if os.getenv('OMP_NUM_THREADS') is not None:
        omp_num_threads = int(os.getenv('OMP_NUM_THREADS'))
        ncpu_default = min(ncpu_default, omp_num_threads)

    parser.add_argument('-n',
                        dest='num_workers',
                        type=int,
                        default=ncpu_default,
                        help='Number of concurrent workers.')

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


def split_runconfig(path_runconfig_in, path_log_in):
    '''
    Split the input runconfig into single burst runconfigs.
    Writes out the runconfigs.
    Return the list of the burst runconfigs.

    Parameters:
    path_runconfig_in: str
        Path to the original runconfig
    path_log_in: str
        Path to the original logfile

    Returns:
    list_runconfig_burst: list(str)
        List of the burst runconfigs
    list_logfile_burst: list(str)
        List of the burst logfiles,
        which corresponds to `list_runconfig_burst`
    path_output: str
        Path to the output directory,
        which will be the temporary dir. of the following process
    '''
    with open(path_runconfig_in, 'r+', encoding='utf8') as fin:
        runconfig_dict_in = yaml.safe_load(fin.read())

    cfg = RunConfig.load_from_yaml(path_runconfig_in, 'rtc_s1')


    list_runconfig_burst = []
    list_logfile_burst = []

    # determine the bursrs to process
    list_burst_id = cfg.bursts.keys()
    #runconfig_dict_in['runconfig']['groups']['product_group']['output_dir']

    # determine the output directory for child process
    time_stamp = str(float(time.time()))
    dir_output = os.path.join(cfg.groups.product_group.scratch_path, f'temp_{time_stamp}')
    for burst_id in list_burst_id:
        path_temp_runconfig = os.path.join(cfg.scratch_path,
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
                                dir_output)

        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'save_secondary_layers_as_hdf5'],
                                 False)

        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'output_imagery_format'],
                                 'GTiff')

        set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig',
                                 'groups',
                                 'product_group',
                                 'save_mosaics'],
                                False)

        if runconfig_dict_out['runconfig']['groups']['product_group']['save_mosaics']:
            
            set_dict_item_recursive(runconfig_dict_out,
                                ['runconfig', 'groups', 'processing', 'geocoding', 'save_nlooks'],
                                True)

        list_runconfig_burst.append(path_temp_runconfig)

        if path_log_in is None:
            list_logfile_burst.append(None)
        else:
            list_logfile_burst.append(f'{path_log_in}.{burst_id}')

        with open(path_temp_runconfig, 'w+', encoding='utf8') as fout:
            yaml.dump(runconfig_dict_out, fout)

    return list_runconfig_burst, list_logfile_burst, dir_output


def set_dict_item_recursive(dict_in, list_path, val):
    '''
    Recursively locate the dict item in the multiple layer of dict,
    whose path is provided in the list of keys.
    Add or update the value of the item
    Create the key with empty dict when the key does not exist

    Parameters:
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
        #return dict_in
    else:
        key_next = list_path[0]
        if key_next not in dict_in.keys():
            dict_in[key_next] = {}
        set_dict_item_recursive(dict_in[key_next], list_path[1:], val)


def process_runconfig(path_runconfig_burst, path_logfile_burst, full_log_format):
    '''
    single worker to process runconfig from terminal using `subprocess`

    Parameters:
    path_runconfig_burst: str
        Path to the burst runconfig
    path_logfile_burst: str
        Path to the burst logfile
    full_log_format: bool
        Enable full formatting of log messages.
        See `get_rtc_s1_parser()`

    '''

    list_arg_subprocess = ['rtc_s1.py', path_runconfig_burst]
    if path_logfile_burst is not None:
        list_arg_subprocess += ['--log-file', path_logfile_burst]

    if full_log_format:
        list_arg_subprocess.append('--full-log-format')

    #list_arg_subprocess.append('--keep-scratch')
    #list_arg_subprocess.append('--timestamp')
    #list_arg_subprocess.append(timestamp)

    rtnval = subprocess.run(list_arg_subprocess)

    # TODO Add some routine to take a look into `rtnval` to see if everything is okay.

    os.remove(path_runconfig_burst)


def process_frame_parallel(arg_in):
    '''
    Take in the parsed arguments from CLI,
    split the original runconfign into bursts,
    and process them concurrently

    Parameter:
    arg_in: Namespace
        Parsed argument. See `get_rtc_s1_parser()`
    '''
    timestamp = str(float(time.time()))
    t0 = time.time()
    list_burst_runconfig, list_burst_log, dir_temp = split_runconfig(
                                                arg_in.run_config_path,
                                                arg_in.log_file)

    list_full_log_formatting = [arg_in.full_log_formatting] * len(list_burst_log)
    #list_timestamp = [timestamp] * len(list_burst_log)
    with multiprocessing.Pool(arg_in.num_workers) as p:
        p.starmap(process_runconfig,
                  zip(list_burst_runconfig,
                      list_burst_log,
                      list_full_log_formatting))
                      #list_timestamp))

    t1 = time.time()

    print(f'elapsed time: {t1-t0:06f} seconds.')
    return dir_temp


def main():
    '''
    Main entrypoint of the script
    '''
    t0 = time.time()
    parser  = get_rtc_s1_parser()

    # parse arguments
    args = parser.parse_args()

    # Spawn multiple processes for parallelization
    dir_temporary = process_frame_parallel(args)
    create_logger(args.log_file, args.full_log_formatting)

    # Get a runconfig dict from command line argumens
    cfg = RunConfig.load_from_yaml(args.run_config_path, 'rtc_s1')

    rtc_s1._load_parameters(cfg)

    # Run geocode burst workflow
    rtc_s1.run(cfg, tempdir=dir_temporary, skip_burst_process=True)
    t1 = time.time()

    print(f'Elapsed time: {t1-t0} seconds')

if __name__ == "__main__":
    # load arguments from command line
    main()
