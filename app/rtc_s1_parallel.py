#!/usr/bin/env python

'''
RTC Workflow
'''

import argparse
import multiprocessing
import os
import subprocess
import time
import yaml

from rtc.runconfig import RunConfig


def get_rtc_s1_parser():
    '''Initialize YamlArgparse class and parse CLI arguments for OPERA RTC.
    Modified after copied from `rtc_s1.py`
    '''
    parser = argparse.ArgumentParser(description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_config_path',
                        type=str,
                        nargs='1',
                        default=None,
                        help='Path to run config file')

    parser.add_argument('-n',
                        dest='num_workers',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of concurrent workers')

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
    '''
    with open(path_runconfig_in, 'r+', encoding='utf8') as fin:
        runconfig_dict_in = yaml.safe_load(fin.read())

    cfg = RunConfig.load_from_yaml(path_runconfig_in, 'rtc_s1')


    list_runconfig_burst = []
    list_logfile_burst = []

    # determine the bursrs to process
    list_burst_id = cfg.bursts.keys()

    for burst_id in list_burst_id:
        path_temp_runconfig = os.path.join(cfg.scratch_path,
                                           f'burst_runconfig_{burst_id}.yaml')

        runconfig_dict_out = runconfig_dict_in.copy()
        runconfig_dict_out['runconfig']['groups']['input_file_group']['burst_id'] = [burst_id]
        list_runconfig_burst.append(path_temp_runconfig)

        if path_log_in is None:
            list_logfile_burst.append(None)
        else:
            list_logfile_burst.append(f'{path_log_in}.{burst_id}')

        with open(path_temp_runconfig, 'w+', encoding='utf8') as fout:
            yaml.dump(runconfig_dict_out, fout)

    return list_runconfig_burst, list_logfile_burst


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

    rtnval = subprocess.run(list_arg_subprocess)
    os.remove(path_runconfig_burst)

    # TODO add log file into the argument for `subprocess.run()`


def process_frame_parallel(arg_in):
    '''
    Take in the parsed arguments from CLI,
    split the original runconfign into bursts,
    and process them concurrently

    Parameter:
    arg_in: Namespace
        Parsed argument. See `get_rtc_s1_parser()`
    '''
    t0 = time.time()
    list_burst_runconfig, list_burst_log = split_runconfig(
                                                arg_in.run_config_path,
                                                arg_in.log_file)

    with multiprocessing.Pool(arg_in.num_workers) as p:
        p.starmap(process_runconfig,
              zip(list_burst_runconfig,
                  list_burst_log,
                  [arg_in.full_log_formatting]*len(list_burst_log)))

    t1 = time.time()

    print(f'Took {t1-t0:06f} seconds to process.')

if __name__ == "__main__":
    # load arguments from command line
    parser  = get_rtc_s1_parser()

    # parse arguments
    args = parser.parse_args()

    process_frame_parallel(args)
