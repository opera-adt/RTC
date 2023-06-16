#!/usr/bin/env python

'''
RTC-S1 Science Application Software
'''

import logging
from rtc.runconfig import RunConfig, load_parameters
from rtc.core import create_logger
from rtc.rtc_s1_single_job import get_rtc_s1_parser
from rtc.rtc_s1 import run_parallel


logger = logging.getLogger('rtc_s1')

def main():
    '''
    Main entrypoint of the script
    '''
    parser = get_rtc_s1_parser()
    # parse arguments
    args = parser.parse_args()

    # Spawn multiple processes for parallelization
    create_logger(args.log_file, args.full_log_formatting)

    # Get a runconfig dict from command line argumens
    cfg = RunConfig.load_from_yaml(args.run_config_path)

    load_parameters(cfg)

    # Run geocode burst workflow
    path_logfile_parent = args.log_file
    flag_full_log_formatting = args.full_log_formatting
    run_parallel(cfg, path_logfile_parent, flag_full_log_formatting)


if __name__ == "__main__":
    # load arguments from command line
    main()