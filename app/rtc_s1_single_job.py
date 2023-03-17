#!/usr/bin/env python

'''
RTC-S1 Science Application Software (single job)
'''

import logging
from rtc.runconfig import RunConfig, load_parameters
from rtc.core import create_logger
from rtc.rtc_s1_single_job import get_rtc_s1_parser, run_single_job


logger = logging.getLogger('rtc_s1')

if __name__ == "__main__":
    '''
    Run geocode rtc workflow from command line
    '''

    # load arguments from command line
    parser  = get_rtc_s1_parser()

    # parse arguments
    args = parser.parse_args()

    # create logger
    logger = create_logger(args.log_file, args.full_log_formatting)

    # Get a runconfig dict from command line argumens
    cfg = RunConfig.load_from_yaml(args.run_config_path, 'rtc_s1')

    load_parameters(cfg)

    # Run geocode burst workflow
    run_single_job(cfg)
