import argparse

class YamlArgparse():
    def __init__(self):
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

        # parse arguments
        self.args = parser.parse_args()

    @property
    def run_config_path(self) -> str:
        return self.args.run_config_path
