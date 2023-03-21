import os
import re
from setuptools import setup

def _get_version():
    """Returns the RTC-S1 science application software version from the
    file `src/rtc/version.py`

       Returns
       -------
       version : str
            RTC-S1 science application software version
    """

    version_file = os.path.join('src','rtc','version.py')

    with open(version_file, 'r') as f:
        text = f.read()

    # Get first match of the version number contained in the version file
    # This regex should match a pattern like: VERSION = '3.2.5', but it
    # allows for varying spaces, number of major/minor versions,
    # and quotation mark styles.
    p = re.search("VERSION[ ]*=[ ]*['\"]\d+([.]\d+)*['\"]", text)

    # Check that the version file contains properly formatted text string
    if p is None:
        raise ValueError(
            f'Version file {version_file} not properly formatted.'
            " It should contain text matching e.g. VERSION = '2.3.4'")

    # Extract just the numeric version number from the string
    p = re.search("\d+([.]\d+)*", p.group(0))

    return p.group(0)


__version__ = version = VERSION = _get_version()

print(f'RTC-S1 SAS version {version}')

long_description = ''

package_data_dict = {}

package_data_dict['rtc'] = [
    os.path.join('defaults', 'rtc_s1.yaml'),
    os.path.join('schemas', 'rtc_s1.yaml')]

setup(
    name = 'rtc',
    version = version,
    description = ('OPERA Radiometric Terrain-Corrected (RTC) SAR backscatter'
                   ' from Sentinel-1 Science Application Software (SAS)'),
    package_dir = {'rtc': 'src/rtc'},
    include_package_data = True,
    package_data = package_data_dict,
    classifiers = ['Programming Language :: Python',],
    scripts = ['app/rtc_s1.py'],
    install_requires = ['argparse', 'numpy', 'yamale',
                       'scipy', 'pytest', 'requests',
                       'pyproj', 'shapely', 'matplotlib'],
    url = 'https://github.com/opera-adt/RTC',
    license = 'Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
)
