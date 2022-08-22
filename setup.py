import os
from setuptools import setup

__version__ = version = VERSION = '0.1'

long_description = ''

package_data_dict = {}

package_data_dict['rtc'] = [
    os.path.join('defaults', 'rtc_s1.yaml'),
    os.path.join('schemas', 'rtc_s1.yaml')]

setup(
    name = 'rtc',
    version = version,
    description = 'OPERA Radiometric Terrain-Corrected (RTC) Product',
    package_dir = {'rtc': 'src/rtc'},
    #packages = ['rtc'],
    include_package_data = True,
    package_data = package_data_dict,
    classifiers = ['Programming Language :: Python',],
    scripts = ['app/rtc_s1.py'],
    install_requires = ['argparse', 'numpy', 'yamale',
                       'scipy', 'pytest', 'requests'],
    url = 'https://github.com/opera-adt/RTC',
    author = ('Gustavo H. X. Shiroma', 'Seongsu Jeong'),
    author_email = ('gustavo.h.shiroma@jpl.nasa.gov, seongsu.jeong@jpl.nasa.gov'),
    license = 'Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
)
