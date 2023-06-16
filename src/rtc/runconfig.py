from __future__ import annotations
from dataclasses import dataclass
import os
from types import SimpleNamespace
import sys
import numpy as np

import isce3
from isce3.product import GeoGridParameters

import logging
import yamale
from ruamel.yaml import YAML

from rtc import helpers
from rtc.radar_grid import file_to_rdr_grid
from rtc.geogrid import generate_geogrids, generate_geogrids_from_db
from rtc.wrap_namespace import wrap_namespace, unwrap_to_dict
from s1reader.s1_burst_slc import Sentinel1BurstSlc
from s1reader.s1_orbit import get_orbit_file_from_list
from s1reader.s1_reader import load_bursts

STATIC_LAYERS_PRODUCT_TYPE = 'RTC_S1_SL'

logger = logging.getLogger('rtc_s1')


def load_parameters(cfg):
    '''
    Load RTC-S1 specific parameters.
    '''

    geocode_namespace = cfg.groups.processing.geocoding
    rtc_namespace = cfg.groups.processing.rtc

    if geocode_namespace.clip_max is None:
        geocode_namespace.clip_max = np.nan

    if geocode_namespace.clip_min is None:
        geocode_namespace.clip_min = np.nan

    if geocode_namespace.geogrid_upsampling is None:
        geocode_namespace.geogrid_upsampling = 1.0

    if geocode_namespace.memory_mode == 'single_block':
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.SingleBlock
    elif geocode_namespace.memory_mode == 'geogrid':
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.BlocksGeogrid
    elif geocode_namespace.memory_mode == 'geogrid_and_radargrid':
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.BlocksGeogridAndRadarGrid
    elif (geocode_namespace.memory_mode == 'auto' or
          geocode_namespace.memory_mode is None):
        geocode_namespace.memory_mode = \
            isce3.core.GeocodeMemoryMode.Auto
    else:
        err_msg = f"ERROR memory_mode: {geocode_namespace.memory_mode}"
        raise ValueError(err_msg)

    rtc_output_type = rtc_namespace.output_type
    if rtc_output_type == 'sigma0':
        rtc_namespace.output_type = \
            isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
    else:
        rtc_namespace.output_type = \
            isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT

    if rtc_namespace.input_terrain_radiometry == "sigma0":
        rtc_namespace.input_terrain_radiometry = \
            isce3.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
    else:
        rtc_namespace.input_terrain_radiometry = \
            isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

    if rtc_namespace.rtc_min_value_db is None:
        rtc_namespace.rtc_min_value_db = np.nan

    # Update the DEM interpolation method
    dem_interp_method = \
        cfg.groups.processing.dem_interpolation_method

    if dem_interp_method == 'biquintic':
        dem_interp_method_enum = isce3.core.DataInterpMethod.BIQUINTIC
    elif dem_interp_method == 'sinc':
        dem_interp_method_enum = isce3.core.DataInterpMethod.SINC
    elif dem_interp_method == 'bilinear':
        dem_interp_method_enum = isce3.core.DataInterpMethod.BILINEAR
    elif dem_interp_method == 'bicubic':
        dem_interp_method_enum = isce3.core.DataInterpMethod.BICUBIC
    elif dem_interp_method == 'nearest':
        dem_interp_method_enum = isce3.core.DataInterpMethod.NEAREST
    else:
        err_msg = ('ERROR invalid DEM interpolation method:'
                   f' {dem_interp_method}')
        raise ValueError(err_msg)

    cfg.groups.processing.dem_interpolation_method_enum = \
        dem_interp_method_enum


def load_validate_yaml(yaml_path: str) -> dict:
    """Initialize RunConfig class with options from given yaml file.

    Parameters
    ----------
    yaml_path : str
        Path to yaml file containing the options to load
    """
    schema_filepath = f'{helpers.WORKFLOW_SCRIPTS_DIR}/schemas/rtc_s1.yaml'
    try:
        schema = yamale.make_schema(schema_filepath, parser='ruamel')
    except:
        err_str = f'unable to load schema from file {schema_filepath}'
        logger.error(err_str)
        raise ValueError(err_str)

    # load yaml file or string from command line
    if os.path.isfile(yaml_path):
        try:
            data = yamale.make_data(yaml_path, parser='ruamel')
        except yamale.YamaleError as yamale_err:
            err_str = (f'Yamale unable to load runconfig yaml'
                       f'{yaml_path} for validation.')
            logger.error(err_str)
            raise yamale.YamaleError(err_str) from yamale_err
    else:
        error_msg = f'ERROR file not found: {yaml_path}'
        raise FileNotFoundError(error_msg)

    # validate yaml file taken from command line
    try:
        yamale.validate(schema, data)
    except yamale.YamaleError as yamale_err:
        err_str = (f'Validation failed for runconfig yaml {yaml_path}.')
        logger.error(err_str)
        raise yamale.YamaleError(err_str) from yamale_err

    # load default runconfig
    parser = YAML(typ='safe')

    with open(yaml_path, 'r') as f_yaml:
        user_cfg = parser.load(f_yaml)

    product_type = user_cfg['runconfig']['groups']['primary_executable'][
        'product_type']
    if (product_type == STATIC_LAYERS_PRODUCT_TYPE):
        default_cfg_path = (f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/'
                            f'rtc_s1_sl.yaml')
        print('Loading RTC-S1 runconfig default for static layers')
    else:
        default_cfg_path = (f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/'
                            f'rtc_s1.yaml')
        print('Loading RTC-S1 runconfig default')

    with open(default_cfg_path, 'r') as f_default:
        default_cfg = parser.load(f_default)

    # Copy user-supplied configuration options into default runconfig
    helpers.deep_update(default_cfg, user_cfg)

    # Validate YAML values under groups dict
    validate_group_dict(default_cfg['runconfig']['groups'])

    return default_cfg


def validate_group_dict(group_cfg: dict) -> None:
    """Check and validate runconfig entries.

    Parameters
    ----------
    group_cfg : dict
        Dictionary storing runconfig options to validate
    """

    # Check 'input_file_group' section of runconfig
    input_group = group_cfg['input_file_group']

    # Check SAFE files
    run_pol_mode = group_cfg['processing']['polarization']
    safe_pol_modes = []
    for safe_file in input_group['safe_file_path']:
        # Check if files exists
        helpers.check_file_path(safe_file)

        # Get and save safe pol mode ('SV', 'SH', 'DH', 'DV')
        safe_pol_mode = helpers.get_file_polarization_mode(safe_file)
        safe_pol_modes.append(safe_pol_mode)

        # Raise error if given co-pol file and expecting cross-pol or dual-pol
        if run_pol_mode != 'co-pol' and safe_pol_mode in ['SV', 'SH']:
            err_str = f'{run_pol_mode} polarization lacks cross-pol in {safe_file}'
            logger.error(err_str)
            raise ValueError(err_str)

    # Check SAFE file pols consistency. i.e. no *H/*V with *V/*H respectively
    if len(safe_pol_modes) > 1:
        first_safe_pol_mode = safe_pol_modes[0][1]
        for safe_pol_mode in safe_pol_modes[1:]:
            if safe_pol_mode[1] != first_safe_pol_mode:
                err_str = 'SH/SV SAFE file mixed with DH/DV'
                logger.error(err_str)
                raise ValueError(err_str)

    for orbit_file in input_group['orbit_file_path']:
        helpers.check_file_path(orbit_file)

    # Check 'dynamic_ancillary_file_groups' section of runconfig
    # Check that DEM file exists and is GDAL-compatible
    dem_path = group_cfg['dynamic_ancillary_file_group']['dem_file']
    helpers.check_file_path(dem_path)
    helpers.check_dem(dem_path)

    # Check 'product_group' section of runconfig.
    # Check that directories herein have writing permissions
    product_group = group_cfg['product_group']
    helpers.check_write_dir(product_group['product_path'])
    helpers.check_write_dir(product_group['scratch_path'])


def runconfig_to_bursts(cfg: SimpleNamespace):
    '''Return bursts based on parameters in given runconfig

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration of bursts to be loaded.

    Returns
    -------
    bursts : list[Sentinel1BurstSlc]
        List of bursts loaded according to given configuration.
    orbit_file_path : str
        Orbit file path
    '''

    # dict to store list of bursts keyed by burst_ids
    bursts = {}

    # extract given SAFE zips to find bursts identified in cfg.burst_id
    for safe_file in cfg.input_file_group.safe_file_path:
        # get orbit file
        orbit_file_path = get_orbit_file_from_list(
            safe_file,
            cfg.input_file_group.orbit_file_path)

        if not orbit_file_path:
            err_str = ("No orbit file correlates to safe file:"
                       f" {os.path.basename(safe_file)}")
            logger.error(err_str)
            raise ValueError(err_str)

        # from SAFE file mode, create dict of runconfig pol mode to
        # polarization(s)
        safe_pol_mode = helpers.get_file_polarization_mode(safe_file)
        if safe_pol_mode == 'SV':
            mode_to_pols = {'co-pol': ['VV']}
        elif safe_pol_mode == 'DV':
            mode_to_pols = {'co-pol': ['VV'], 'cross-pol': ['VH'],
                            'dual-pol': ['VV', 'VH']}
        elif safe_pol_mode == 'SH':
            mode_to_pols = {'co-pol': ['HH']}
        else:
            mode_to_pols = {'co-pol': ['HH'], 'cross-pol': ['HV'],
                            'dual-pol': ['HH', 'HV']}
        pols = mode_to_pols[cfg.processing.polarization]

        # zip pol and IW subswath indices together
        i_subswaths = [1, 2, 3]
        pol_subswath_index_pairs = [(pol, i)
                                    for pol in pols for i in i_subswaths]

        # list of burst ID + polarization tuples
        # used to prevent reference repeats
        id_pols_found = []

        # loop over pol and subswath index combinations
        for pol, i_subswath in pol_subswath_index_pairs:

            # loop over burst objs extracted from SAFE zip
            for burst in load_bursts(safe_file, orbit_file_path, i_subswath,
                                     pol, flag_apply_eap=False):
                # get burst ID
                burst_id = str(burst.burst_id)

                # is burst_id wanted? skip if not given in config
                if (cfg.input_file_group.burst_id is not None and
                        burst_id not in cfg.input_file_group.burst_id):
                    continue

                # get polarization and save as tuple with burst ID
                pol = burst.polarization
                id_pol = (burst_id, pol)

                # has burst_id + pol combo been found?
                burst_id_pol_exist = id_pol in id_pols_found
                if burst_id_pol_exist:
                    continue

                id_pols_found.append(id_pol)

                # append burst to bursts dict
                if burst_id not in bursts.keys():
                    bursts[burst_id] = {}
                bursts[burst_id][pol] = burst

    # check if no bursts were found
    if not bursts:
        err_str = ("Could not find any of the burst IDs in the provided safe"
                   " files")
        logger.error(err_str)
        raise ValueError(err_str)

    return bursts, orbit_file_path


def get_ref_radar_grid_info(ref_path, burst_id):
    ''' Find all reference radar grids info

    Parameters
    ----------
    ref_path: str
        Path where reference radar grids processing is stored
    burst_id: str
        Burst IDs for reference radar grids

    Returns
    -------
    ref_radar_grids:
        reference radar path and grid values found associated with
        burst ID keys
    '''
    rdr_grid_files = f'{ref_path}/radar_grid.txt'

    if not os.path.isfile(rdr_grid_files):
        raise FileNotFoundError('No reference radar grids not found in'
                                f' {ref_path}')

    ref_rdr_path = os.path.dirname(rdr_grid_files)
    ref_rdr_grid = file_to_rdr_grid(rdr_grid_files)

    return ReferenceRadarInfo(ref_rdr_path, ref_rdr_grid)


def check_geogrid_dict(geocode_cfg: dict) -> None:

    # check output EPSG
    output_epsg = geocode_cfg['output_epsg']
    if output_epsg is not None:
        # check 1024 <= output_epsg <= 32767:
        if output_epsg < 1024 or 32767 < output_epsg:
            err_str = f'output epsg {output_epsg} in YAML out of bounds'
            logger.error(err_str)
            raise ValueError(err_str)

    for xy in 'xy':
        # check posting value in current axis
        posting_key = f'{xy}_posting'
        if geocode_cfg[posting_key] is not None:
            posting = geocode_cfg[posting_key]
            if posting <= 0:
                err_str = '{xy} posting from config of {posting} <= 0'
                logger.error(err_str)
                raise ValueError(err_str)

        # check snap value in current axis
        snap_key = f'{xy}_snap'
        if geocode_cfg[snap_key] is not None:
            snap = geocode_cfg[snap_key]
            if snap <= 0:
                err_str = '{xy} snap from config of {snap} <= 0'
                logger.error(err_str)
                raise ValueError(err_str)


@dataclass(frozen=True)
class ReferenceRadarInfo:
    path: str
    grid: isce3.product.RadarGridParameters


@dataclass(frozen=True)
class RunConfig:
    '''dataclass containing RTC runconfig'''
    # workflow name
    name: str
    # runconfig options converted from dict
    groups: SimpleNamespace
    # list of lists where bursts in interior list have a common burst_id
    bursts: list[Sentinel1BurstSlc]
    # dict of reference radar paths and grids values keyed on burst ID
    # (empty/unused if rdr2geo)
    reference_radar_info: ReferenceRadarInfo
    # run config path
    run_config_path: str
    # output product geogrid
    geogrid: GeoGridParameters
    # dict of geogrids associated to burst IDs
    geogrids: dict[str, GeoGridParameters]
    # orbit file path
    orbit_file_path: str

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> RunConfig:
        """Initialize RunConfig class with options from given yaml file.

        Parameters
        ----------
        yaml_path : str
            Path to yaml file containing the options to load
        workflow_name: str
            Name of the workflow for which uploading default options
        """
        cfg = load_validate_yaml(yaml_path)
        groups_cfg = cfg['runconfig']['groups']

        # Read mosaic dict
        mosaic_dict = groups_cfg['processing']['mosaicking']
        check_geogrid_dict(mosaic_dict['mosaic_geogrid'])

        # Read geocoding dict
        geocoding_dict = groups_cfg['processing']['geocoding']
        check_geogrid_dict(geocoding_dict['bursts_geogrid'])

        # Convert runconfig dict to SimpleNamespace
        sns = wrap_namespace(groups_cfg)

        # Load bursts
        bursts, orbit_file_path = runconfig_to_bursts(sns)

        # Load geogrids
        burst_database_file = groups_cfg['static_ancillary_file_group'][
            'burst_database_file']
        if burst_database_file is None:
            geogrid_all, geogrids = generate_geogrids(bursts, geocoding_dict,
                                                      mosaic_dict)
        else:
            geogrid_all, geogrids = generate_geogrids_from_db(
                bursts, geocoding_dict, mosaic_dict, burst_database_file)
        
        # Empty reference dict for base runconfig class constructor
        empty_ref_dict = {}

        return cls(cfg['runconfig']['name'], sns, bursts, empty_ref_dict,
                   yaml_path, geogrid_all, geogrids, orbit_file_path)

    @property
    def geocoding_params(self) -> dict:
        return self.groups.processing.geocoding

    @property
    def burst_id(self) -> list[str]:
        return self.groups.input_file_group.burst_id

    @property
    def dem(self) -> str:
        return self.groups.dynamic_ancillary_file_group.dem_file

    @property
    def dem_file_description(self) -> str:
        return self.groups.dynamic_ancillary_file_group.dem_file_description

    @property
    def is_reference(self) -> bool:
        return self.groups.input_file_group.reference_burst.is_reference

    @property
    def orbit_path(self) -> bool:
        return self.groups.input_file_group.orbit_file_path

    @property
    def polarization(self) -> list[str]:
        return self.groups.processing.polarization

    @property
    def product_path(self):
        return self.groups.product_group.product_path

    @property
    def reference_path(self) -> str:
        return self.groups.input_file_group.reference_burst.file_path

    @property
    def rdr2geo_params(self) -> dict:
        return self.groups.processing.rdr2geo

    @property
    def geo2rdr_params(self) -> dict:
        return self.groups.processing.geo2rdr

    @property
    def split_spectrum_params(self) -> dict:
        return self.groups.processing.range_split_spectrum

    @property
    def resample_params(self) -> dict:
        return self.groups.processing.resample

    @property
    def safe_files(self) -> list[str]:
        return self.groups.input_file_group.safe_file_path

    @property
    def product_id(self):
        return self.groups.product_group.product_id

    @property
    def scratch_path(self):
        return self.groups.product_group.scratch_path

    @property
    def gpu_enabled(self):
        return self.groups.worker.gpu_enabled

    @property
    def gpu_id(self):
        return self.groups.worker.gpu_id

    def as_dict(self):
        ''' Convert self to dict for write to YAML/JSON

        Unable to dataclasses.asdict() because isce3 objects can not be pickled
        '''
        self_as_dict = {}
        for key, val in self.__dict__.items():
            if key == 'groups':
                val = unwrap_to_dict(val)
            elif key == 'bursts':
                # just date in datetime obj as string
                date_str = lambda b : b.sensing_start.date().strftime('%Y%m%d')

                # create an unique burst key
                burst_as_key = lambda b : '_'.join([str(b.burst_id),
                                                    date_str(b),
                                                    b.polarization])

                val = {burst_as_key(burst): burst.as_dict() for burst in val}
            self_as_dict[key] = val
        return self_as_dict

    def to_yaml(self):
        self_as_dict = self.as_dict()
        yaml = YAML(typ='safe')
        yaml.dump(self_as_dict, sys.stdout, indent=4)
