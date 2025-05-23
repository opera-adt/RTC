#!/usr/bin/env python

import os
import re
import numpy as np
import h5py
import logging
from datetime import datetime
from osgeo import gdal

import isce3
import shapely

from s1reader.s1_burst_slc import Sentinel1BurstSlc
from s1reader.version import release_version
from rtc.runconfig import RunConfig, STATIC_LAYERS_PRODUCT_TYPE
from rtc.version import VERSION as SOFTWARE_VERSION

from nisar.workflows.h5_prep import set_get_geo_info

# Data base HDF5 group
DATE_TIME_METADATA_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
DATE_TIME_FILENAME_FORMAT = '%Y%m%dT%H%M%SZ'

DATA_BASE_GROUP = '/data'
PRODUCT_SPECIFICATION_VERSION = 1.0

logger = logging.getLogger('rtc_s1')

LAYER_NAME_VV = 'VV'
LAYER_NAME_VH = 'VH'
LAYER_NAME_HH = 'HH'
LAYER_NAME_HV = 'HV'
LAYER_NAME_LAYOVER_SHADOW_MASK = 'mask'
LAYER_NAME_RTC_ANF_GAMMA0_TO_BETA0 = 'rtc_anf_gamma0_to_beta0'
LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0 = 'rtc_anf_gamma0_to_sigma0'
LAYER_NAME_RTC_ANF_SIGMA0_TO_BETA0 = 'rtc_anf_sigma0_to_beta0'
LAYER_NAME_RTC_ANF_BETA0_TO_BETA0 = 'rtc_anf_beta0_to_beta0'
LAYER_NAME_NUMBER_OF_LOOKS = 'number_of_looks'
LAYER_NAME_INCIDENCE_ANGLE = 'incidence_angle'
LAYER_NAME_LOCAL_INCIDENCE_ANGLE = 'local_incidence_angle'
LAYER_NAME_PROJECTION_ANGLE = 'projection_angle'
LAYER_NAME_RTC_ANF_PROJECTION_ANGLE = 'rtc_anf_projection_angle'
LAYER_NAME_RANGE_SLOPE = 'range_slope'
LAYER_NAME_DEM = 'interpolated_dem'

# RTC-S1 product layer names
layer_hdf5_dict = {
    LAYER_NAME_VV: 'VV',
    LAYER_NAME_VH: 'VH',
    LAYER_NAME_HH: 'HH',
    LAYER_NAME_HV: 'HV',
    LAYER_NAME_LAYOVER_SHADOW_MASK: 'mask',
    LAYER_NAME_RTC_ANF_GAMMA0_TO_BETA0:
        'rtcAreaNormalizationFactorGamma0ToBeta0',
    LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0:
        'rtcAreaNormalizationFactorGamma0ToSigma0',
    LAYER_NAME_RTC_ANF_SIGMA0_TO_BETA0:
        'rtcAreaNormalizationFactorSigma0ToBeta0',
    LAYER_NAME_RTC_ANF_BETA0_TO_BETA0: 'rtcAreaNormalizationFactorBeta0ToBeta0',
    LAYER_NAME_NUMBER_OF_LOOKS: 'numberOfLooks',
    LAYER_NAME_INCIDENCE_ANGLE: 'incidenceAngle',
    LAYER_NAME_LOCAL_INCIDENCE_ANGLE: 'localIncidenceAngle',
    LAYER_NAME_PROJECTION_ANGLE: 'projectionAngle',
    LAYER_NAME_RTC_ANF_PROJECTION_ANGLE:
        'rtcAreaNormalizationFactorProjectionAngle',
    LAYER_NAME_RANGE_SLOPE: 'rangeSlope',
    LAYER_NAME_DEM: 'interpolatedDem'
}

# RTC-S1 product layer names
layer_names_dict = {
    LAYER_NAME_VV: 'RTC-S1 VV Backscatter',
    LAYER_NAME_VH: 'RTC-S1 VH Backscatter',
    LAYER_NAME_HH: 'RTC-S1 HH Backscatter',
    LAYER_NAME_HV: 'RTC-S1 HV Backscatter',
    LAYER_NAME_LAYOVER_SHADOW_MASK: 'Mask Layer',
    LAYER_NAME_RTC_ANF_GAMMA0_TO_BETA0: ('RTC Area Normalization Factor'
                                         ' Gamma0 to Beta0'),
    LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0: ('RTC Area Normalization Factor'
                                          ' Gamma0 to Sigma0'),
    LAYER_NAME_RTC_ANF_SIGMA0_TO_BETA0: ('RTC Area Normalization Factor'
                                         ' Sigma0 to Beta0'),
    LAYER_NAME_RTC_ANF_BETA0_TO_BETA0: ('RTC Area Normalization Factor'
                                        ' Beta0 to Beta0'),
    LAYER_NAME_NUMBER_OF_LOOKS: 'Number of Looks',
    LAYER_NAME_INCIDENCE_ANGLE: 'Incidence Angle',
    LAYER_NAME_LOCAL_INCIDENCE_ANGLE: 'Local Incidence Angle',

    # TODO improve description below
    LAYER_NAME_PROJECTION_ANGLE: 'Projection Angle',
    LAYER_NAME_RTC_ANF_PROJECTION_ANGLE: (
        'RTC Area Normalization Factor'
        ' Gamma0 to Beta0 (Projection Angle - ProjectionAngle)'),
    LAYER_NAME_RANGE_SLOPE: 'Range Slope',
    LAYER_NAME_DEM: 'Digital Elevation Model (DEM)'
}

# RTC-S1 product layer description dict
layer_description_dict = {
    LAYER_NAME_VV: ('Radiometric terrain corrected Sentinel-1 VV backscatter'
                    ' coefficient normalized to gamma0'),
    LAYER_NAME_VH: ('Radiometric terrain corrected Sentinel-1 VH backscatter'
                    ' coefficient normalized to gamma0'),
    LAYER_NAME_HH: ('Radiometric terrain corrected Sentinel-1 HH backscatter'
                    ' coefficient normalized to gamma0'),
    LAYER_NAME_HV: ('Radiometric terrain corrected Sentinel-1 HV backscatter'
                    ' coefficient normalized to gamma0'),
    LAYER_NAME_LAYOVER_SHADOW_MASK: ('Mask Layer. Values: 0: not'
                                     ' masked; 1: shadow; 2: layover;'
                                     ' 3: layover and shadow;'
                                     ' 255: invalid/fill value'),
    LAYER_NAME_RTC_ANF_GAMMA0_TO_BETA0: ('Radiometric terrain correction (RTC)'
                                         ' area normalization factor (ANF)'
                                         ' gamma0 to beta0'),
    LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0: ('Radiometric terrain correction'
                                          ' (RTC) area normalization factor'
                                          ' (ANF) gamma0 to sigma0'),
    LAYER_NAME_RTC_ANF_SIGMA0_TO_BETA0: ('Radiometric terrain correction (RTC)'
                                         ' area normalization factor (ANF)'
                                         ' sigma0 to beta0'),
    LAYER_NAME_RTC_ANF_BETA0_TO_BETA0: ('Radiometric terrain correction (RTC)'
                                        ' area normalization factor (ANF)'
                                        ' beta0 to beta0'),
    LAYER_NAME_NUMBER_OF_LOOKS: 'Number of looks',
    LAYER_NAME_INCIDENCE_ANGLE: ('Incidence angle is defined as the angle'
                                 ' between the line-of-sight (LOS) vector and'
                                 ' the normal to the ellipsoid at the target'
                                 ' height'),
    LAYER_NAME_LOCAL_INCIDENCE_ANGLE: ('Local incidence angle is defined as'
                                       ' the angle between the line-of-sight'
                                       ' (LOS) vector and the normal to the'
                                       ' ellipsoid at the target height'),

    # TODO improve description below
    LAYER_NAME_PROJECTION_ANGLE: 'Projection angle (psi)',
    LAYER_NAME_RTC_ANF_PROJECTION_ANGLE: (
        'Radiometric terrain correction (RTC)'
        ' area normalization factor (ANF) '
        ' gamma0 to beta0 computed with'
        ' the projection angle method'),
    LAYER_NAME_RANGE_SLOPE: 'Range slope',
    LAYER_NAME_DEM: 'Digital elevation model (DEM)'
}


def get_polygon_wkt(burst_in: Sentinel1BurstSlc):
    '''
    Get the WKT representation of the burst's bounding polygon
    It returns "POLYGON" when
    there is only one polygon that defines the burst's border
    It returns "MULTIPOLYGON" when
    there is more than one polygon that defines the burst's border

    Parameters
    -----------
    burst_in: Sentinel1BurstSlc
        Input burst

    Return:
    _ : str
        "POLYGON" or "MULTIPOLYGON" in WKT
        as the bounding polygon of the input burst
    '''

    if len(burst_in.border) == 1:
        geometry_polygon = shapely.geometry.Polygon(burst_in.border[0])
    else:
        geometry_polygon = shapely.geometry.MultiPolygon(burst_in.border)
    if geometry_polygon.is_empty:
        error_msg = f'empty bounding polygon for burst ID {burst_in.burst_id}'
        raise RuntimeError(error_msg)
    if not geometry_polygon.is_valid:
        error_msg = f'invalid bounding polygon for burst ID {burst_in.burst_id}'
        raise RuntimeError(error_msg)
    return geometry_polygon.wkt


def save_hdf5_file(hdf5_obj, output_hdf5_file, clip_max,
                   clip_min, output_radiometry_str,
                   geogrid, pol_list, geo_burst_filename, nlooks_file,
                   rtc_anf_file, rtc_anf_file_str,
                   rtc_anf_gamma0_to_sigma0_file,
                   layover_shadow_mask_file,
                   radar_grid_file_dict,
                   save_imagery=True, save_secondary_layers=True):

    # save grids metadata
    h5_ds = f'{DATA_BASE_GROUP}/listOfPolarizations'
    if h5_ds in hdf5_obj:
        del hdf5_obj[h5_ds]
    pol_list_s2 = np.array(pol_list, dtype='S2')
    dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
    dset.attrs['description'] = np.bytes_(
                'List of processed polarization layers')

    # save geogrid coordinates
    yds, xds = set_get_geo_info(hdf5_obj, DATA_BASE_GROUP, geogrid)

    if save_imagery:
        # save RTC imagery
        save_hdf5_dataset(geo_burst_filename, hdf5_obj, DATA_BASE_GROUP,
                          yds, xds, pol_list,
                          long_name=output_radiometry_str,
                          units='',
                          valid_min=clip_min,
                          valid_max=clip_max)

    if save_secondary_layers:
        # save nlooks
        if nlooks_file:
            save_hdf5_dataset(nlooks_file, hdf5_obj, DATA_BASE_GROUP,
                              yds, xds,
                              LAYER_NAME_NUMBER_OF_LOOKS,
                              units='',
                              valid_min=0)

        # save RTC ANF
        if rtc_anf_file:
            save_hdf5_dataset(rtc_anf_file, hdf5_obj, DATA_BASE_GROUP,
                              yds, xds,
                              rtc_anf_file_str,
                              units='',
                              valid_min=0)

        # save RTC ANF gamma0 to sigma0
        if rtc_anf_gamma0_to_sigma0_file:
            save_hdf5_dataset(rtc_anf_gamma0_to_sigma0_file, hdf5_obj,
                              DATA_BASE_GROUP,
                              yds, xds,
                              LAYER_NAME_RTC_ANF_GAMMA0_TO_SIGMA0,
                              units='',
                              valid_min=0)

        # save layover shadow mask
        if layover_shadow_mask_file:
            save_hdf5_dataset(layover_shadow_mask_file, hdf5_obj,
                              DATA_BASE_GROUP,
                              yds, xds,
                              LAYER_NAME_LAYOVER_SHADOW_MASK,
                              units='',
                              valid_min=0)

        for ds_hdf5, filename in radar_grid_file_dict.items():
            save_hdf5_dataset(filename, hdf5_obj, DATA_BASE_GROUP, yds, xds,
                              ds_hdf5, long_name='', units='')

    logger.info(f'file saved: {output_hdf5_file}')


def get_product_version(product_version_runconfig):
    '''
    Returns the product version from the product
    version defined by the user in the runconfig.
    If the runconfig product version is not set, use
    the SOFTWARE_VERSION instead

    Parameters
    ---------
    product_version_runconfig: scalar
        RunConfig product version

    Returns
    -------
    product_version: scalar
        Product version
    '''
    if product_version_runconfig:
        return product_version_runconfig
    return SOFTWARE_VERSION


def create_hdf5_file(product_id, output_hdf5_file, orbit, burst, cfg,
                     processing_datetime, is_mosaic):
    '''Create HDF5 file

    Parameters
    -----------
    product_id: str
        Product ID
    output_hdf5_file: h5py.File
        HDF5 object into which write the metadata
    orbit: isce3.core.Orbit
        Orbit ISCE3 object
    burst: Sentinel1BurstSlc
        Source burst of the RTC
    cfg: RunConfig
        A class that contains the information defined in runconfig
    processing_datetime: datetime
        Processing datetime object
    is_mosaic: bool
        Flag to indicate whether the RTC-S1 product is a mosaic (True)
        or burst (False) product
    '''

    hdf5_obj = h5py.File(output_hdf5_file, 'w')
    hdf5_obj.attrs['Conventions'] = np.bytes_("CF-1.8")
    hdf5_obj.attrs["contact"] = np.bytes_("operasds@jpl.nasa.gov")
    hdf5_obj.attrs["institution"] = np.bytes_("NASA JPL")
    hdf5_obj.attrs["project"] = np.bytes_("OPERA")
    hdf5_obj.attrs["reference_document"] = np.bytes_(
        "Product Specification Document for the OPERA Radiometric"
        " Terrain-Corrected SAR Backscatter from Sentinel-1,"
        " JPL D-108758, Rev. Working Version 1, Aug 31, 2023")

    # product type
    product_type = cfg.groups.primary_executable.product_type
    if product_type == STATIC_LAYERS_PRODUCT_TYPE:
        hdf5_obj.attrs["title"] = np.bytes_("OPERA RTC-S1-STATIC Product")
    else:
        hdf5_obj.attrs["title"] = np.bytes_("OPERA RTC-S1 Product")

    populate_metadata_group(product_id, hdf5_obj, burst, cfg,
                            processing_datetime, is_mosaic)

    # save orbit
    orbit_group = hdf5_obj.require_group('/metadata/orbit')
    save_orbit(orbit, orbit_group, cfg.orbit_file_path)
    return hdf5_obj


def save_orbit(orbit, orbit_group, orbit_file_path):

    # ensure that the orbit reference epoch has not fractional part
    # otherwise, trancate it to seconds precision
    orbit_reference_epoch = orbit.reference_epoch
    if orbit_reference_epoch.frac != 0:
        logger.warning('the orbit reference epoch is not an'
                       ' integer number. Truncating it'
                       ' to seconds precision and'
                       ' updating the orbit ephemeris'
                       ' accordingly.')

        epoch = isce3.core.DateTime(orbit_reference_epoch.year,
                                    orbit_reference_epoch.month,
                                    orbit_reference_epoch.day,
                                    orbit_reference_epoch.hour,
                                    orbit_reference_epoch.minute,
                                    orbit_reference_epoch.second)

        orbit.update_reference_epoch(epoch)

    orbit.save_to_h5(orbit_group)

    # Add description attributes.
    orbit_group["time"].attrs["description"] = np.bytes_(
        "Time vector record. This"
        " record contains the time corresponding to position, velocity,"
        " acceleration records")
    orbit_group["position"].attrs["description"] = np.bytes_(
        "Position vector"
        " record. This record contains the platform position data with"
        " respect to WGS84 G1762 reference frame")
    orbit_group["velocity"].attrs["description"] = np.bytes_(
        "Velocity vector"
        " record. This record contains the platform velocity data with"
        " respect to WGS84 G1762 reference frame")
    orbit_group.create_dataset(
        'referenceEpoch',
        data=np.bytes_(orbit.reference_epoch.isoformat()))

    # Orbit source/type
    orbit_type = 'Undefined'
    if isinstance(orbit_file_path, str):
        orbit_file_basename = os.path.basename(orbit_file_path)
        if 'RESORB' in orbit_file_basename:
            orbit_type = 'RES restituted orbit'
        elif 'POEORB' in orbit_file_basename:
            orbit_type = 'POE precise orbit'

    elif isinstance(orbit_file_path, list):
        orbit_type_list = []
        for individual_orbit_file in orbit_file_path:
            if 'RESORB' in individual_orbit_file:
                orbit_type_individual = 'RES restituted orbit'
            elif 'POEORB' in individual_orbit_file:
                orbit_type_individual = 'POE precise orbit'
            else:
                orbit_type_individual = 'Undefined'
            orbit_type_list.append(orbit_type_individual)
        orbit_type = '; '.join(orbit_type_list)

    if 'orbitType' in orbit_group:
        del orbit_group['orbitType']
    d = orbit_group.create_dataset("orbitType",
                                   data=np.bytes_(orbit_type))

    d.attrs["description"] = np.bytes_(
        "Type of orbit file used in processing")


def get_metadata_dict(product_id: str,
                      burst_in: Sentinel1BurstSlc,
                      cfg_in: RunConfig,
                      processing_datetime: datetime,
                      is_mosaic: bool):
    '''Create RTC-S1 metadata dictionary

    Parameters
    -----------
    product_id: str
        Product ID
    burst_in: Sentinel1BurstSlc
        Source burst of the RTC
    cfg_in: RunConfig
        A class that contains the information defined in runconfig
    processing_datetime: datetime
        Processing datetime object
    is_mosaic: bool
        Flag to indicate whether the RTC-S1 product is a mosaic (True)
        or burst (False) product

    Returns
    -------
    _ : dict
        Metadata dict organized as follows:
        - Dictionary item key: HDF5 dataset key;
        - Dictionary item value: list of
            [GeoTIFF metadata key,
             metadata value,
             metadata description]
        The value `None` for the GeoTIFF metadata key indicates that
        the field is not saved on the GeoTIFF file
    '''

    # orbit files
    orbit_files = [os.path.basename(f) for f in cfg_in.orbit_path]

    # L1 SLC granules
    l1_slc_granules = [os.path.basename(f) for f in cfg_in.safe_files]

    # processing type
    processing_type = cfg_in.groups.product_group.processing_type

    # product type
    product_type = cfg_in.groups.primary_executable.product_type

    # product type. Update "RTC_S1" and "RTC_S1_STATIC" to
    # "RTC_S1" and "RTC_S1_STATIC", respectively.
    product_type_metadata_value = product_type.replace('_', '-')

    # product version
    product_version_runconfig = cfg_in.groups.product_group.product_version
    product_version = get_product_version(product_version_runconfig)

    # DEM description
    dem_file_description = cfg_in.dem_file_description

    # Slant range and azimuth resolution of the sensor
    slant_range_resolution, azimuth_resolution = \
        get_range_azimuth_resolution(burst_in)

    if not dem_file_description:
        # If the DEM description is not provided, use DEM source
        dem_file_description = os.path.basename(cfg_in.dem)

    # reformat burst ID to URL data access format used by ASF
    # (e.g., from "t018_038602_iw2" to "T018-038602-iw2")
    burst_id_asf = str(burst_in.burst_id).upper().replace('_', '-')

    # create substring "{end_date}"
    end_date_str = burst_in.sensing_stop.strftime('%Y-%m-%d')

    # source data access (URL or DOI)
    source_data_access = cfg_in.groups.input_file_group.source_data_access
    if not source_data_access:
        source_data_access = '(NOT PROVIDED)'

    # product data access (URL or DOI)
    product_data_access = cfg_in.groups.product_group.product_data_access
    if not product_data_access:
        product_data_access = '(NOT PROVIDED)'
    else:
        # replace "{burst_id}"" and "{end_date}" substrings
        product_data_access = product_data_access.format(
            burst_id=burst_id_asf, end_date=end_date_str)

    # static layers data access (URL or DOI)
    static_layers_data_access = \
        cfg_in.groups.product_group.static_layers_data_access
    if not static_layers_data_access:
        static_layers_data_access = '(NOT PROVIDED)'
    else:
        # replace "{burst_id}"" and "{end_date}" substrings
        static_layers_data_access = static_layers_data_access.format(
            burst_id=burst_id_asf, end_date=end_date_str)

    # platform ID
    if burst_in.platform_id == 'S1A':
        platform_id = 'Sentinel-1A'
    elif burst_in.platform_id == 'S1B':
        platform_id = 'Sentinel-1B'
    elif burst_in.platform_id == 'S1C':
        platform_id = 'Sentinel-1C'
    elif burst_in.platform_id == 'S1D':
        platform_id = 'Sentinel-1D'
    else:
        error_msg = f'ERROR Not recognized platform ID: {burst_in.platform_id}'
        raise NotImplementedError(error_msg)

    # geocoding algorithm
    geocoding_algorithm = cfg_in.groups.processing.geocoding.algorithm_type
    if geocoding_algorithm == 'area_projection':
        geocoding_algorithm = ('Area-Based SAR Geocoding with Adaptive'
                               ' Multilooking (GEO-AP)')

    # RTC algorithm
    rtc_algorithm = cfg_in.groups.processing.rtc.algorithm_type
    if rtc_algorithm == 'area_projection':
        rtc_algorithm = ('Area-Based SAR Radiometric Terrain Correction'
                         ' (RTC-AP)')

    # burst and mosaic snap values
    burst_snap_x = cfg_in.groups.processing.geocoding.bursts_geogrid.x_snap
    if not burst_snap_x:
        burst_snap_x = '(DISABLED)'
    else:
        burst_snap_x = float(burst_snap_x)
    burst_snap_y = cfg_in.groups.processing.geocoding.bursts_geogrid.y_snap
    if not burst_snap_y:
        burst_snap_y = '(DISABLED)'
    else:
        burst_snap_y = float(burst_snap_y)
    mosaic_snap_x = cfg_in.groups.processing.mosaicking.mosaic_geogrid.x_snap
    if not mosaic_snap_x:
        mosaic_snap_x = '(DISABLED)'
    else:
        mosaic_snap_x = float(mosaic_snap_x)
    mosaic_snap_y = cfg_in.groups.processing.mosaicking.mosaic_geogrid.y_snap
    if not mosaic_snap_y:
        mosaic_snap_y = '(DISABLED)'
    else:
        mosaic_snap_y = float(mosaic_snap_y)

    # average azimuth pixel spacing
    try:
        average_zero_doppler_spacing_in_meters = \
            burst_in.average_azimuth_pixel_spacing
    except AttributeError:
        average_zero_doppler_spacing_in_meters = '(UNSPECIFIED)'

    # Geometric accuracy
    estimated_geometric_accuracy_bias_y = \
        cfg_in.groups.processing.geocoding.estimated_geometric_accuracy_bias_y
    estimated_geometric_accuracy_bias_x = \
        cfg_in.groups.processing.geocoding.estimated_geometric_accuracy_bias_x
    estimated_geometric_accuracy_stddev_y = \
        cfg_in.groups.processing.geocoding.estimated_geometric_accuracy_stddev_y
    estimated_geometric_accuracy_stddev_x = \
        cfg_in.groups.processing.geocoding.estimated_geometric_accuracy_stddev_x

    if not estimated_geometric_accuracy_bias_y:
        estimated_geometric_accuracy_bias_y = '(UNSPECIFIED)'
    else:
        estimated_geometric_accuracy_bias_y = float(
            estimated_geometric_accuracy_bias_y)
    if not estimated_geometric_accuracy_bias_x:
        estimated_geometric_accuracy_bias_x = '(UNSPECIFIED)'
    else:
        estimated_geometric_accuracy_bias_x = float(
            estimated_geometric_accuracy_bias_x)
    if not estimated_geometric_accuracy_stddev_y:
        estimated_geometric_accuracy_stddev_y = '(UNSPECIFIED)'
    else:
        estimated_geometric_accuracy_stddev_y = float(
            estimated_geometric_accuracy_stddev_y)
    if not estimated_geometric_accuracy_stddev_x:
        estimated_geometric_accuracy_stddev_x = '(UNSPECIFIED)'
    else:
        estimated_geometric_accuracy_stddev_x = float(
            estimated_geometric_accuracy_stddev_x)

    subswath_id = burst_in.swath_name.upper()

    # `metadata_dict`` is organized as follows:
    # 'field_name': [
    #     GeoTIFF metadata field,
    #     Flag indicating whether the field is present in RTC-S1 Static Layer
    #     products (*1),
    #     Metadata field value,
    #     Metadata field description
    # ]

    # where the constants below represent the states for flag (*1)
    ALL_PRODUCTS = True
    STANDARD_RTC_S1_ONLY = False

    metadata_dict = {
        'identification/absoluteOrbitNumber':
            ['absolute_orbit_number',
             ALL_PRODUCTS,
             burst_in.abs_orbit_number,
             'Absolute orbit number'],
        'identification/trackNumber':
            ['track_number',
             ALL_PRODUCTS,
             burst_in.burst_id.track_number,
             'Track number'],
        'identification/platform':
            ['platform',
             ALL_PRODUCTS,
             platform_id,
             'Platform name'],
        # Instrument name mentioned at:
        # https://sentinel.esa.int/documents/247904/349449/s1_sp-1322_1.pdf
        'identification/instrumentName':
            ['instrument_name',
             ALL_PRODUCTS,
             f'{platform_id} CSAR',
             'Name of the instrument used to'
             ' collect the remote sensing data provided in this product'],
        'identification/productType':
            ['product_type',
             ALL_PRODUCTS,
             product_type_metadata_value,
             'Product type'],
        'identification/project':
            ['project',
             ALL_PRODUCTS,
             'OPERA',
             'Project name'],
        'identification/institution':
            ['institution',
             ALL_PRODUCTS,
             'NASA JPL',
             'Institution that created this product'],
        'identification/contactInformation':
            ['contact_information',
             ALL_PRODUCTS,
             'operasds@jpl.nasa.gov',
             'Contact information for producer of this product'],
        'identification/productVersion':
            ['product_version',
             ALL_PRODUCTS,
             product_version,
             'Product version which represents the structure of the product'
             ' and the science content governed by the algorithm, input data,'
             ' and processing parameter'],
        'identification/productSpecificationVersion':
            ['product_specification_version',
             ALL_PRODUCTS,
             str(PRODUCT_SPECIFICATION_VERSION),
             'Product specification version which represents the schema of'
             ' this product'],
        'identification/acquisitionMode':
            ['acquisition_mode',
             ALL_PRODUCTS,
             subswath_id[0:2],
             'Acquisition mode'],
        'identification/ceosAnalysisReadyDataProductType':  # 1.3
            ['ceos_analysis_ready_data_product_type',
             ALL_PRODUCTS,
             'Normalised Radar Backscatter',
             'CEOS Analysis Ready Data (CARD) product type'],
        'identification/lookDirection':
            ['look_direction',
             ALL_PRODUCTS,
             'right',
             'Look direction can be left or right'],
        'identification/orbitPassDirection':
            ['orbit_pass_direction',
             ALL_PRODUCTS,
             burst_in.orbit_direction.lower(),
             'Orbit direction can be ascending or descending'],
        'identification/isGeocoded':
            [None,
             ALL_PRODUCTS,
             True,
             'Flag to indicate whether the primary product data is in radar'
             ' geometry ("False") or map geometry ("True")'],
        'identification/productLevel':
            ['product_level',
             ALL_PRODUCTS,
             'L2',
             'Product level. L0A: Unprocessed instrument data; L0B:'
             ' Reformatted, unprocessed instrument data; L1: Processed'
             ' instrument data in radar coordinates system; and L2:'
             ' Processed instrument data in geocoded coordinates system'],
        # 'identification/productID':
        #    ['product_id',
        #     ALL_PRODUCTS,
        #     product_id,
        #     'Product identifier'],

        'identification/processingType':
            ['processing_type',
             ALL_PRODUCTS,
             processing_type,
             'Processing type: "NOMINAL", "URGENT", "CUSTOM", or "UNDEFINED"'],
        'identification/processingDateTime':
            ['processing_datetime',
             ALL_PRODUCTS,
             processing_datetime.strftime(DATE_TIME_METADATA_FORMAT),
             'Processing date and time in the format YYYY-MM-DDThh:mm:ss.sZ'],
        'identification/radarBand':  # 1.6.4
            ['radar_band',
             ALL_PRODUCTS,
             'C',
             'Acquired frequency band'],
        # 'metadata/processingCenter': # 1.7.1
        #     ['source_data_processing_center',
        #      (f'organization: "Jet Propulsion Laboratory", '
        #       f'site: "Pasadena, CA", '
        #       f'country: "United States of America"'),
        #      'Data processing center'],
        'identification/ceosAnalysisReadyDataDocumentIdentifier':
            ['ceos_analysis_ready_data_document_identifier',
             ALL_PRODUCTS,
             'https://ceos.org/ard/files/PFS/NRB/v5.5/CARD4L-PFS_NRB_v5.5.pdf',
             'CEOS Analysis Ready Data (CARD) document identifier'],
        'identification/dataAccess':
            ['product_data_access',
             ALL_PRODUCTS,
             product_data_access,
             'Location from where this product can be retrieved'
             ' (URL or DOI)'],
        'identification/staticLayersDataAccess':
            ['static_layers_data_access',
             STANDARD_RTC_S1_ONLY,
             static_layers_data_access,
             'Location of the static layers product associated with this'
             ' product (URL or DOI)'],
        'metadata/sourceData/numberOfAcquisitions':  # 1.6.4
            ['source_data_number_of_acquisitions',
             ALL_PRODUCTS,
             1,
             'Number of source data acquisitions'],
        'metadata/sourceData/dataAccess':
            ['source_data_access',
             ALL_PRODUCTS,
             source_data_access,
             'Location from where the source data can be retrieved'
             ' (URL or DOI)'],

        'metadata/sourceData/institution':
            ['source_data_institution',
             ALL_PRODUCTS,
             'ESA',
             'Institution that created the source data product'],
        'metadata/sourceData/processingCenter':  # 1.6.6
            ['source_data_processing_center',
             ALL_PRODUCTS,
             (f'organization: \"{burst_in.burst_misc_metadata.processing_info_dict["organisation"]}\", '
              f'site: \"{burst_in.burst_misc_metadata.processing_info_dict["site"]}\", '
              f'country: \"{burst_in.burst_misc_metadata.processing_info_dict["country"]}\"'),
             'Source data processing center'],

        'metadata/sourceData/rangeBandwidth':
            ['source_data_range_bandwidth',
             STANDARD_RTC_S1_ONLY,
             burst_in.range_bandwidth,
             'Processed range bandwidth in Hz'],

        'metadata/sourceData/processingDateTime':  # 1.6.6
            ['source_data_processing_datetime',
             ALL_PRODUCTS,
             burst_in.burst_misc_metadata.processing_info_dict['stop'],
             'Processing UTC date and time of the source data product (SLC'
             ' post processing date time) in the format'
             ' YYYY-MM-DDThh:mm:ss.sZ'],

        'metadata/sourceData/softwareVersion':  # 1.6.6
            ['source_data_software_version',
             ALL_PRODUCTS,
             str(burst_in.ipf_version),
             'Version of the software used to create the source data'
             ' (IPF version)'],
        'metadata/sourceData/azimuthLooks':  # 1.6.6
            [None,
             STANDARD_RTC_S1_ONLY,
             burst_in.burst_misc_metadata.azimuth_looks,
             'Number of looks in azimuth used to generate source data'],
        'metadata/sourceData/slantRangeLooks':  # 1.6.6
            [None,
             STANDARD_RTC_S1_ONLY,
             burst_in.burst_misc_metadata.slant_range_looks,
             'Number of looks in slant range used to generate source data'],

        'metadata/sourceData/productLevel':  # 1.6.6
            ['source_data_product_level',
             ALL_PRODUCTS,
             'L1',
             'Product level of the source data. L0A: Unprocessed instrument'
             ' data; L0B: Reformatted, unprocessed instrument data; L1:'
             ' Processed instrument data in radar coordinates system; and L2:'
             ' Processed instrument data in geocoded coordinates system'],
        'metadata/sourceData/centerFrequency':
            ['center_frequency',
             STANDARD_RTC_S1_ONLY,
             burst_in.radar_center_frequency,
             'Center frequency of the processed image in Hz'],
        'metadata/sourceData/zeroDopplerTimeSpacing':  # 1.6.7
            ['source_data_zero_doppler_time_spacing',
             ALL_PRODUCTS,
             burst_in.azimuth_time_interval,
             'Time interval in the along-track direction of the source data'],
        'metadata/sourceData/averageZeroDopplerSpacingInMeters':  # 1.6.7
            ['source_data_average_zero_doppler_spacing_in_meters',
             ALL_PRODUCTS,
             average_zero_doppler_spacing_in_meters,
             'Average pixel spacing in meters between consecutive lines'
             ' in the along-track direction of the source data'],
        'metadata/sourceData/slantRangeSpacing':  # 1.6.7
            ['source_data_slant_range_spacing',
             ALL_PRODUCTS,
             burst_in.range_pixel_spacing,
             'Distance in meters between consecutive range samples'
             ' of the source data'],
        'metadata/sourceData/azimuthResolutionInMeters':  # 1.6.7
            ['source_data_azimuth_resolution_in_meters',
             STANDARD_RTC_S1_ONLY,
             azimuth_resolution,
             'Azimuth resolution of the source data in meters'],
        'metadata/sourceData/slantRangeResolutionInMeters':  # 1.6.7
            ['source_data_slant_range_resolution_in_meters',
             STANDARD_RTC_S1_ONLY,
             slant_range_resolution,
             'Slant-range resolution of the source data in meters'],

        'metadata/sourceData/nearRangeIncidenceAngle':  # 1.6.7
            [None,
             STANDARD_RTC_S1_ONLY,
             burst_in.burst_misc_metadata.inc_angle_near_range,
             'Near range incidence angle in degrees'],
        'metadata/sourceData/farRangeIncidenceAngle':  # 1.6.7
            [None,
             STANDARD_RTC_S1_ONLY,
             burst_in.burst_misc_metadata.inc_angle_far_range,
             'Far range incidence angle in degrees'],
        # Source for the max. NESZ:
        # (https://sentinels.copernicus.eu/web/sentinel/user-guides/
        #  sentinel-1-sar/acquisition-modes/interferometric-wide-swath)
        'metadata/sourceData/maxNoiseEquivalentSigmaZero':  # 1.6.9
            [None,
             STANDARD_RTC_S1_ONLY,
             -22.0,
             'Maximum Noise equivalent sigma0 in dB'],

        'metadata/processingInformation/algorithms/softwareVersion':
            ['software_version',
             ALL_PRODUCTS,
             str(SOFTWARE_VERSION),
             'Software version'],

        # 1.7.4
        ('metadata/processingInformation/parameters/'
            'preprocessingMultilookingApplied'):
            ['processing_information_multilooking_applied',
             ALL_PRODUCTS,
             False,
             'Flag to indicate if a preprocessing multilooking has been'
             ' applied'],

        # 1.7.4
        ('metadata/processingInformation/parameters/'
            'filteringApplied'):
            ['processing_information_filtering_applied',
             ALL_PRODUCTS,
             False,
             'Flag to indicate if post-processing filtering has been applied'],

        # 3.3
        'metadata/processingInformation/parameters/noiseCorrectionApplied':
            ['processing_information_noise_correction_applied',
             STANDARD_RTC_S1_ONLY,
             cfg_in.groups.processing.apply_thermal_noise_correction,
             'Flag to indicate if noise removal has been applied'],
        ('metadata/processingInformation/parameters/'
            'radiometricTerrainCorrectionApplied'):
            ['processing_information_radiometric_terrain_correction_applied',
             STANDARD_RTC_S1_ONLY,
             cfg_in.groups.processing.apply_rtc,
             'Flag to indicate if radiometric terrain correction (RTC) has'
             ' been applied'],
        ('metadata/processingInformation/parameters/'
            'staticTroposphericGeolocationCorrectionApplied'):
            ['processing_information'
             '_static_tropospheric_geolocation_correction_applied',
             ALL_PRODUCTS,
             cfg_in.groups.processing.apply_static_tropospheric_delay_correction,
             'Flag to indicate if the static tropospheric correction has been'
             ' applied'],
        ('metadata/processingInformation/parameters/'
            'wetTroposphericGeolocationCorrectionApplied'):
            ['processing_information'
             '_wet_tropospheric_geolocation_correction_applied',
             ALL_PRODUCTS,
             False,
             'Flag to indicate if the wet tropospheric correction has been'
             ' applied'],
        ('metadata/processingInformation/parameters/'
            'bistaticDelayCorrectionApplied'):
            ['processing_information'
             '_bistatic_delay_correction_applied',
             ALL_PRODUCTS,
             cfg_in.groups.processing.apply_bistatic_delay_correction,
             'Flag to indicate if the bistatic delay correction has been'
             ' applied'],
        ('metadata/processingInformation/parameters/'
            'inputBackscatterNormalizationConvention'):
            ['processing_information'
             '_input_backscatter_normalization_convention',
             ALL_PRODUCTS,
             cfg_in.groups.processing.rtc.input_terrain_radiometry,
             'Backscatter normalization convention of the source data'],
        ('metadata/processingInformation/parameters/'
            'outputBackscatterNormalizationConvention'):
            ['processing_information'
             '_output_backscatter_normalization_convention',
             ALL_PRODUCTS,
             cfg_in.groups.processing.rtc.output_type,
             'Backscatter normalization convention of the radar imagery'
             ' associated with this product'],

        # 3.1
        ('metadata/processingInformation/parameters/'
            'outputBackscatterExpressionConvention'):
            ['processing_information'
             '_output_backscatter_expression_convention',
             ALL_PRODUCTS,
             'linear backscatter intensity',
             'Backscatter expression convention'],

        # 3.2
        ('metadata/processingInformation/parameters/'
            'outputBackscatterDecibelConversionEquation'):
            ['processing_information'
             '_output_backscatter_decibel_conversion_equation',
             ALL_PRODUCTS,
             '10*log10(backscatter_linear)',
             'Equation to convert provided backscatter to decibel (dB)'],

        # 4.4
        ('metadata/processingInformation/parameters/geocoding/'
            'burstGeogridSnapX'):
            ['processing_information_burst_geogrid_snap_x',
             ALL_PRODUCTS,
             burst_snap_x,
             'Burst geogrid snap for Coordinate X (W/E)'],
        ('metadata/processingInformation/parameters/geocoding/'
            'burstGeogridSnapY'):
            ['processing_information_burst_geogrid_snap_y',
             ALL_PRODUCTS,
             burst_snap_y,
             'Burst geogrid snap for Coordinate Y (S/N)'],
        # 'metadata/processingInformation/geoidReference':  # for 4.2

        # 4.3
        'metadata/qa/geometricAccuracy/bias/y':
            ['qa_geometric_accuracy_bias_y',
             STANDARD_RTC_S1_ONLY,
             estimated_geometric_accuracy_bias_y,
             ('An estimate of the localization error bias in the northing'
              ' direction')],

        'metadata/qa/geometricAccuracy/stddev/y':
            ['qa_geometric_accuracy_stddev_y',
             STANDARD_RTC_S1_ONLY,
             estimated_geometric_accuracy_stddev_y,
             ('An estimate of the localization error standard deviation'
              ' in the northing direction')],

        'metadata/qa/geometricAccuracy/bias/x':
            ['qa_geometric_accuracy_bias_x',
             STANDARD_RTC_S1_ONLY,
             estimated_geometric_accuracy_bias_x,
             ('An estimate of the localization error bias in the easting'
              ' direction')],

        'metadata/qa/geometricAccuracy/stddev/x':
            ['qa_geometric_accuracy_stddev_x',
             STANDARD_RTC_S1_ONLY,
             estimated_geometric_accuracy_stddev_x,
             ('An estimate of the localization error standard deviation'
              ' in the easting direction')],

        'metadata/processingInformation/algorithms/demInterpolation':
            ['processing_information'
             '_dem_interpolation_algorithm',
             ALL_PRODUCTS,
             cfg_in.groups.processing.dem_interpolation_method,
             'DEM interpolation method'],
        'metadata/processingInformation/algorithms/demEgmModel':
            ['processing_information'
             '_dem_egm_model',
             ALL_PRODUCTS,
             'Earth Gravitational Model 2008 (EGM2008)',
             'Earth Gravitational Model associated with the DEM'],
        'metadata/processingInformation/algorithms/geocoding':
            ['processing_information'
             '_geocoding_algorithm',
             ALL_PRODUCTS,
             geocoding_algorithm,
             'Geocoding algorithm'],
        'metadata/processingInformation/algorithms/' +
        'radiometricTerrainCorrection':
            ['processing_information'
             '_radiometric_terrain_correction_algorithm',
             ALL_PRODUCTS,
             rtc_algorithm,
             'Radiometric terrain correction (RTC) algorithm'],
        'metadata/processingInformation/algorithms/isce3Version':
            ['isce3_version',
             ALL_PRODUCTS,
             isce3.__version__,
             'Version of the ISCE3 framework used for processing'],
        'metadata/processingInformation/algorithms/s1ReaderVersion':
            ['s1_reader_version',
             ALL_PRODUCTS,
             release_version,
             'Version of the OPERA s1-reader used for processing'],

        'metadata/processingInformation/inputs/l1SlcGranules':
            ['input_l1_slc_granules',
             ALL_PRODUCTS,
             l1_slc_granules,
             'List of input L1 SLC products used'],
        'metadata/processingInformation/inputs/orbitFiles':
            ['input_orbit_files',
             ALL_PRODUCTS,
             orbit_files,
             'List of input orbit files used'],
        'metadata/processingInformation/inputs/annotationFiles':
            ['input_annotation_files',
             ALL_PRODUCTS,
             [burst_in.burst_calibration.basename_cads,
              burst_in.burst_noise.basename_nads],
             'List of input annotation files used'],
        'metadata/processingInformation/inputs/demSource':
            ['input_dem_source',
             ALL_PRODUCTS,
             dem_file_description,
             'Description of the input digital elevation model (DEM)']
    }

    # Add reference to the thermal noise correction algorithm when the
    # correction is applied
    if cfg_in.groups.processing.apply_thermal_noise_correction:  # 3.3
        noise_removal_algorithm_reference = (
            'Thermal Denoising of Products Generated by the S-1 IPF.'
            ' ESA document. Accessed: May 29, 2023. [Online]. Available:'
            ' https://sentinels.copernicus.eu/documents/247904/2142675/'
            'Thermal-Denoising-of-Products-Generated-by-Sentinel-1-IPF.pdf')
    else:
        noise_removal_algorithm_reference = '(noise removal not applied)'
    metadata_dict['metadata/processingInformation/algorithms/'
                  'noiseCorrectionAlgorithmReference'] =\
        ['processing_information'
         '_noise_removal_algorithm_reference',
         STANDARD_RTC_S1_ONLY,
         noise_removal_algorithm_reference,
         'A reference to the noise removal algorithm applied']

    # Add RTC algorithm reference depending on the algorithm applied
    url_rtc_algorithm_document = '(RTC not applied)'
    if cfg_in.groups.processing.apply_rtc:  # 3.4
        if cfg_in.groups.processing.rtc.algorithm_type == 'area_projection':
            url_rtc_algorithm_document = \
                ('Gustavo H. X. Shiroma, Marco Lavalle, and Sean M. Buckley,'
                 ' "An Area-Based Projection Algorithm for SAR Radiometric'
                 ' Terrain Correction and Geocoding," in IEEE Transactions'
                 ' on Geoscience and Remote Sensing, vol. 60, pp. 1-23, 2022,'
                 ' Art no. 5222723, doi: 10.1109/TGRS.2022.3147472.')
        elif (cfg_in.groups.processing.rtc.algorithm_type ==
                'bilinear_distribution'):
            url_rtc_algorithm_document = \
                ('David Small, "Flattening Gamma: Radiometric Terrain'
                 ' Correction for SAR Imagery," in IEEE Transactions on'
                 ' Geoscience and Remote Sensing, vol. 49, no. 8, pp.'
                 ' 3081-3093, Aug. 2011, doi: 10.1109/TGRS.2011.2120616.')
        else:
            raise NotImplementedError
    metadata_dict['metadata/processingInformation/algorithms/'
                  'radiometricTerrainCorrectionAlgorithmReference'] =\
        ['processing_information'
         '_radiometric_terrain_correction_algorithm_reference',
         ALL_PRODUCTS,
         url_rtc_algorithm_document,
         'A reference to the radiometric terrain correction (RTC) algorithm'
         ' applied']

    # Add geocoding algorithm reference depending on the algorithm applied
    # TODO: add references to the other geocoding algorithms
    if cfg_in.groups.processing.geocoding.algorithm_type == 'area_projection':
        url_geocoding_algorithm_document = \
                ('Gustavo H. X. Shiroma, Marco Lavalle, and Sean M. Buckley,'
                 ' "An Area-Based Projection Algorithm for SAR Radiometric'
                 ' Terrain Correction and Geocoding," in IEEE Transactions'
                 ' on Geoscience and Remote Sensing, vol. 60, pp. 1-23, 2022,'
                 ' Art no. 5222723, doi: 10.1109/TGRS.2022.3147472.')
        metadata_dict['metadata/processingInformation/algorithms/'
                      'geocodingAlgorithmReference'] =\
            ['processing_information'
             '_geocoding_algorithm_reference',
             ALL_PRODUCTS,
             url_geocoding_algorithm_document,
             'A reference to the geocoding algorithm applied']

    if is_mosaic:
        # Metadata only for the mosaic product
        metadata_dict['metadata/processingInformation/parameters/geocoding/'
                      'mosaicGeogridSnapX'] = \
            ['processing_information_mosaic_geogrid_snap_x',
             ALL_PRODUCTS,
             mosaic_snap_x,
             'mosaic geogrid snap for Coordinate X (W/E)']
        metadata_dict['metadata/processingInformation/parameters/geocoding/'
                      'mosaicGeogridSnapY'] = \
            ['processing_information_mosaic_geogrid_snap_y',
             ALL_PRODUCTS,
             mosaic_snap_y,
             'mosaic geogrid snap for Coordinate Y (S/N)']

    else:
        # Metadata only for the burst product
        # Calculate bounding box
        xmin_geogrid = cfg_in.geogrids[str(burst_in.burst_id)].start_x
        ymax_geogrid = cfg_in.geogrids[str(burst_in.burst_id)].start_y
        spacing_x = cfg_in.geogrids[str(burst_in.burst_id)].spacing_x
        spacing_y = cfg_in.geogrids[str(burst_in.burst_id)].spacing_y
        width_geogrid = cfg_in.geogrids[str(burst_in.burst_id)].width
        length_geogrid = cfg_in.geogrids[str(burst_in.burst_id)].length
        epsg_geogrid = cfg_in.geogrids[str(burst_in.burst_id)].epsg

        metadata_dict['identification/boundingPolygon'] = \
            ['bounding_polygon',
             STANDARD_RTC_S1_ONLY,
             get_polygon_wkt(burst_in),
             'OGR compatible WKT representation of the product'
             ' bounding polygon']

        # Attribute `epsg` for HDF5 dataset /identification/boundingPolygon
        metadata_dict['identification/boundingPolygon[epsg]'] = \
            ['bounding_polygon_epsg_code',
             STANDARD_RTC_S1_ONLY,
             '4326',
             'Bounding polygon EPSG code']

        xy_bounding_box = [
           xmin_geogrid,
           ymax_geogrid + length_geogrid * spacing_y,
           xmin_geogrid + width_geogrid * spacing_x,
           ymax_geogrid
        ]

        # 1.7.5
        metadata_dict['identification/boundingBox'] = \
            ['bounding_box',
             ALL_PRODUCTS,
             xy_bounding_box,  
             'Bounding box of the product, in order of xmin, ymin, xmax, ymax']

        # Attribute `epsg` for HDF5 dataset /identification/boundingBox
        metadata_dict['identification/boundingBox[epsg]'] = \
            ['bounding_box_epsg_code',
             ALL_PRODUCTS,
             str(epsg_geogrid),
             'Bounding box EPSG code']

        # 1.7.8
        metadata_dict['identification/boundingBox'
                      '[pixel_coordinate_convention]'] = \
            ['bounding_box_pixel_coordinate_convention',
             ALL_PRODUCTS,
             "Edges/corners. Xmin, Ymin, Xmax, Ymax.",
             'Bounding box pixel coordinate convention']

        metadata_dict['identification/burstID'] = \
            ['burst_id',
             ALL_PRODUCTS,
             str(burst_in.burst_id),
             'Burst identification (burst ID)']

        metadata_dict['identification/subSwathID'] = \
            ['sub_swath_id',
             ALL_PRODUCTS,
             subswath_id,
             'Sub-swath identification']

        # TODO: Update for static layers!!!
        metadata_dict['identification/zeroDopplerStartTime'] = \
            ['zero_doppler_start_time',
             ALL_PRODUCTS,
             burst_in.sensing_start.strftime(DATE_TIME_METADATA_FORMAT),
             'Azimuth start time of the product in the format'
             ' YYYY-MM-DDThh:mm:ss.sZ'] # 1.6.3

        # TODO: Update for static layers!!!
        metadata_dict['identification/zeroDopplerEndTime'] = \
            ['zero_doppler_end_time',
             ALL_PRODUCTS,
             burst_in.sensing_stop.strftime(DATE_TIME_METADATA_FORMAT),
             'Azimuth stop time of the product in the format'
             ' YYYY-MM-DDThh:mm:ss.sZ']  # 1.6.3

        metadata_dict['metadata/sourceData/zeroDopplerStartTime'] = \
            ['source_data_zero_doppler_start_time',
             ALL_PRODUCTS,
             burst_in.sensing_start.strftime(DATE_TIME_METADATA_FORMAT),
             'Azimuth start time of the input product in the format'
             ' YYYY-MM-DDThh:mm:ss.sZ']  # 1.6.3
        metadata_dict['metadata/sourceData/zeroDopplerEndTime'] = \
            ['source_data_zero_doppler_end_time',
             ALL_PRODUCTS,
             burst_in.sensing_stop.strftime(DATE_TIME_METADATA_FORMAT),
             'Azimuth stop time of the input product in the format'
             ' YYYY-MM-DDThh:mm:ss.sZ']  # 1.6.3
        metadata_dict['metadata/sourceData/numberOfAzimuthLines'] = \
            ['source_data_number_of_azimuth_lines',
             STANDARD_RTC_S1_ONLY,
             burst_in.length,
             'Number of azimuth lines within the source data product']

        metadata_dict['metadata/sourceData/numberOfRangeSamples'] = \
            ['source_data_number_of_range_samples',
             STANDARD_RTC_S1_ONLY,
             burst_in.width,
             'Number of slant range samples for each azimuth line within the'
             ' source data']
        metadata_dict['metadata/sourceData/slantRangeStart'] = \
            ['source_data_slant_range_start',
             STANDARD_RTC_S1_ONLY,
             burst_in.starting_range,
             'Slant-range start distance of the source data']

    this_product_metadata_dict = {}
    for h5_path, (geotiff_field, flag_all_products, data, description) in \
            metadata_dict.items():
        if (product_type == STATIC_LAYERS_PRODUCT_TYPE and
                not flag_all_products):
            continue
        this_product_metadata_dict[h5_path] = [geotiff_field, data,
                                               description]

    if not is_mosaic and product_type != STATIC_LAYERS_PRODUCT_TYPE:

        # Add RFI metadata into `metadata_dict`
        rfi_metadata_dict = get_rfi_metadata_dict(burst_in, 'metadata/qa/rfi')
        this_product_metadata_dict.update(rfi_metadata_dict)

    return this_product_metadata_dict


def all_metadata_dict_to_geotiff_metadata_dict(metadata_dict):
    '''
    Convert all metadata dict to GeoTIFF metadata dict
    Parameters
    ----------
    metadata_dict : dict
        Metadata dict organized as follows:
        - Dictionary item key: HDF5 dataset key;
        - Dictionary item value: list of
            [GeoTIFF metadata key,
             metadata value,
             metadata description]
        The value `None` for the GeoTIFF metadata key indicates that
        the field is not saved on the GeoTIFF file

    Returns
    -------
    geotiff_metadata_dict : dict
        Metadata dict to be saved onto the RTC-S1 product GeoTIFF file
    '''
    geotiff_metadata_dict = {}
    for _, (key, value, _) in metadata_dict.items():
        if key is None:
            continue
        if isinstance(value, str):
            geotiff_metadata_dict[key.upper()] = str(value)
            continue
        geotiff_metadata_dict[key.upper()] = value

    return geotiff_metadata_dict


def populate_metadata_group(product_id: str,
                            h5py_obj: h5py.File,
                            burst_in: Sentinel1BurstSlc,
                            cfg_in: RunConfig,
                            processing_datetime: datetime,
                            is_mosaic: bool):
    '''Populate RTC metadata based on Sentinel1BurstSlc and RunConfig

    Parameters
    -----------
    product_id: str
        Product ID
    h5py_obj: h5py.File
        HDF5 object into which write the metadata
    burst_in: Sentinel1BurstSlc
        Source burst of the RTC
    cfg_in: RunConfig
        A class that contains the information defined in runconfig
    processing_datetime: datetime
        Processing datetime object
    is_mosaic: bool
        Flag to indicate if the RTC-S1 product is a mosaic (True)
        or burst (False) product
    '''

    metadata_dict = get_metadata_dict(product_id, burst_in, cfg_in,
                                      processing_datetime, is_mosaic)

    for path_dataset_in_h5, (_, data, description) in metadata_dict.items():

        # check if metadata element is an HDF5 dataset attribute
        # by exctracting substrings within square brackets
        attribute_list = re.findall(r'\[.*?\]', path_dataset_in_h5)
        if len(attribute_list) > 2:
            # if there are two or more substrings raise an error
            error_message = 'ERROR invalid HDF5 path: ' + path_dataset_in_h5
            raise NotImplementedError(error_message)
        elif len(attribute_list) == 1:
            # if there's only one substring, that's an attribute
            attribute_name = attribute_list[0][1:-1]
            path_dataset_in_h5 = path_dataset_in_h5.replace(
                attribute_list[0], '')
            dset = h5py_obj[path_dataset_in_h5]
            dset.attrs[attribute_name] = data
            continue
        if isinstance(data, str):
            dset = h5py_obj.create_dataset(
                path_dataset_in_h5, data=np.bytes_(data))
        else:
            dset = h5py_obj.create_dataset(path_dataset_in_h5, data=data)

        dset.attrs['description'] = np.bytes_(description)


def save_hdf5_dataset(ds_filename, h5py_obj, root_path,
                      yds, xds, layer_name, standard_name=None,
                      long_name=None, units=None, fill_value=None,
                      valid_min=None, valid_max=None, compute_stats=True):
    '''
    write temporary raster file contents to HDF5

    Parameters
    ----------
    ds_filename: str
        source raster file
    h5py_obj: h5py object
        h5py object of destination HDF5
    root_path: str
        path of output raster data
    yds: h5py dataset object
        y-axis dataset
    xds: h5py dataset object
        x-axis dataset
    layer_name: str
        name of dataset to be added to root_path
    standard_name: str, optional
        Standard name
    long_name: str, optional
        Long name
    units: str, optional
        Units
    fill_value: float, optional
        Fill value
    valid_min: float, optional
        Minimum value
    valid_max: float, optional
        Maximum value
    '''
    if not os.path.isfile(ds_filename):
        logger.warning(f'WARNING Cannot open raster file: {ds_filename}')
        return

    if isinstance(layer_name, str):
        ds_name = layer_hdf5_dict[layer_name]
    else:
        ds_name = [layer_hdf5_dict[l] for l in layer_name]

    if long_name is not None:
        description = long_name
    else:
        description = layer_description_dict[layer_name]

    stats_real_imag_vector = None
    stats_vector = None
    if compute_stats:
        raster = isce3.io.Raster(ds_filename)

        if (raster.datatype() == gdal.GDT_CFloat32 or
                raster.datatype() == gdal.GDT_CFloat64):
            stats_real_imag_vector = \
                isce3.math.compute_raster_stats_real_imag(raster)
        elif raster.datatype() == gdal.GDT_Float64:
            stats_vector = isce3.math.compute_raster_stats_float64(raster)
        else:
            stats_vector = isce3.math.compute_raster_stats_float32(raster)

    gdal_ds = gdal.Open(ds_filename, gdal.GA_ReadOnly)
    nbands = gdal_ds.RasterCount
    for band in range(nbands):
        data = gdal_ds.GetRasterBand(band+1).ReadAsArray()

        if isinstance(ds_name, str):
            h5_ds = os.path.join(root_path, ds_name)
        else:
            h5_ds = os.path.join(root_path, ds_name[band])

        if h5_ds in h5py_obj:
            del h5py_obj[h5_ds]

        dset = h5py_obj.create_dataset(h5_ds, data=data)
        dset.dims[0].attach_scale(yds)
        dset.dims[1].attach_scale(xds)
        dset.attrs['grid_mapping'] = np.bytes_("projection")

        if standard_name is not None:
            dset.attrs['standard_name'] = np.bytes_(standard_name)

        if long_name is not None:
            dset.attrs['long_name'] = np.bytes_(long_name)

        dset.attrs['description'] = np.bytes_(description)

        if units is not None:
            dset.attrs['units'] = np.bytes_(units)

        if fill_value is not None:
            dset.attrs.create('_FillValue', data=fill_value)
        elif 'cfloat' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan + 1j * np.nan)
        elif 'float' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan)
        elif 'byte' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=255)

        if stats_vector is not None:
            stats_obj = stats_vector[band]
            dset.attrs.create('min_value', data=stats_obj.min)
            dset.attrs.create('mean_value', data=stats_obj.mean)
            dset.attrs.create('max_value', data=stats_obj.max)
            dset.attrs.create('sample_standard_deviation',
                              data=stats_obj.sample_stddev)

        elif stats_real_imag_vector is not None:
            stats_obj = stats_real_imag_vector[band]
            dset.attrs.create('min_real_value', data=stats_obj.min_real)
            dset.attrs.create('mean_real_value', data=stats_obj.mean_real)
            dset.attrs.create('max_real_value', data=stats_obj.max_real)
            dset.attrs.create('sample_standard_deviation_real',
                              data=stats_obj.sample_stddev_real)

            dset.attrs.create('min_imag_value', data=stats_obj.min_imag)
            dset.attrs.create('mean_imag_value', data=stats_obj.mean_imag)
            dset.attrs.create('max_imag_value', data=stats_obj.max_imag)
            dset.attrs.create('sample_standard_deviation_imag',
                              data=stats_obj.sample_stddev_imag)

        if valid_min is not None:
            dset.attrs.create('valid_min', data=valid_min)

        if valid_max is not None:
            dset.attrs.create('valid_max', data=valid_max)

    del gdal_ds


def get_rfi_metadata_dict(burst_in, rfi_root_path):
    '''
    Populate the RFI information into HDF5 object

    Parameters
    ----------
    burst_in: Sentinel1BurstSlc
        Sentinel-1 Burst SLC object with RFI information
    rfi_root_path: str
        Root path to the RFI information in the metadata HDF5 file

    '''
    rfi_metadata_dict = {}

    is_rfi_info_empty = burst_in.burst_rfi_info is None
    rfi_metadata_dict[f'{rfi_root_path}/isRfiInfoAvailable'] =\
        ['qa_rfi_info_available',
         not is_rfi_info_empty,
         'A flag to indicate whether RFI information is available in the'
         ' source data']

    if is_rfi_info_empty:
        return rfi_metadata_dict

    # Create group for RFI info
    subpath_data_dict = {
        'rfiMitigationPerformed':
            ['qa_rfi_mitigation_performed',
             burst_in.burst_rfi_info.rfi_mitigation_performed,
             'RFI detection and mitigation strategy'],
        'rfiMitigationDomain':
            ['qa_rfi_mitigation_domain',
             burst_in.burst_rfi_info.rfi_mitigation_domain,
             'Domain in which the RFI mitigation was performed'],
        'rfiBurstReport/swath':
            ['qa_rfi_burst_report_swath',
             burst_in.burst_rfi_info.rfi_burst_report['swath'],
             'Swath associated with the IW RFI burst report list'],
        'rfiBurstReport/azimuthTime':
            ['qa_rfi_burst_report_azimuth_time',
             burst_in.burst_rfi_info.rfi_burst_report['azimuthTime'].strftime(
                DATE_TIME_METADATA_FORMAT),
             'Sensing start time of the burst that corresponds to the RFI'
             ' report in the format YYYY-MM-DDThh:mm:ss.sZ'],
        'rfiBurstReport/inBandOutBandPowerRatio':
            ['qa_rfi_in_band_out_band_power_ratio',
             burst_in.burst_rfi_info.rfi_burst_report[
                 'inBandOutBandPowerRatio'],
             'Ratio between the in-band and out-of-band power of the burst.']
    }

    # Aliases for the improvement of code readability
    rfi_burst_report_time =\
        (burst_in.burst_rfi_info.rfi_burst_report['timeDomainRfiReport']
         if 'timeDomainRfiReport' in
         burst_in.burst_rfi_info.rfi_burst_report.keys()
         else None)
    rfi_burst_report_freq =\
        (burst_in.burst_rfi_info.rfi_burst_report[
            'frequencyDomainRfiBurstReport']
         if 'frequencyDomainRfiBurstReport' in
         burst_in.burst_rfi_info.rfi_burst_report.keys()
         else None)

    # Add RFI burst domain report
    if ('timeDomainRfiReport' in
            burst_in.burst_rfi_info.rfi_burst_report.keys()):
        # populate the time domain RFI report
        subpath_data_dict['timeDomainRfiReport/percentageAffectedLines'] = \
            ['qa_rfi_time_domain_report_percentage_affected_lines',
             rfi_burst_report_time['percentageAffectedLines'],
             'Percentage of level-0 lines affected by RFI']

        subpath_data_dict['timeDomainRfiReport/'
                          'avgPercentageAffectedSamples'] = \
            ['qa_rfi_time_domain_report_avg_percentage_affected_samples',
             rfi_burst_report_time['avgPercentageAffectedSamples'],
             ('Average percentage of affected level-0 samples '
              'in the lines containing RFI.')]

        subpath_data_dict['timeDomainRfiReport/'
                          'maxPercentageAffectedSamples'] = \
            ['qa_rfi_time_domain_report_max_percentage_affected_samples',
             rfi_burst_report_time['maxPercentageAffectedSamples'],
             'Maximum percentage of level-0 samples affected by RFI in the'
             ' same line']

    if ('frequencyDomainRfiBurstReport' in
            burst_in.burst_rfi_info.rfi_burst_report.keys()):
        # populate the frequency domain RFI report
        subpath_data_dict['frequencyDomainRfiBurstReport/numSubBlocks'] = \
            ['qa_rfi_frequency_domain_report_num_sub_blocks',
             rfi_burst_report_freq['numSubBlocks'],
             'Number of sub-blocks in the current burst']

        subpath_data_dict['frequencyDomainRfiBurstReport/subBlockSize'] = \
            ['qa_rfi_frequency_domain_report_sub_block_size',
             rfi_burst_report_freq['subBlockSize'],
             'Number of lines in each sub-block']

        subpath_data_dict[('frequencyDomainRfiBurstReport/isolatedRfiReport/'
                           'percentageAffectedLines')] = \
            ['qa_rfi_frequency_domain_report_isolated_'
             'percentage_affected_lines',
             rfi_burst_report_freq['isolatedRfiReport'][
                 'percentageAffectedLines'],
             'Percentage of level-0 lines affected by RFI.']

        subpath_data_dict[('frequencyDomainRfiBurstReport/isolatedRfiReport/'
                           'maxPercentageAffectedBW')] = \
            ['qa_rfi_frequency_domain_report_isolated_'
             'max_bandwidth_percentage_affected_lines',
             rfi_burst_report_freq['isolatedRfiReport'][
                 'maxPercentageAffectedBW'],
             'Maximum percentage of bandwidth affected by isolated RFI in a'
             ' single line.']

        subpath_data_dict['frequencyDomainRfiBurstReport/'
                          'percentageBlocksPersistentRfi'] = \
            ['qa_rfi_frequency_domain_report_percentage_blocks'
             '_persistent_rfi',
             rfi_burst_report_freq['percentageBlocksPersistentRfi'],
             ('Percentage of processing blocks affected by persistent RFI. '
              'In this case the RFI detection is performed on the '
              'mean power spectrum density (PSD) of each processing block.')]

        subpath_data_dict[('frequencyDomainRfiBurstReport/'
                           'maxPercentageBWAffectedPersistentRfi')] = \
            ['qa_rfi_frequency_domain_report_max_percentage_bw_affected'
             '_persistent_rfi',
             rfi_burst_report_freq['maxPercentageBWAffectedPersistentRfi'],
             ('Maximum percentage of the bandwidth affected by '
              'persistent RFI in a single processing block')]

    for fieldname, data in subpath_data_dict.items():
        path_in_rfi_dict = os.path.join(rfi_root_path, fieldname)
        rfi_metadata_dict[path_in_rfi_dict] = data

    return rfi_metadata_dict


def get_range_azimuth_resolution(burst: Sentinel1BurstSlc):
    '''
    Get the range and azimuth resolution based on the ESA documentation

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        Burst object to compute the resolution

    Returns
    -------
    slant_range_resolution: float
        Slant-range resolution in meters
    azimuth_resolution: float
        Azimuth resolution in meters

    Notes
    -----
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/
    products-algorithms/level-1/single-look-complex/interferometric-wide-swath
    '''

    resolution_subswath_range_azimuth_dict = {
        'IW1': [2.7, 22.5],
        'IW2': [3.1, 22.7],
        'IW3': [3.5, 22.6]
    }

    slant_range_resolution, azimuth_resolution =\
        resolution_subswath_range_azimuth_dict[burst.burst_id.subswath]

    return slant_range_resolution, azimuth_resolution
