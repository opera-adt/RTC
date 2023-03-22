#!/usr/bin/env python

import os
import numpy as np
import h5py
import logging
from datetime import datetime
from osgeo import gdal

import isce3
import shapely

from s1reader.s1_burst_slc import Sentinel1BurstSlc
from s1reader.version import release_version
from rtc.runconfig import RunConfig
from rtc.version import VERSION as SOFTWARE_VERSION

from nisar.workflows.h5_prep import set_get_geo_info

BASE_HDF5_DATASET = f'/science/SENTINEL1'
FREQ_GRID_SUB_PATH = 'RTC/grids/frequencyA'
FREQ_GRID_DS = f'{BASE_HDF5_DATASET}/{FREQ_GRID_SUB_PATH}'

logger = logging.getLogger('rtc_s1')

def get_polygon_wkt(burst_in: Sentinel1BurstSlc):
    '''
    Get WKT for butst's bounding polygon
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

    if len(burst_in.border) ==1:
        geometry_polygon = burst_in.border[0]
    else:
        geometry_polygon = shapely.geometry.MultiPolygon(burst_in.border)
    
    return geometry_polygon.wkt


def save_hdf5_file(hdf5_obj, output_hdf5_file, flag_apply_rtc, clip_max,
                   clip_min, output_radiometry_str,
                   geogrid, pol_list, geo_burst_filename, nlooks_file,
                   rtc_anf_file, layover_shadow_mask_file,
                   radar_grid_file_dict,
                   save_imagery=True, save_secondary_layers=True):

    # save grids metadata
    h5_ds = os.path.join(FREQ_GRID_DS, 'listOfPolarizations')
    if h5_ds in hdf5_obj:
        del hdf5_obj[h5_ds]
    pol_list_s2 = np.array(pol_list, dtype='S2')
    dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
    dset.attrs['description'] = np.string_(
                'List of processed polarization layers')

    h5_ds = os.path.join(FREQ_GRID_DS, 'radiometricTerrainCorrectionFlag')
    if h5_ds in hdf5_obj:
        del hdf5_obj[h5_ds]
    dset = hdf5_obj.create_dataset(h5_ds, data=bool(flag_apply_rtc))



    # save geogrid coordinates
    yds, xds = set_get_geo_info(hdf5_obj, FREQ_GRID_DS, geogrid)

    if save_imagery:
        # save RTC imagery
        save_hdf5_dataset(geo_burst_filename, hdf5_obj, FREQ_GRID_DS,
                           yds, xds, pol_list,
                           long_name=output_radiometry_str,
                           units='',
                           valid_min=clip_min,
                           valid_max=clip_max)

    if save_secondary_layers:
        # save nlooks
        if nlooks_file:
            save_hdf5_dataset(nlooks_file, hdf5_obj, FREQ_GRID_DS,
                               yds, xds, 'numberOfLooks',
                               long_name = 'number of looks',
                               units = '',
                               valid_min = 0)

        # save rtc
        if rtc_anf_file:
            save_hdf5_dataset(rtc_anf_file, hdf5_obj, FREQ_GRID_DS,
                               yds, xds, 'RTCAreaNormalizationFactor',
                               long_name = 'RTC area factor',
                               units = '',
                               valid_min = 0)

        # save layover shadow mask
        if layover_shadow_mask_file:
            save_hdf5_dataset(layover_shadow_mask_file, hdf5_obj, FREQ_GRID_DS,
                               yds, xds, 'layoverShadowMask',
                               long_name = 'Layover/shadow mask',
                               units = '',
                               valid_min = 0)

        for ds_hdf5, filename in radar_grid_file_dict.items():
             save_hdf5_dataset(filename, hdf5_obj, FREQ_GRID_DS, yds, xds, ds_hdf5,
                                long_name = '', units = '')

    logger.info(f'file saved: {output_hdf5_file}')


def create_hdf5_file(product_id, output_hdf5_file, orbit, burst, cfg):
    hdf5_obj = h5py.File(output_hdf5_file, 'w')
    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
    hdf5_obj.attrs["contact"] = np.string_("operaops@jpl.nasa.gov")
    hdf5_obj.attrs["institution"] = np.string_("NASA JPL")
    hdf5_obj.attrs["mission_name"] = np.string_("OPERA")
    hdf5_obj.attrs["reference_document"] = np.string_("TBD")
    hdf5_obj.attrs["title"] = np.string_("OPERA L2 RTC-S1 Product")

    populate_metadata_group(product_id, hdf5_obj, burst, cfg)

    # save orbit
    orbit_group = hdf5_obj.require_group(
        f'{BASE_HDF5_DATASET}/RTC/metadata/orbit')
    save_orbit(orbit, orbit_group)
    return hdf5_obj


def save_orbit(orbit, orbit_group):
    orbit.save_to_h5(orbit_group)
    # Add description attributes.
    orbit_group["time"].attrs["description"] = np.string_("Time vector record. This"
        " record contains the time corresponding to position, velocity,"
        " acceleration records")
    orbit_group["position"].attrs["description"] = np.string_("Position vector"
        " record. This record contains the platform position data with"
        " respect to WGS84 G1762 reference frame")
    orbit_group["velocity"].attrs["description"] = np.string_("Velocity vector"
        " record. This record contains the platform velocity data with"
        " respect to WGS84 G1762 reference frame")
    orbit_group.create_dataset(
        'referenceEpoch',
        data=np.string_(orbit.reference_epoch.isoformat()))

    # Orbit source/type
    # TODO: Update orbit type:
    d = orbit_group.require_dataset("orbitType", (), "S10", data=np.string_("POE"))
    d.attrs["description"] = np.string_("PrOE (or) NOE (or) MOE (or) POE"
                                        " (or) Custom")


def get_metadata_dict(product_id: str,
                      burst_in: Sentinel1BurstSlc,
                      cfg_in: RunConfig):
    '''Create RTC-S1 metadata dictionary

    Parameters
    -----------
    product_id: str
        Product ID
    burst_in: Sentinel1BurstCls
        Source burst of the RTC
    cfg_in: RunConfig
        A class that contains the information defined in runconfig

    Returns
    -------
    metadata_dict : dict
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

    # product version
    product_version_float = cfg_in.groups.product_group.product_version
    if product_version_float is None:
        product_version = SOFTWARE_VERSION
    else:
        product_version = f'{product_version_float:.1f}'

    # DEM description
    dem_description = cfg_in.dem_description

    if not dem_description:
        # If the DEM description is not provided, use DEM source
        dem_description = os.path.basename(cfg_in.dem)

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

    # mission_id = 'Sentinel'
    beam_id = str(burst_in.burst_id)[-3:]

    # Manifests the field names, corresponding values from RTC workflow, and the description.
    # To extend this, add the lines with the format below:
    # 'field_name': [corresponding_variables_in_workflow, description]
    metadata_dict = {
        'identification/absoluteOrbitNumber':
            ['absolute_orbit_number', 
             burst_in.abs_orbit_number,
             'Absolute orbit number'],
        # NOTE: The field below does not exist on opera_rtc.xml
        # 'identification/relativeOrbitNumber':
        #   [int(burst_in.burst_id[1:4]), 'Relative orbit number'],
        'identification/trackNumber':
            ['track_number', burst_in.burst_id.track_number,
             'Track number'],
        'identification/burstID':
            ['burst_id', str(burst_in.burst_id),
             'Burst identification (burst ID)'],
        'identification/boundingPolygon':
            ['bounding_polygon', get_polygon_wkt(burst_in),
            'OGR compatible WKT representation of bounding polygon of the image'],
        # 'identification/missionId':
        #    [mission_id, 'Mission identifier'],
        'identification/platformId':
            ['platform_id', platform_id, 'Platform identifier'],
        'identification/instrumentName':
            ['instrument_name', platform_id, 'Instrument name'],
        'identification/productType':
            ['product_type', 'RTC-S1', 'Product type'],
        'identification/projectName':
            ['project_name', 'OPERA', 'Project name'],
        'identification/acquisitionMode':
            ['acquisition_mode', 'Interferometric Wide (IW)',
            'Acquisition mode'],
        'identification/beamID':
            ['beam_id', beam_id, 'Beam identification (Beam ID)'],
        'identification/lookDirection':
            ['look_direction', 'right', 'Look direction can be left or right'],
        'identification/orbitPassDirection':
            ['orbit_pass_direction', burst_in.orbit_direction.lower(),
            'Orbit direction can be ascending or descending'],
        # NOTE: using the same date format as `s1_reader.as_datetime()`
        'identification/zeroDopplerStartTime':
            ['zero_doppler_start_time',
             burst_in.sensing_start.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
             'Azimuth start time of product'],
        'identification/zeroDopplerEndTime':
            ['zero_doppler_end_time',
             burst_in.sensing_stop.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'Azimuth stop time of product'],
        'identification/listOfFrequencies':
             [None, ['A'],
             'List of frequency layers available in the product'],  # TBC
        'identification/isGeocoded':
            [None, True,
            'Flag to indicate radar geometry or geocoded product'],
        'identification/productLevel':
            ['product_level', 'L2', 'Product level'],
        'identification/productID':
            ['product_id', product_id, 'Product identificator'],
        # 'identification/productSource':
        # [platform_id, 'Product source'],
        'identification/isUrgentObservation':
            ['is_urgent_observation', False,
             'List of booleans indicating if datatakes are nominal or urgent'],
        'identification/diagnosticModeFlag':
            ['diagnostic_mode_flag', False,
             'Indicates if the radar mode is a diagnostic mode or not: True or False'],
        'identification/processingType':
            ['processing_type', processing_type,
             'NOMINAL (or) URGENT (or) CUSTOM (or) UNDEFINED'],
        # datetime format 'YYYY-MM-DD HH:MM:SS'
        'identification/processingDateTime':
            ['processing_data_time',
             datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
             'Processing date and time'],
        'identification/productVersion':
            ['product_version', product_version, 'Product version'],
        'identification/softwareVersion':
            ['software_version', str(SOFTWARE_VERSION), 'Software version'],
        # 'identification/CEOSDocumentIdentifier':
        #     ["https://ceos.org/ard/files/PFS/NRB/v5.5/CARD4L-PFS_NRB_v5.5.pdf",
        #      'CEOS document identifier'],
        # 'identification/frameNumber':  # TBD
        # 'identification/plannedDatatakeId':
        # 'identification/plannedObservationId':

        f'{FREQ_GRID_SUB_PATH}/rangeBandwidth':
            ['range_bandwidth', burst_in.range_bandwidth,
             'Processed range bandwidth in Hz'],
        # 'frequencyA/azimuthBandwidth':
        f'{FREQ_GRID_SUB_PATH}/centerFrequency':
            ['center_frequency', burst_in.radar_center_frequency,
             'Center frequency of the processed image in Hz'],
        f'{FREQ_GRID_SUB_PATH}/slantRangeSpacing':
            ['slant_range_spacing', burst_in.range_pixel_spacing,
             'Slant range spacing of grid. '
             'Same as difference between consecutive samples in slantRange array'],
        f'{FREQ_GRID_SUB_PATH}/zeroDopplerTimeSpacing':
            ['zero_doppler_time_spacing', burst_in.azimuth_time_interval,
             'Time interval in the along track direction for raster layers. This is same '
             'as the spacing between consecutive entries in the zeroDopplerTime array'],
        f'{FREQ_GRID_SUB_PATH}/faradayRotationFlag':
            ['faraday_rotation_flag', False,
             'Flag to indicate if Faraday Rotation correction was applied'],
        f'{FREQ_GRID_SUB_PATH}/polarizationOrientationFlag':
            ['polarization_orientation_flag', False,
            'Flag to indicate if Polarization Orientation correction was applied'],

        'RTC/metadata/processingInformation/algorithms/demInterpolation':
            ['dem_interpolation_algorithm',
             cfg_in.groups.processing.dem_interpolation_method, 'DEM interpolation method'],
        'RTC/metadata/processingInformation/algorithms/geocoding':
            ['geocoding_algorithm', cfg_in.groups.processing.geocoding.algorithm_type,
             'Geocoding algorithm'],
        'RTC/metadata/processingInformation/algorithms/radiometricTerrainCorrection':
            ['radiometric_terrain_correction_algorithm',
             cfg_in.groups.processing.rtc.algorithm_type,
            'Radiometric terrain correction (RTC) algorithm'],
        'RTC/metadata/processingInformation/algorithms/ISCEVersion':
            ['isce3_version', isce3.__version__, 'ISCE version used for processing'],
        # 'RTC/metadata/processingInformation/algorithms/RTCVersion':
        #     [str(SOFTWARE_VERSION), 'RTC-S1 SAS version used for processing'],
        'RTC/metadata/processingInformation/algorithms/S1ReaderVersion':
            ['s1_reader_version', release_version,
             'S1-Reader version used for processing'],

        'RTC/metadata/processingInformation/inputs/l1SLCGranules':
            ['l1_slc_granules', l1_slc_granules,
             'List of input L1 RSLC products used'],
        'RTC/metadata/processingInformation/inputs/orbitFiles':
            ['orbit_files', orbit_files, 'List of input orbit files used'],
        'RTC/metadata/processingInformation/inputs/auxCalFiles':
            ['aux_cal_files', [burst_in.burst_calibration.basename_cads,
             burst_in.burst_noise.basename_nads],
             'List of input calibration files used'],
        'RTC/metadata/processingInformation/inputs/configFiles':
            ['config_files', cfg_in.run_config_path,
             'List of input config files used'],
        'RTC/metadata/processingInformation/inputs/demSource':
            ['dem_source', dem_description, 'DEM source description']
    }
    return metadata_dict


def populate_metadata_group(product_id: str,
                            h5py_obj: h5py.File,
                            burst_in: Sentinel1BurstSlc,
                            cfg_in: RunConfig,
                            root_path: str = BASE_HDF5_DATASET):
    '''Populate RTC metadata based on Sentinel1BurstSlc and RunConfig

    Parameters
    -----------
    product_id: str
        Product ID
    h5py_obj: h5py.File
        HDF5 object into which write the metadata
    burst_in: Sentinel1BurstCls
        Source burst of the RTC
    cfg_in: RunConfig
        A class that contains the information defined in runconfig
    root_path: str
        Root path inside the HDF5 object on which the metadata will be placed
    '''

    metadata_dict = get_metadata_dict(product_id, burst_in, cfg_in)

    for fieldname, (_, data, description) in metadata_dict.items():
        path_dataset_in_h5 = os.path.join(root_path, fieldname)
        if data is str:
            dset = h5py_obj.create_dataset(path_dataset_in_h5, data=np.string_(data))
        else:
            dset = h5py_obj.create_dataset(path_dataset_in_h5, data=data)

        dset.attrs['description'] = np.string_(description)


def save_hdf5_dataset(ds_filename, h5py_obj, root_path,
                       yds, xds, ds_name, standard_name=None,
                       long_name=None, units=None, fill_value=None,
                       valid_min=None, valid_max=None, compute_stats=True):
    '''
    write temporary raster file contents to HDF5

    Parameters
    ----------
    ds_filename: string
        source raster file
    h5py_obj: h5py object
        h5py object of destination HDF5
    root_path: string
        path of output raster data
    yds: h5py dataset object
        y-axis dataset
    xds: h5py dataset object
        x-axis dataset
    ds_name: string
        name of dataset to be added to root_path
    standard_name: string, optional
    long_name: string, optional
    units: string, optional
    fill_value: float, optional
    valid_min: float, optional
    valid_max: float, optional
    '''
    if not os.path.isfile(ds_filename):
        return

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
        dset.attrs['grid_mapping'] = np.string_("projection")

        if standard_name is not None:
            dset.attrs['standard_name'] = np.string_(standard_name)

        if long_name is not None:
            dset.attrs['long_name'] = np.string_(long_name)

        if units is not None:
            dset.attrs['units'] = np.string_(units)

        if fill_value is not None:
            dset.attrs.create('_FillValue', data=fill_value)
        elif 'cfloat' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan + 1j * np.nan)
        elif 'float' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan)

        if stats_vector is not None:
            stats_obj = stats_vector[band]
            dset.attrs.create('min_value', data=stats_obj.min)
            dset.attrs.create('mean_value', data=stats_obj.mean)
            dset.attrs.create('max_value', data=stats_obj.max)
            dset.attrs.create('sample_standard_deviation', data=stats_obj.sample_stddev)

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

