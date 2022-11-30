#!/usr/bin/env python

import os
import numpy as np
import h5py
import logging
from osgeo import gdal

import isce3

from s1reader.s1_burst_slc import Sentinel1BurstSlc
from rtc.runconfig import RunConfig

from nisar.workflows.h5_prep import set_get_geo_info

BASE_DS = f'/science/CSAR'
FREQ_GRID_SUB_PATH = 'RTC/grids/frequencyA'
FREQ_GRID_DS = f'{BASE_DS}/{FREQ_GRID_SUB_PATH}'

logger = logging.getLogger('rtc_s1')


def save_hdf5_file(hdf5_obj, output_hdf5_file, flag_apply_rtc, clip_max,
                   clip_min, output_radiometry_str, output_file_list,
                   geogrid, pol_list, geo_burst_filename, nlooks_file,
                   rtc_anf_file, layover_shadow_mask_file,
                   radar_grid_file_dict, save_imagery=True):

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

    if not save_imagery:
        return

    # save geogrid coordinates
    yds, xds = set_get_geo_info(hdf5_obj, FREQ_GRID_DS, geogrid)

    # save RTC imagery
    save_hdf5_dataset(geo_burst_filename, hdf5_obj, FREQ_GRID_DS,
                       yds, xds, pol_list,
                       long_name=output_radiometry_str,
                       units='',
                       valid_min=clip_min,
                       valid_max=clip_max)
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
                           yds, xds, 'areaNormalizationFactor',
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
    output_file_list.append(output_hdf5_file)


def create_hdf5_file(output_hdf5_file, orbit, burst, cfg):
    hdf5_obj = h5py.File(output_hdf5_file, 'w')
    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
    hdf5_obj.attrs["contact"] = np.string_("operaops@jpl.nasa.gov")
    hdf5_obj.attrs["institution"] = np.string_("NASA JPL")
    hdf5_obj.attrs["mission_name"] = np.string_("OPERA")
    hdf5_obj.attrs["reference_document"] = np.string_("TBD")
    hdf5_obj.attrs["title"] = np.string_("OPERA L2 RTC-S1 Product")

    populate_metadata_group(hdf5_obj, burst, cfg)

    # save orbit
    orbit_group = hdf5_obj.require_group(f'{BASE_DS}/RTC/metadata/orbit')
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

    # Orbit source/type
    # TODO: Update orbit type:
    d = orbit_group.require_dataset("orbitType", (), "S10", data=np.string_("POE"))
    d.attrs["description"] = np.string_("PrOE (or) NOE (or) MOE (or) POE"
                                        " (or) Custom")


def populate_metadata_group(h5py_obj: h5py.File,
                            burst_in: Sentinel1BurstSlc,
                            cfg_in: RunConfig,
                            root_path: str = BASE_DS):
    '''Populate RTC metadata based on Sentinel1BurstSlc and RunConfig

    Parameters:
    -----------
    h5py_obj: h5py.File
        HDF5 object into which write the metadata
    burst_in: Sentinel1BurstCls
        Source burst of the RTC
    cfg_in: RunConfig
        A class that contains the information defined in runconfig
    root_path: str
        Root path inside the HDF5 object on which the metadata will be placed
    '''
    orbit_files = [os.path.basename(f) for f in cfg_in.orbit_path]
    l1_slc_granules = [os.path.basename(f) for f in cfg_in.safe_files]
    dem_files = [os.path.basename(cfg_in.dem)]

    # Manifests the field names, corresponding values from RTC workflow, and the description.
    # To extend this, add the lines with the format below:
    # 'field_name': [corresponding_variables_in_workflow, description]
    dict_field_and_data = {
        'identification/absoluteOrbitNumber':
            [burst_in.abs_orbit_number, 'Absolute orbit number'],
        # NOTE: The field below does not exist on opera_rtc.xml
        # 'identification/relativeOrbitNumber':
        #   [int(burst_in.burst_id[1:4]), 'Relative orbit number'],
        'identification/trackNumber':
            [int(str(burst_in.burst_id).split('_')[1]), 'Track number'],
        'identification/missionId':
            [burst_in.platform_id, 'Mission identifier'],
        # NOTE maybe `SLC` has to be sth. like RTC?
        'identification/productType':
            ['SLC', 'Product type'],
        # NOTE: in NISAR, the value has to be in UPPERCASE or lowercase?
        'identification/lookDirection':
            ['Right', 'Look direction can be left or right'],
        'identification/orbitPassDirection':
            [burst_in.orbit_direction, 'Orbit direction can be ascending or descending'],
        # NOTE: using the same date format as `s1_reader.as_datetime()`
        'identification/zeroDopplerStartTime':
            [burst_in.sensing_start.strftime('%Y-%m-%dT%H:%M:%S.%f'),
             'Azimuth start time of product'],
        'identification/zeroDopplerEndTime':
            [burst_in.sensing_stop.strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'Azimuth stop time of product'],
        'identification/listOfFrequencies':
             [['A'], 'List of frequency layers available in the product'],  # TBC
        'identification/isGeocoded':
            [True, 'Flag to indicate radar geometry or geocoded product'],
        'identification/isUrgentObservation':
            [False, 'List of booleans indicating if datatakes are nominal or urgent'],
        'identification/diagnosticModeFlag':
            [False, 'Indicates if the radar mode is a diagnostic mode or not: True or False'],
        'identification/processingType':
            ['UNDEFINED', 'NOMINAL (or) URGENT (or) CUSTOM (or) UNDEFINED'],
        # 'identification/frameNumber':  # TBD
        # 'identification/productVersion': # Defined by RTC SAS
        # 'identification/plannedDatatakeId':
        # 'identification/plannedObservationId':

        f'{FREQ_GRID_SUB_PATH}/rangeBandwidth':
            [burst_in.range_bandwidth, 'Processed range bandwidth in Hz'],
        # 'frequencyA/azimuthBandwidth':
        f'{FREQ_GRID_SUB_PATH}/centerFrequency':
            [burst_in.radar_center_frequency, 'Center frequency of the processed image in Hz'],
        f'{FREQ_GRID_SUB_PATH}/slantRangeSpacing':
            [burst_in.range_pixel_spacing,
             'Slant range spacing of grid. '
             'Same as difference between consecutive samples in slantRange array'],
        f'{FREQ_GRID_SUB_PATH}/zeroDopplerTimeSpacing':
            [burst_in.azimuth_time_interval,
             'Time interval in the along track direction for raster layers. This is same '
             'as the spacing between consecutive entries in the zeroDopplerTime array'],
        f'{FREQ_GRID_SUB_PATH}/faradayRotationFlag':
            [False, 'Flag to indicate if Faraday Rotation correction was applied'],
        f'{FREQ_GRID_SUB_PATH}/polarizationOrientationFlag':
            [False, 'Flag to indicate if Polarization Orientation correction was applied'],

        'RTC/metadata/processingInformation/algorithms/demInterpolation':
            [cfg_in.groups.processing.dem_interpolation_method, 'DEM interpolation method'],
        'RTC/metadata/processingInformation/algorithms/geocoding':
            [cfg_in.groups.processing.geocoding.algorithm_type, 'Geocoding algorithm'],
        'RTC/metadata/processingInformation/algorithms/radiometricTerrainCorrection':
            [cfg_in.groups.processing.rtc.algorithm_type,
            'Radiometric terrain correction (RTC) algorithm'],
        'RTC/metadata/processingInformation/algorithms/ISCEVersion':
            [isce3.__version__, 'ISCE version used for processing'],

        'RTC/metadata/processingInformation/inputs/l1SlcGranules':
            [l1_slc_granules, 'List of input L1 RSLC products used'],
        'RTC/metadata/processingInformation/inputs/orbitFiles':
            [orbit_files, 'List of input orbit files used'],
        'RTC/metadata/processingInformation/inputs/auxcalFiles':
            [[burst_in.burst_calibration.basename_cads, burst_in.burst_noise.basename_nads],
             'List of input calibration files used'],
        'RTC/metadata/processingInformation/inputs/configFiles':
            [cfg_in.run_config_path, 'List of input config files used'],
        'RTC/metadata/processingInformation/inputs/demFiles':
            [dem_files, 'List of input dem files used']
    }
    for fieldname, data in dict_field_and_data.items():
        path_dataset_in_h5 = os.path.join(root_path, fieldname)
        if data[0] is str:
            dset = h5py_obj.create_dataset(path_dataset_in_h5, data=np.string_(data[0]))
        else:
            dset = h5py_obj.create_dataset(path_dataset_in_h5, data=data[0])

        dset.attrs['description'] = np.string_(data[1])


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

    gdal_ds = gdal.Open(ds_filename)
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

