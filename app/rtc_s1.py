'''
RTC Workflow
'''

import os
from subprocess import call
import time

import isce3
import journal
import numpy as np
from osgeo import gdal
import h5py

from s1reader.s1_burst_slc import Sentinel1BurstSlc

from rtc.geogrid import snap_coord
from rtc.geo_runconfig import GeoRunConfig
from rtc.yaml_argparse import YamlArgparse
from rtc import mosaic_geobursts

from nisar.workflows.h5_prep import set_get_geo_info


def _update_mosaic_boundaries(mosaic_geogrid_dict, geogrid):
    xf = geogrid.start_x + geogrid.spacing_x * geogrid.width
    yf = geogrid.start_y + geogrid.spacing_y * geogrid.length
    if ('x0' not in mosaic_geogrid_dict.keys() or
            geogrid.start_x < mosaic_geogrid_dict['x0']):
        mosaic_geogrid_dict['x0'] = geogrid.start_x
    if ('xf' not in mosaic_geogrid_dict.keys() or
            xf > mosaic_geogrid_dict['xf']):
        mosaic_geogrid_dict['xf'] = xf
    if ('y0' not in mosaic_geogrid_dict.keys() or
            geogrid.start_y > mosaic_geogrid_dict['y0']):
        mosaic_geogrid_dict['y0'] = geogrid.start_y
    if ('yf' not in mosaic_geogrid_dict.keys() or
            yf < mosaic_geogrid_dict['yf']):
        mosaic_geogrid_dict['yf'] = yf
    if 'dx' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['dx'] = geogrid.spacing_x
    else:
        assert(mosaic_geogrid_dict['dx'] == geogrid.spacing_x)
    if 'dy' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['dy'] = geogrid.spacing_y
    else:
        assert(mosaic_geogrid_dict['dy'] == geogrid.spacing_y)
    if 'epsg' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['epsg'] = geogrid.epsg
    else:
        assert(mosaic_geogrid_dict['epsg'] == geogrid.epsg)


def _get_raster(output_dir, ds_name, ds_description, dtype, shape,
                radar_grid_file_dict, output_obj_list, flag_save_vector_1,
                extension):
    if flag_save_vector_1 is not True:
        return None

    output_file = os.path.join(output_dir, ds_name)+'.'+extension
    raster_obj = isce3.io.Raster(
        output_file,
        shape[2],
        shape[1],
        shape[0],
        dtype,
        "GTiff")
    output_obj_list.append(raster_obj)
    radar_grid_file_dict[output_file] = ds_description
    return raster_obj


def _add_output_to_output_metadata_dict(flag, key, output_dir,
        output_metadata_dict, product_id, extension):
    if not flag:
        return
    output_image_list = []
    output_metadata_dict[key] = \
        [os.path.join(output_dir, f'{product_id}_{key}.{extension}'),
                      output_image_list]


def apply_slc_corrections(info_channel,
                          burst_in: Sentinel1BurstSlc,
                          path_slc_vrt: str,
                          path_slc_out: str,
                          flag_output_complex: bool = False,
                          flag_thermal_correction: bool = True,
                          flag_apply_abs_rad_correction: bool = True):
    '''Apply thermal correction stored in burst_in. Save the corrected signal
    back to ENVI format. Preserves the phase.'''

    # Load the SLC of the burst
    burst_in.slc_to_vrt_file(path_slc_vrt)
    raster_slc_from = gdal.Open(path_slc_vrt)
    arr_slc_from = raster_slc_from.ReadAsArray()

    # Apply the correction
    if flag_thermal_correction:
        info_channel.log(f'applying thermal noise correction to SLC')
        corrected_image = np.abs(arr_slc_from) ** 2 - burst_in.thermal_noise_lut
        min_backscatter = 0
        max_backscatter = None
        corrected_image = np.clip(corrected_image, min_backscatter,
                                  max_backscatter)
    else:
        corrected_image=np.abs(arr_slc_from) ** 2

    if flag_apply_abs_rad_correction:
        info_channel.log(f'applying absolute radiometric correction to SLC')
    if flag_output_complex:
        factor_mag = np.sqrt(corrected_image) / np.abs(arr_slc_from)
        factor_mag[np.isnan(factor_mag)] = 0.0
        corrected_image = arr_slc_from * factor_mag
        dtype = gdal.GDT_CFloat32
        if flag_apply_abs_rad_correction:
            corrected_image = \
                corrected_image / burst_in.burst_calibration.beta_naught
    else:
        dtype = gdal.GDT_Float32
        if flag_apply_abs_rad_correction:
            corrected_image = \
                corrected_image / burst_in.burst_calibration.beta_naught ** 2

    # Save the corrected image
    drvout = gdal.GetDriverByName('GTiff')
    raster_out = drvout.Create(path_slc_out, burst_in.shape[1],
                               burst_in.shape[0], 1, dtype)
    band_out = raster_out.GetRasterBand(1)
    band_out.WriteArray(corrected_image)
    band_out.FlushCache()
    del band_out


def run(cfg):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig `cfg`

    Parameters
    ---------
    cfg : dict
        Dictionary with user runconfig options
    '''
    info_channel = journal.info("rtc.run")


    # Start tracking processing time
    t_start = time.time()
    time_stamp = str(float(time.time()))
    temp_suffix = f'temp_{time_stamp}'
    info_channel.log("Starting geocode burst")


    dem_interp_method_enum = cfg.groups.processing.dem_interpolation_method_enum

    # read product path group / output format
    product_id = cfg.groups.product_path_group.product_id
    if product_id is None:
        product_id = 'rtc_product'
    product_path = cfg.groups.product_path_group.product_path
    scratch_path = cfg.groups.product_path_group.scratch_path
    output_dir = cfg.groups.product_path_group.output_dir
    flag_mosaic = cfg.groups.product_path_group.mosaic_bursts

    output_format = cfg.geocoding_params.output_format
    flag_hdf5 = output_format == 'HDF5'
    if flag_hdf5:
        output_raster_format = 'GTiff'
    else:
        output_raster_format = cfg.geocoding_params.output_format
    if output_raster_format == 'GTiff':
        extension = 'tif'
    else:
        extension = 'bin'
    
    # unpack geocode run parameters
    geocode_namespace = cfg.groups.processing.geocoding
    geocode_algorithm = geocode_namespace.algorithm_type
    output_mode = geocode_namespace.output_mode
    flag_apply_rtc = geocode_namespace.apply_rtc
    flag_apply_thermal_noise_correction = \
        geocode_namespace.apply_thermal_noise_correction
    flag_apply_abs_rad_correction=True
    memory_mode = geocode_namespace.memory_mode
    geogrid_upsampling = geocode_namespace.geogrid_upsampling
    clip_max = geocode_namespace.clip_max
    clip_min = geocode_namespace.clip_min
    # geogrids = geocode_namespace.geogrids
    flag_upsample_radar_grid = geocode_namespace.upsample_radargrid
    flag_save_incidence_angle = geocode_namespace.save_incidence_angle
    flag_save_local_inc_angle = geocode_namespace.save_local_inc_angle
    flag_save_projection_angle = geocode_namespace.save_projection_angle
    flag_save_simulated_radar_brightness = \
        geocode_namespace.save_simulated_radar_brightness
    flag_save_range_slope_angle = \
        geocode_namespace.save_range_slope_angle
    flag_save_nlooks = geocode_namespace.save_nlooks
    flag_save_rtc_anf = geocode_namespace.save_rtc_anf
    flag_save_dem = geocode_namespace.save_dem

    flag_call_radar_grid = (flag_save_incidence_angle or
        flag_save_local_inc_angle or flag_save_projection_angle or
        flag_save_simulated_radar_brightness or flag_save_dem or
        flag_save_range_slope_angle)

    # unpack RTC run parameters
    rtc_namespace = cfg.groups.processing.rtc
    output_terrain_radiometry = rtc_namespace.output_type
    rtc_algorithm = rtc_namespace.algorithm_type
    input_terrain_radiometry = rtc_namespace.input_terrain_radiometry
    rtc_min_value_db = rtc_namespace.rtc_min_value_db
    rtc_upsampling = rtc_namespace.dem_upsampling
    if (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT):
        output_radiometry_str = "radar backscatter sigma0"
    elif (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT):
        output_radiometry_str = 'radar backscatter gamma0'
    elif input_terrain_radiometry == isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT:
        output_radiometry_str = 'radar backscatter beta0'
    else:
        output_radiometry_str = 'radar backscatter sigma0'

    # Common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    zero_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    maxiter = cfg.geo2rdr_params.numiter
    exponent = 1 if (flag_apply_thermal_noise_correction or
                     flag_apply_abs_rad_correction) else 2

    # output mosaics
    geo_filename = f'{output_dir}/'f'{product_id}.{extension}'
    output_imagery_list = []
    output_file_list = []
    output_metadata_dict = {}

    _add_output_to_output_metadata_dict(
        flag_save_nlooks, 'nlooks', output_dir, output_metadata_dict,
        product_id, extension)
    _add_output_to_output_metadata_dict(
        flag_save_rtc_anf, 'rtc', output_dir, output_metadata_dict,
        product_id, extension)

    mosaic_geogrid_dict = {}
    temp_files_list = []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)
    vrt_options_mosaic = gdal.BuildVRTOptions(separate=True)

    n_bursts = len(cfg.bursts.items())
    print('number of bursts to process:', n_bursts)

    # iterate over sub-burts
    for burst_index, (burst_id, burst_pol_dict) in enumerate(cfg.bursts.items()):
        
        # ===========================================================
        # start burst processing

        t_burst_start = time.time()
        info_channel.log(f'processing burst: {burst_id} ({burst_index+1}/'
                         f'{n_bursts})')

        pols = list(burst_pol_dict.keys())
        burst = burst_pol_dict[pols[0]]

        flag_bursts_files_are_temporary = \
            flag_hdf5 or (flag_mosaic and not n_bursts == 1)

        burst_scratch_path = f'{scratch_path}/{burst_id}/'
        os.makedirs(burst_scratch_path, exist_ok=True)

        if flag_bursts_files_are_temporary:
            bursts_output_dir = burst_scratch_path
        else:
            bursts_output_dir = os.path.join(output_dir, burst_id)
            os.makedirs(bursts_output_dir, exist_ok=True)
        
        geogrid = cfg.geogrids[burst_id]

        # snap coordinates
        x_snap = geogrid.spacing_x
        y_snap = geogrid.spacing_y
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)

        # update mosaic boundaries
        _update_mosaic_boundaries(mosaic_geogrid_dict, geogrid)

        info_channel.log(f'reading burst SLC')
        radar_grid = burst.as_isce3_radargrid()
        # native_doppler = burst.doppler.lut2d
        orbit = burst.orbit
        if 'orbit' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['orbit'] = orbit
        if 'wavelength' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['wavelength'] = burst.wavelength
        if 'lookside' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['lookside'] = radar_grid.lookside

        input_file_list = []
        pol_list = list(burst_pol_dict.keys())
        for pol, burst_pol in burst_pol_dict.items():
            temp_slc_path = \
                f'{burst_scratch_path}/rslc_{pol}_{temp_suffix}.vrt'
            temp_slc_corrected_path = (
                f'{burst_scratch_path}/rslc_{pol}_corrected_{temp_suffix}'
                f'.{extension}')
            burst_pol.slc_to_vrt_file(temp_slc_path)

            if (flag_apply_thermal_noise_correction or
                    flag_apply_abs_rad_correction):
                apply_slc_corrections(
                    info_channel, burst_pol, temp_slc_path,
                    temp_slc_corrected_path,
                    flag_output_complex=False,
                    flag_thermal_correction=flag_apply_thermal_noise_correction,
                    flag_apply_abs_rad_correction=True)
                input_burst_filename = temp_slc_corrected_path
            else:
                input_burst_filename = temp_slc_path

            temp_files_list.append(input_burst_filename)
            input_file_list.append(input_burst_filename)

        # create multi-band VRT
        if len(input_file_list) == 1:
            rdr_burst_raster = isce3.io.Raster(input_file_list[0])
        else:
            temp_vrt_path = f'{burst_scratch_path}/rslc_{temp_suffix}.vrt'
            gdal.BuildVRT(temp_vrt_path, input_file_list, options=vrt_options_mosaic)
            rdr_burst_raster = isce3.io.Raster(temp_vrt_path)
            temp_files_list.append(temp_vrt_path)

        # Generate output geocoded burst raster
        if flag_bursts_files_are_temporary:
            # files are temporary
            geo_burst_filename = \
                f'{burst_scratch_path}/{product_id}_{temp_suffix}.{extension}'
            temp_files_list.append(geo_burst_filename)
        else:
            os.makedirs(f'{output_dir}/{burst_id}', exist_ok=True)
            geo_burst_filename = \
                f'{output_dir}/{burst_id}/{product_id}.{extension}'
            output_file_list.append(geo_burst_filename)
        
        geo_burst_raster = isce3.io.Raster(
            geo_burst_filename,
            geogrid.width, geogrid.length,
            rdr_burst_raster.num_bands, gdal.GDT_Float32,
            output_raster_format)

        # init Geocode object depending on raster type
        if rdr_burst_raster.datatype() == gdal.GDT_Float32:
            geo_obj = isce3.geocode.GeocodeFloat32()
        elif rdr_burst_raster.datatype() == gdal.GDT_Float64:
            geo_obj = isce3.geocode.GeocodeFloat64()
        elif rdr_burst_raster.datatype() == gdal.GDT_CFloat32:
            geo_obj = isce3.geocode.GeocodeCFloat32()
        elif rdr_burst_raster.datatype() == gdal.GDT_CFloat64:
            geo_obj = isce3.geocode.GeocodeCFloat64()
        else:
            err_str = 'Unsupported raster type for geocoding'
            raise NotImplementedError(err_str)

        # init geocode members
        geo_obj.orbit = orbit
        geo_obj.ellipsoid = ellipsoid
        geo_obj.doppler = zero_doppler
        geo_obj.threshold_geo2rdr = threshold
        geo_obj.numiter_geo2rdr = maxiter

        # set data interpolator based on the geocode algorithm
        if output_mode == isce3.geocode.GeocodeOutputMode.INTERP:
            geo_obj.data_interpolator = geocode_algorithm

        geo_obj.geogrid(geogrid.start_x, geogrid.start_y,
                        geogrid.spacing_x, geogrid.spacing_y,
                        geogrid.width, geogrid.length, geogrid.epsg)

        if flag_save_nlooks:
            if flag_bursts_files_are_temporary:
                nlooks_file = (f'{bursts_output_dir}/{product_id}'
                               f'_nlooks_{temp_suffix}.{extension}')
                temp_files_list.append(nlooks_file)
            else:
                nlooks_file = (f'{bursts_output_dir}/{product_id}'
                               f'_nlooks.{extension}')
                output_file_list.append(nlooks_file)
            out_geo_nlooks_obj = isce3.io.Raster(
                nlooks_file, geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, output_raster_format)
        else:
            nlooks_file = None
            out_geo_nlooks_obj = None

        if flag_save_rtc_anf:
            if flag_bursts_files_are_temporary:
                rtc_anf_file = (f'{bursts_output_dir}/{product_id}'
                               f'_rtc_anf_{temp_suffix}.{extension}')
                temp_files_list.append(rtc_anf_file)
            else:
                rtc_anf_file = (f'{bursts_output_dir}/{product_id}'
                               f'_rtc_anf.{extension}')
                output_file_list.append(rtc_anf_file)
            out_geo_rtc_obj = isce3.io.Raster(
                rtc_anf_file,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, output_raster_format)
        else:
            rtc_anf_file = None
            out_geo_rtc_obj = None

        # Extract burst boundaries and create sub_swaths object to mask
        # invalid radar samples
        n_subswaths = 1
        sub_swaths = isce3.product.SubSwaths(radar_grid.length,
                                             radar_grid.width,
                                             n_subswaths)
        last_range_sample = min([burst.last_valid_sample, radar_grid.width])
        valid_samples_sub_swath = np.repeat(
            [[burst.first_valid_sample, last_range_sample + 1]],
            radar_grid.length, axis=0)
        for i in range(burst.first_valid_line):
            valid_samples_sub_swath[i, :] = 0
        for i in range(burst.last_valid_line, radar_grid.length):
            valid_samples_sub_swath[i, :] = 0
        
        sub_swaths.set_valid_samples_array(1, valid_samples_sub_swath)

        # geocode
        geo_obj.geocode(radar_grid=radar_grid,
                        input_raster=rdr_burst_raster,
                        output_raster=geo_burst_raster,
                        dem_raster=dem_raster,
                        output_mode=output_mode,
                        geogrid_upsampling=geogrid_upsampling,
                        flag_apply_rtc=flag_apply_rtc,
                        input_terrain_radiometry=input_terrain_radiometry,
                        output_terrain_radiometry=output_terrain_radiometry,
                        exponent=exponent,
                        rtc_min_value_db=rtc_min_value_db,
                        rtc_upsampling=rtc_upsampling,
                        rtc_algorithm=rtc_algorithm,
                        flag_upsample_radar_grid=flag_upsample_radar_grid,
                        clip_min = clip_min,
                        clip_max = clip_max,
                        # radargrid_nlooks=radar_grid_nlooks,
                        # out_off_diag_terms=out_off_diag_terms_obj,
                        out_geo_nlooks=out_geo_nlooks_obj,
                        out_geo_rtc=out_geo_rtc_obj,
                        # out_geo_dem=out_geo_dem_obj,
                        input_rtc=None,
                        output_rtc=None,
                        dem_interp_method=dem_interp_method_enum,
                        memory_mode=memory_mode,
                        sub_swaths=sub_swaths)

        del geo_burst_raster
        if not flag_bursts_files_are_temporary:
            info_channel.log(f'file saved: {geo_burst_filename}')
        output_imagery_list.append(geo_burst_filename)

        if flag_save_nlooks:
            del out_geo_nlooks_obj
            if not flag_bursts_files_are_temporary:
                info_channel.log(f'file saved: {nlooks_file}')
            output_metadata_dict['nlooks'][1].append(nlooks_file)
    
        if flag_save_rtc_anf:
            del out_geo_rtc_obj
            if not flag_bursts_files_are_temporary:
                info_channel.log(f'file saved: {rtc_anf_file}')
            output_metadata_dict['rtc'][1].append(rtc_anf_file)

        radar_grid_file_dict = {}
        if flag_call_radar_grid and not flag_mosaic:
            get_radar_grid(
                geogrid, info_channel, dem_interp_method_enum, product_id,
                bursts_output_dir, extension, flag_save_incidence_angle,
                flag_save_local_inc_angle, flag_save_projection_angle,
                flag_save_simulated_radar_brightness,
                flag_save_range_slope_angle, flag_save_dem,
                dem_raster, radar_grid_file_dict,
                mosaic_geogrid_dict, orbit,
                verbose = not flag_bursts_files_are_temporary)
            if flag_hdf5:
                # files are temporary
                temp_files_list += list(radar_grid_file_dict.values())
            else:
                output_file_list += list(radar_grid_file_dict.values())
 
        if flag_hdf5 and not flag_mosaic:
            output_hdf5_file = f'{output_dir}/{burst_id}/{product_id}.h5'
            save_hdf5_file(
                info_channel, output_hdf5_file, flag_apply_rtc,
                clip_max, clip_min, output_radiometry_str, output_file_list,
                geogrid, pol_list, geo_burst_filename, nlooks_file,
                rtc_anf_file, radar_grid_file_dict)


        t_burst_end = time.time()
        info_channel.log(
            f'elapsed time (burst): {t_burst_end - t_burst_start}')

        # end burst processing
        # ===========================================================

    if flag_call_radar_grid and flag_mosaic:
        radar_grid_file_dict = {}

        get_radar_grid(cfg.geogrid, info_channel, dem_interp_method_enum, product_id,
                       output_dir, extension, flag_save_incidence_angle,
                       flag_save_local_inc_angle, flag_save_projection_angle,
                       flag_save_simulated_radar_brightness,
                       flag_save_range_slope_angle, flag_save_dem,
                       dem_raster, radar_grid_file_dict,
                       mosaic_geogrid_dict,
                       orbit, verbose = not flag_hdf5)
        if flag_hdf5:
            # files are temporary
            temp_files_list += list(radar_grid_file_dict.values())
        else:
            output_file_list += list(radar_grid_file_dict.values())

    if flag_mosaic:
        # mosaic sub-bursts
        geo_filename = f'{output_dir}/{product_id}.{extension}'
        info_channel.log(f'mosaicking file: {geo_filename}')

        nlooks_list = output_metadata_dict['nlooks'][1]
        mosaic_geobursts.weighted_mosaic(output_imagery_list, nlooks_list,
                                     geo_filename, cfg.geogrid, verbose=False)

        if flag_hdf5:
            temp_files_list.append(geo_filename)
        else:
            output_file_list.append(geo_filename)

        # mosaic other bands
        for key in output_metadata_dict.keys():
            output_file, input_files = output_metadata_dict[key]
            info_channel.log(f'mosaicking file: {output_file}')
            mosaic_geobursts.weighted_mosaic(input_files, nlooks_list,
                                             output_file,
                                             cfg.geogrid, verbose=False)
            if flag_hdf5:
                temp_files_list.append(output_file)
            else:
                output_file_list.append(output_file)

        if flag_hdf5:
            if flag_save_nlooks:
                nlooks_mosaic_file = output_metadata_dict['nlooks'][0]
            else:
                nlooks_mosaic_file = None
            if flag_save_rtc_anf:
                rtc_anf_mosaic_file = output_metadata_dict['rtc'][0]
            else:
                rtc_anf_mosaic_file = None
            output_hdf5_file = f'{output_dir}/{product_id}.h5'
            save_hdf5_file(info_channel, output_hdf5_file, flag_apply_rtc,
                           clip_max, clip_min, output_radiometry_str,
                           output_file_list, cfg.geogrid, pol_list,
                           geo_filename, nlooks_mosaic_file,
                           rtc_anf_mosaic_file, radar_grid_file_dict)

    info_channel.log('removing temporary files:')
    for filename in temp_files_list:
        if not os.path.isfile(filename):
            continue
        os.remove(filename)
        info_channel.log(f'    {filename}')

    info_channel.log('output files:')
    for filename in output_file_list:
        info_channel.log(f'    {filename}')


    t_end = time.time()
    info_channel.log(f'elapsed time: {t_end - t_start}')


def save_hdf5_file(info_channel, output_hdf5_file, flag_apply_rtc, clip_max,
                   clip_min, output_radiometry_str,
                   output_file_list, geogrid, pol_list, geo_burst_filename,
                   nlooks_file, rtc_anf_file, radar_grid_file_dict):

    hdf5_obj = h5py.File(output_hdf5_file, 'w')
    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
    root_ds = f'/science/CSAR/RTC/grids/frequencyA'

    h5_ds = os.path.join(root_ds, 'listOfPolarizations')
    if h5_ds in hdf5_obj:
        del hdf5_obj[h5_ds]
    pol_list_s2 = np.array(pol_list, dtype='S2')
    dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
    dset.attrs['description'] = np.string_(
                'List of processed polarization layers')

    h5_ds = os.path.join(root_ds, 'radiometricTerrainCorrectionFlag')
    if h5_ds in hdf5_obj:
        del hdf5_obj[h5_ds]
    dset = hdf5_obj.create_dataset(h5_ds, data=bool(flag_apply_rtc))

    # save geogrid coordinates
    yds, xds = set_get_geo_info(hdf5_obj, root_ds, geogrid)

    # save RTC imagery
    _save_hdf5_dataset(geo_burst_filename, hdf5_obj, root_ds,
                       yds, xds, pol_list,
                       long_name=output_radiometry_str,
                       units='',
                       valid_min=clip_min,
                       valid_max=clip_max)
    # save nlooks
    if nlooks_file:
        _save_hdf5_dataset(nlooks_file, hdf5_obj, root_ds,
                           yds, xds, 'numberOfLooks',
                           long_name = 'number of looks',
                           units = '',
                           valid_min = 0)

    # save rtc
    if rtc_anf_file:
        _save_hdf5_dataset(rtc_anf_file, hdf5_obj, root_ds,
                           yds, xds, 'areaNormalizationFactor',
                           long_name = 'RTC area factor',
                           units = '',
                           valid_min = 0)

    for filename, descr in radar_grid_file_dict.items():
         _save_hdf5_dataset(filename, hdf5_obj, root_ds, yds, xds, descr,
                            long_name = '', units = '')

    info_channel.log(f'file saved: {output_hdf5_file}')
    output_file_list.append(output_hdf5_file)


def _save_hdf5_dataset(ds_filename, h5py_obj, root_path,
                       yds, xds, ds_name, standard_name=None,
                       long_name=None, units=None, fill_value=None,
                       valid_min=None, valid_max=None, compute_stats=True):
    '''
    write temporary raster file contents to HDF5

    Parameters
    ----------
    ds_filename : string
        source raster file
    h5py_obj : h5py object
        h5py object of destination HDF5
    root_path : string
        path of output raster data
    yds : h5py dataset object
        y-axis dataset
    xds : h5py dataset object
        x-axis dataset
    ds_name : string
        name of dataset to be added to root_path
    standard_name : string, optional
    long_name : string, optional
    units : string, optional
    fill_value : float, optional
    valid_min : float, optional
    valid_max : float, optional
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



def get_radar_grid(geogrid, info_channel, dem_interp_method_enum, product_id,
                   output_dir, extension, flag_save_incidence_angle,
                   flag_save_local_inc_angle, flag_save_projection_angle,
                   flag_save_simulated_radar_brightness,
                   flag_save_range_slope_angle, flag_save_dem, dem_raster,
                   radar_grid_file_dict, mosaic_geogrid_dict, orbit,
                   verbose = True):
    output_obj_list = []
    layers_nbands = 1
    shape = [layers_nbands, geogrid.length, geogrid.width]

    incidence_angle_raster = _get_raster(
            output_dir, f'{product_id}_incidence_angle',
            'incidenceAngle', gdal.GDT_Float32, shape, radar_grid_file_dict,
            output_obj_list, flag_save_incidence_angle, extension)
    local_incidence_angle_raster = _get_raster(
            output_dir, f'{product_id}_local_incidence_angle',
            'localIncidenceAngle', gdal.GDT_Float32, shape,
            radar_grid_file_dict, output_obj_list, flag_save_local_inc_angle,
            extension)
    projection_angle_raster = _get_raster(
            output_dir, f'{product_id}_projection_angle',
            'projectionAngle', gdal.GDT_Float32, shape, radar_grid_file_dict,
            output_obj_list, flag_save_projection_angle, extension)
    simulated_radar_brightness_raster = _get_raster(
            output_dir, f'{product_id}_simulated_radar_brightness',
            'areaNormalizationFactorPsi', gdal.GDT_Float32, shape,
            radar_grid_file_dict, output_obj_list, 
            flag_save_simulated_radar_brightness, extension)
    range_slope_angle_raster = _get_raster(
            output_dir, f'{product_id}_range_slope_angle',
            'rangeSlope', gdal.GDT_Float32, shape, radar_grid_file_dict,
            output_obj_list, flag_save_range_slope_angle, extension)
    interpolated_dem_raster = _get_raster(
            output_dir, f'{product_id}_interpolated_dem',
            'interpolatedDem', gdal.GDT_Float32, shape, radar_grid_file_dict,
            output_obj_list, flag_save_dem, extension)

    # TODO review this (Doppler)!!!
    # native_doppler = burst.doppler.lut2d
    native_doppler = isce3.core.LUT2d()
    native_doppler.bounds_error = False
    grid_doppler = isce3.core.LUT2d()
    grid_doppler.bounds_error = False

    # call get_radar_grid()
    isce3.geogrid.get_radar_grid(mosaic_geogrid_dict['lookside'],
                                     mosaic_geogrid_dict['wavelength'],
                                     dem_raster,
                                     geogrid,
                                     orbit,
                                     native_doppler,
                                     grid_doppler,
                                     incidence_angle_raster =
                                        incidence_angle_raster,
                                     local_incidence_angle_raster =
                                        local_incidence_angle_raster,
                                     projection_angle_raster =
                                        projection_angle_raster,
                                     simulated_radar_brightness_raster =
                                        simulated_radar_brightness_raster,
                                     directional_slope_angle_raster =
                                        range_slope_angle_raster,
                                     interpolated_dem_raster =
                                        interpolated_dem_raster,
                                     dem_interp_method=dem_interp_method_enum)

    # Flush data
    for obj in output_obj_list:
        del obj

    if not verbose:
        return

    for filename in output_file_list:
        info_channel.log(f'file saved: {filename}')


def _load_parameters(cfg):
    '''
    Load GCOV specific parameters.
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

    geocode_algorithm = cfg.groups.processing.geocoding.algorithm_type
    if geocode_algorithm == "area_projection":
        output_mode = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION
    else:
        output_mode = isce3.geocode.GeocodeOutputMode.INTERP
    geocode_namespace.output_mode = output_mode

    # only 2 RTC algorithms supported: area_projection (default) &
    # bilinear_distribution
    if rtc_namespace.algorithm_type == "bilinear_distribution":
        rtc_namespace.algorithm_type = \
            isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION
    else:
        rtc_namespace.algorithm_type = \
            isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION

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
    elif (dem_interp_method == 'sinc'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.SINC
    elif (dem_interp_method == 'bilinear'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.BILINEAR
    elif (dem_interp_method == 'bicubic'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.BICUBIC
    elif (dem_interp_method == 'nearest'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.NEAREST
    else:
        err_msg = ('ERROR invalid DEM interpolation method:'
                   f' {dem_interp_method}')
        raise ValueError(err_msg)

    cfg.groups.processing.dem_interpolation_method_enum = \
        dem_interp_method_enum

if __name__ == "__main__":
    '''Run geocode rtc workflow from command line'''
    # load arguments from command line
    geo_parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    cfg = GeoRunConfig.load_from_yaml(geo_parser.run_config_path,
                                      'rtc_s1')

    _load_parameters(cfg)

    # Run geocode burst workflow
    run(cfg)
