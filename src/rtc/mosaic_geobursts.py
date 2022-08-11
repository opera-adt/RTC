'''
A module to mosaic Sentinel-1 geobursts from RTC workflow
'''


import os

import numpy as np
from osgeo import gdal



import matplotlib.pyplot as plt
import glob

def weighted_mosaic(list_rtc, list_nlooks):
    '''
    Mosaic the snapped S1 geobursts
    paremeters:
    -----------
        list_rtc: list
            list of the path to the rtc geobursts
        list_nlooks list
            list of the nlooks raster that corresponds to list_rtc

    '''

    num_raster = len(list_rtc)
    num_bands = None
    posting_x = None
    posting_y = None

    # load geotransformation
    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)

    for i, path_rtc in enumerate(list_rtc):
        print(f'Processing: {i+1} of {num_raster}')
        raster_in = gdal.Open(path_rtc, 0)
        list_geo_transform[i, :] = raster_in.GetGeoTransform()
        list_dimension[i, :] = (raster_in.RasterYSize, raster_in.RasterXSize)

        # Check if the number of bands are consistent over the input RTC rasters
        if num_bands is None:
            num_bands = raster_in.RasterCount
            continue
        elif num_bands != raster_in.RasterCount:
            raise ValueError(f'Different number of bands detected from RTC file: {os.path.basename(path_rtc)}')

        # Check the posting x and y of the RTC bursts. Determine the postings of the output mosaic
        if posting_x is None:
            posting_x = list_geo_transform[i, 1]
            continue
        elif posting_x != list_geo_transform[i, 1]:
            raise ValueError(f'Abnormaly detected from posting X: {os.path.basename(path_rtc)}')

        if posting_y is None:
            posting_y = list_geo_transform[i, 5]
            continue
        elif posting_y != list_geo_transform[i, 5]:
            raise ValueError(f'Abnormaly detected from posting Y: {os.path.basename(path_rtc)}')

        raster_in = None

    #determine the dimension and the upper left corner of the output mosaic
    #for i, geo_transform in enumerate(list_geo_transform):
    xmin_mosaic = list_geo_transform[:, 0].min()
    ymax_mosaic = list_geo_transform[:, 3].max()
    xmax_mosaic = (list_geo_transform[:, 0]+list_geo_transform[:, 1]*list_dimension[:, 1]).max()
    ymin_mosaic = (list_geo_transform[:, 3]+list_geo_transform[:, 5]*list_dimension[:, 0]).min()

    #posting_x=list_geo_transform[0,1]
    #posting_y=list_geo_transform[0,5]

    dim_mosaic = (int(np.ceil((ymin_mosaic - ymax_mosaic)/posting_y)),
                  int(np.ceil((xmax_mosaic - xmin_mosaic)/posting_x)))

    if num_bands == 1:
        arr_numerator = np.zeros(dim_mosaic)

    else:
        arr_numerator = np.zeros((num_bands,dim_mosaic[0], dim_mosaic[1]))

    arr_denominator = np.zeros(dim_mosaic)

    for i, path_rtc in enumerate(list_rtc):
        path_nlooks = list_nlooks[i]
        print(f'Mosaicking: {i+1} of {num_raster}')

        #calculate the burst RTC's offset wrt. the output mosaic in the image coordinate
        offset_imgx = int((list_geo_transform[i,0] - xmin_mosaic) / posting_x + 0.5)
        offset_imgy = int((list_geo_transform[i,3] - ymax_mosaic) / posting_y + 0.5)

        raster_rtc = gdal.Open(path_rtc,0)
        arr_rtc = raster_rtc.ReadAsArray()
        #Replace NaN values with 0
        arr_rtc[np.isnan(arr_rtc)] = 0.0

        raster_nlooks = gdal.Open(path_nlooks, 0)
        arr_nlooks = raster_nlooks.ReadAsArray()
        arr_nlooks[np.isnan(arr_nlooks)] = 0.0

        if num_bands == 1:
            arr_numerator[offset_imgy:offset_imgy+list_dimension[i, 0],
                          offset_imgx:offset_imgx+list_dimension[i, 1]] += arr_rtc * arr_nlooks
        else:
            for i_band in range(num_bands):
                arr_numerator[i_band,
                              offset_imgy:offset_imgy+list_dimension[i, 0],
                              offset_imgx:offset_imgx+list_dimension[i, 1]] += arr_rtc[i_band, :, :] * arr_nlooks

        arr_denominator[offset_imgy:offset_imgy + list_dimension[i, 0],
                        offset_imgx:offset_imgx + list_dimension[i, 1]] += arr_nlooks

        raster_rtc = None
        raster_nlooks = None

    if num_bands == 1:
        arr_out = arr_numerator/arr_denominator
    else:
        arr_out = np.zeros(arr_numerator.shape)
        for i_band in range(num_bands):
            arr_out[i_band, :, :]=arr_numerator[i_band, :, :] / arr_denominator


    # write out the array
    drv_out = gdal.GetDriverByName('Gtiff')
    raster_out = drv_out.Create('mosaic_out.tif', dim_mosaic[1], dim_mosaic[0], num_bands, gdal.GDT_Float32)
    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))

    raster_srs_src = gdal.Open(list_rtc[0], 0)
    raster_out.SetProjection(raster_srs_src.GetProjection())
    raster_srs_src = None

    if num_bands == 1:
        raster_out.GetRasterBand(1).WriteArray(arr_out)

    else:
        for i_band in range(num_bands):
            raster_out.GetRasterBand(i_band+1).WriteArray(arr_out[i_band])


#test code below. remove before commit!!!
if __name__=='__main__':
    PATH_HOME=os.getenv('HOME')
    PATH_RTC_BURST=PATH_HOME+'/opera-adt/scratch/COMPASS_RTC_Indonesia_WIP_v15/scratch_dir_all_bursts_with_correction_all_frame'
    list_rtc_vh=glob.glob(f'{PATH_RTC_BURST}/t??_??????_iw?_????????.tif')
    list_nlooks_vh=[rtc.replace('.tif','_VH_nlooks_temp_1660198055.814647.tif') for rtc in list_rtc_vh]

    weighted_mosaic(list_rtc_vh,list_nlooks_vh)
