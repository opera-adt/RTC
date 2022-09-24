'''
Compare two RTC products if they are equivalent
Part of the codes are copied from PROTEUS SAS
'''

import argparse

import h5py
import numpy as np

RTC_S1_PRODUCTS_ERROR_TOLERANCE = 1e-6

def _get_parser():
    parser = argparse.ArgumentParser(
        description='Compare two DSWx-HLS products',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('input_file',
                        type=str,
                        nargs=2,
                        help='Input RTC products in NETCDF/HDF5 format')

    return parser

def compare_hdf5(hdf_obj_1, hdf_obj_2, key_in='/', compare_attr=True):
    '''
    Recursively traverse the Dataset and Attributes in the 1st HDF5 (the reference).
    Try to find the same dataset in the 2nd HDF5 (the target).
    Returns 0 if the reference and the target HDF5 are identical within threshold
    Prints out the difference report when they are not.

    Paremeters:
    ----------
    hdf_obj_1: Reference HDF5 object
    hdf_obj_2: Target HDF5 object

    Return:
    -------
    _ : 0 if the two HDF are identical; 1 otherwise

    '''

    if isinstance(hdf_obj_1[key_in], h5py.Group):
        for key_1, _ in hdf_obj_1[key_in].items():
            if key_1 in hdf_obj_2[key_in].keys():
                compare_hdf5(hdf_obj_1, hdf_obj_2, f'{key_in}/{key_1}')
            else:
                print(f'Cannot find {key_in}/{key_1} in the target object.')
                return 1
    else:
        # Compare the dataset
        # Detect the kind of data based on the shape
        if len(hdf_obj_1[key_in].shape)==0:
            print('Scalar value, shape:',hdf_obj_1[key_in].shape)
            # TODO Use np.array_equal() to compare

        elif len(hdf_obj_1[key_in].shape)==1:
            print('1d vector, shape:',hdf_obj_1[key_in].shape)
            # TODO Use np.array_equal() to compare

        elif len(hdf_obj_1[key_in].shape)<=3:
            print('Single or multiband raster. shape:',hdf_obj_1[key_in].shape)
            # TODO: Use compare_raster_dataset() to compare

        else:
            print('Dimmension of the reference data is not supported: ',
                 f'{key_in}/{key_1}, shape:',
                 hdf_obj_1[key_in].shape)
            # TODO: Use compare_raster_dataset() to compare

        # Compare the attribute
        if compare_attr:
            for key_attr_1 in hdf_obj_1[key_in].attrs:
                if key_attr_1 in hdf_obj_2[key_in].attrs:
                    print(f'Checking attribute: {key_attr_1} in {key_in}')
                    # TODO check the attribute value
                else:
                    print(f'ERROR: Cannot find attribute: \'{key_attr_1}\' in '
                          f'\'{key_in}\' in the target HDF object.')
                    return 1

        print('\n')



def compare_raster_dataset(raster_1: np.ndarray, raster_2: np.ndarray,
                           tolerance: float=RTC_S1_PRODUCTS_ERROR_TOLERANCE):
    '''
    Check if the two raster arrays are equivalent
    - Same size
    - all Pixel values are within the tolerance element-wise

    Parameters:
    -----------
    raster_1, raster_2: np.ndarray
    Input rasters to compare

    tolerance: float
    Accepted tolerance between the two raster


    '''


    return True  # placeholder


def main():
    '''
    Main function of the RTC comparison script
    - Compares the two HDF files by setting one of them as reference
    - Perform the same comparison abobe by settint the other HDF5 as reference
    '''
    parser = _get_parser()

    args = parser.parse_args()

    file_1 = args.input_file[0]
    file_2 = args.input_file[1]

    hdf5_in_1 = h5py.File(file_1,'r')
    hdf5_in_2 = h5py.File(file_2,'r')

    compare_hdf5(hdf5_in_1, hdf5_in_2)
    compare_hdf5(hdf5_in_2, hdf5_in_1)

    hdf5_in_1.close()
    hdf5_in_2.close()


if __name__ == '__main__':
    main()
