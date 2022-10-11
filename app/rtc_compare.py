'''
Compare two RTC products (in HDF5) if they are equivalent.
Part of the codes are copied from PROTEUS SAS
'''

import argparse
import itertools

import h5py
import numpy as np

RTC_S1_PRODUCTS_ERROR_TOLERANCE = 1e-6

def _get_parser():
    parser = argparse.ArgumentParser(
        description='Compare two RTC products',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('input_file',
                        type=str,
                        nargs=2,
                        help='Input RTC products in NETCDF/HDF5 format')

    return parser


def _unpack_array(val_in, hdf5_obj_in):
    '''
    Unpack the array of array into ordinary numpy array.
    Convert an HDF5 object reference into the path it is pointing to.

    For internal use in this script.

    Parameter:
    -----------
    val_in: np.ndarray
        numpy array to unpack
    hdf5_obj_in:
        Source HDF5 object of `val_in`

    Return:
    val_out: np.ndarray
        unpacked array

    '''
    list_val_in = list(itertools.chain.from_iterable(val_in))

    list_val_out = [None] * len(list_val_in)
    for i_val, element_in in enumerate(list_val_in):
        if isinstance(element_in, h5py.h5r.Reference):
            list_val_out[i_val] = np.str_(hdf5_obj_in[element_in].name)
        else:
            list_val_out[i_val] = element_in
    val_out = np.array(list_val_out)

    return val_out


def print_data_difference(val_1, val_2, indent=4):
    '''
    Print out the difference of the data whose dimension is >= 1

    Parameters:
    -----------
    val_1, val_2: np.array
        Data that has difference to each other
    indent: int
        Number of spaces for indentation
    '''

    str_indent = ' ' * indent + '-'

    # printout the difference
    if not issubclass(val_1.dtype.type, np.number):
        # Routine for non-numeric array
        # Print out the first discrepancy in the array
        flag_discrepancy = (val_1 != val_2)
        index_first_discrepancy = np.where(flag_discrepancy)[0][0]
        print(f'{str_indent} The first discrepancy has detected from index '
              f'[{index_first_discrepancy}] : '
              f'1st=({val_1[index_first_discrepancy]}), '
              f'2nd=({val_2[index_first_discrepancy]})')
        return

    # Routine for numeric array
    difference_val = val_1 - val_2
    index_max_diff = np.nanargmax(np.abs(difference_val))
    print(f'{str_indent} Maximum difference detected from index '
          f'[{index_max_diff}]: '
          f'1st: ({val_1[index_max_diff]}), 2nd: ({val_2[index_max_diff]}) = '
          f'diff: ({difference_val[index_max_diff]})')

    if not (issubclass(val_1.dtype.type, np.floating) or
            issubclass(val_1.dtype.type, np.complexfloating)):
        return

    # Check pixel-by-pixel nan / non-nan difference
    mask_nan_val_1 = np.isnan(val_1)
    mask_nan_val_2 = np.isnan(val_2)

    mask_nan_discrepancy = np.logical_xor(mask_nan_val_1, mask_nan_val_2)

    if not np.any(mask_nan_discrepancy):
        return

    num_pixel_nan_discrepancy = mask_nan_discrepancy.sum()
    index_pixel_nan_discrepancy = np.where(mask_nan_discrepancy)
    print(f'{str_indent} Found {num_pixel_nan_discrepancy} '
        'NaN inconsistencies between input arrays. '
        'First index of the discrepancy: '
        f'[{index_pixel_nan_discrepancy[0][0]}]')
    print(f'{str_indent} val_1[{index_pixel_nan_discrepancy[0][0]}] = '
        f'{val_1[index_pixel_nan_discrepancy[0][0]]}')
    print(f'{str_indent} val_2[{index_pixel_nan_discrepancy[0][0]}] = '
        f'{val_2[index_pixel_nan_discrepancy[0][0]]}')

    # Operations to print out further info regarding the discrapancy
    num_nan_both = np.logical_and(mask_nan_val_1, mask_nan_val_2).sum()
    num_nan_val_1 = np.sum(mask_nan_val_1)
    num_nan_val_2 = np.sum(mask_nan_val_2)
    print(f'{str_indent} # NaNs on val_1 only: {num_nan_val_1 - num_nan_both}')
    print(f'{str_indent} # NaNs on val_2 only: {num_nan_val_2 - num_nan_both}')

    # A line of space for better readability of the log
    print('')


def get_list_dataset_attrs_keys(hdf_obj_1: h5py.Group,
                                key_in: str='/',
                                list_dataset_so_far: list=None,
                                list_attrs_so_far: list=None):

    '''
    Recursively traverse the datasets and attributes within the input HDF5 group.
    Returns the list of keys for datasets and attributes.

    NOTE:
    In case of attributes, the path and the attribute keys are
    separated by newline character ('\n')

    Parameters:
    ----------
    hdf_obj_1: h5py.Group
        HDF5 object to retrieve the dataset and the attribute list
    key_in: str
        path in the HDF5 object
    list_dataset_so_far: list
        list of the dataset keys that have found so far
    list_attrs_so_far: list
        list of the attribute path/keys that have found so far

    Return:
    -------
    list_dataset_so_far : list
        List of datasets keys found for given HDF5 group
    list_attrs_so_far : list
        List of attributes found for given HDF5 group.
        Each attribute is identified by its path and key (attribute name).

    '''

    # default values for the lists
    if list_dataset_so_far is None:
        list_dataset_so_far = []
    if list_attrs_so_far is None:
        list_attrs_so_far = []

    if isinstance(hdf_obj_1[key_in], h5py.Group):
        # Append the attributes keys if there are any
        for key_attr_1 in hdf_obj_1[key_in].attrs:
            list_attrs_so_far.append('\n'.join([key_in, key_attr_1]))

        for key_1, _ in hdf_obj_1[key_in].items():
            get_list_dataset_attrs_keys(hdf_obj_1, f'{key_in}/{key_1}',
                                        list_dataset_so_far,
                                        list_attrs_so_far)

    else:
        list_dataset_so_far.append(key_in)
        for key_attr_1 in hdf_obj_1[key_in].attrs:
            # Append the attributes keys if there are any
            list_attrs_so_far.append('\n'.join([key_in, key_attr_1]))
    return list_dataset_so_far, list_attrs_so_far


def compare_hdf5_elements(hdf5_obj_1, hdf5_obj_2, str_key, is_attr=False,
                          id_key=None, total_key=None, print_passed_element=True):
    '''
    Compare the dataset or attribute defined by `str_key`
    NOTE: For attributes, the path and the key are
    separated by newline character ('\n')

    Parameters:
    -----------
    hdf5_obj_1: h5py.Group
        The 1st HDF5 object to compare
    hdf5_obj_2: h5py.Group
        The 2nd HDF5 object to compare
    str_key: str
        Key to the dataset or attribute
    is_attr: bool
        Designate if `str_key` is for dataset or attribute
    id_key: int
        index of the key in the list. Optional for printout purpose.
    id_key: int
        total number of the list. Optional for printout purpose.
    print_passed_element: bool, default = True
        turn on / off the printout for the given test when it's successful.


    Return:
    -------
    _: True when the dataset / attribute are equivalent; False otherwise
    '''


    if id_key is None or total_key is None:
        str_order = ''
    else:
        str_order = f'{id_key+1} of {total_key}'

    # Prepare to comapre the data in the HDF objects
    if is_attr:
        # str_key is for attribute
        path_attr, key_attr = str_key.split('\n')
        val_1 = hdf5_obj_1[path_attr].attrs[key_attr]
        val_2 = hdf5_obj_2[path_attr].attrs[key_attr]

        str_message_data_location = f'Attribute {str_order}. path: {path_attr} ; key: {key_attr}'
        # Force the types of the values to np.ndarray to utulize numpy features
        if not isinstance(val_1,np.ndarray):
            val_1 = np.array(val_1)

        if not isinstance(val_2,np.ndarray):
            val_2 = np.array(val_2)

    else:
        # str_key is for dataset
        str_message_data_location = f'Dataset {str_order}: {str_key}'
        val_1 = np.array(hdf5_obj_1[str_key])
        val_2 = np.array(hdf5_obj_2[str_key])

    # convert object reference to the path to which it is pointing
    # Example:
    # attribute `REFERENCE_LIST` in
    # /science/CSAR/RTC/grids/frequencyA/xCoordinates'
    # attribute `DIMENSION_LIST` in
    # /science/CSAR/RTC/grids/frequencyA/VH
    if (len(val_1.shape) >= 1) and ('shape' in dir(val_1[0])):
        if (isinstance(val_1[0], np.void) or
        ((len(val_1[0].shape) == 1) and (isinstance(val_1[0][0], h5py.h5r.Reference)))):
            val_1 = _unpack_array(val_1, hdf5_obj_1)

    # Repeat the same process for val_2
    if (len(val_2.shape) >= 1) and ('shape' in dir(val_2[0])):
        if (isinstance(val_2[0], np.void) or
        ((len(val_2[0].shape) == 1) and (isinstance(val_2[0][0], h5py.h5r.Reference)))):
            val_2 = _unpack_array(val_2, hdf5_obj_2)

    shape_val_1 = val_1.shape
    shape_val_2 = val_2.shape

    if shape_val_1 != shape_val_2:
        # Dataset or attribute shape does not match
        print('\033[91mFAILED.\033[00m', str_message_data_location)
        print(f'    - Data shapes do not match. {shape_val_1} vs. {shape_val_2}\n')
        return False

    if val_1.dtype != val_2.dtype:
        print('\033[91mFAILED.\033[00m', str_message_data_location)
        print(f'    - Data types do not match. ({val_1.dtype}) vs. ({val_2.dtype})\n')
        return False

    if len(shape_val_1) == 0:
        # Scalar value
        if issubclass(val_1.dtype.type, np.number):
            # numerical array
            return_val = np.allclose(val_1,
                                     val_2,
                                     RTC_S1_PRODUCTS_ERROR_TOLERANCE,
                                     equal_nan=True)

            if return_val:
                if print_passed_element:
                    print('\033[32mPASSED.\033[00m', str_message_data_location)
            else:
                print('\033[91mFAILED.\033[00m', str_message_data_location)
                print( '    - numerical scalar. Failed to pass the test. '
                      f'Tolerance = {RTC_S1_PRODUCTS_ERROR_TOLERANCE}')
                print(f'    - 1st value: {val_1}')
                print(f'    - 2nd value: {val_2}\n')
            return return_val

        # Not a numerical array
        return_val = np.array_equal(val_1, val_2)
        if return_val:
            if print_passed_element:
                print('\033[32mPASSED.\033[00m', str_message_data_location)
        else:
            print('\033[91mFAILED.\033[00m', str_message_data_location)
            print( '    - non-numerical scalar. Failed to pass the test.')
            print(f'    - 1st value: {val_1}')
            print(f'    - 2nd value: {val_2}\n')
        return return_val

    if len(shape_val_1) == 1:
        # 1d vector

        if issubclass(val_1.dtype.type, np.number):
            # val_1 and val_2 are numeric numpy array
            return_val = np.allclose(val_1, val_2, RTC_S1_PRODUCTS_ERROR_TOLERANCE, equal_nan=True)

            if return_val:
                if print_passed_element:
                    print('\033[32mPASSED.\033[00m', str_message_data_location)
            else:
                print('\033[91mFAILED.\033[00m', str_message_data_location)
                print('    - Numerical 1D array. Failed to pass the test. '
                     f'Tolerance = {RTC_S1_PRODUCTS_ERROR_TOLERANCE}')
                print_data_difference(val_1, val_2)
            return return_val

        # All other non-numerical cases, including the npy array with bytes
        return_val = np.array_equal(val_1, val_2)
        if return_val:
            if print_passed_element:
                print('\033[32mPASSED.\033[00m', str_message_data_location)
        else:
            print('\033[91mFAILED.\033[00m', str_message_data_location)
            print('    non-numerical 1D array. Failed to pass the test.')
        return return_val


    if len(shape_val_1) >= 2:
        return_val = np.allclose(val_1,
                             val_2,
                             RTC_S1_PRODUCTS_ERROR_TOLERANCE,
                             equal_nan=True)
        if return_val:
            if print_passed_element:
                print('\033[32mPASSED.\033[00m', str_message_data_location)
        else:
            print('\033[91mFAILED.\033[00m', str_message_data_location)
            print(f'    {len(shape_val_1)}D raster array. Failed to pass the test. '
                  f'Tolerance = {RTC_S1_PRODUCTS_ERROR_TOLERANCE}')
            print_data_difference(val_1, val_2)
        return return_val

    # Unexpected failure to compare `val_1` and `val_2`
    raise ValueError(f'Failed to compare the element: {str_message_data_location}'
                     f'dataset shape in the 1st HDF5: {shape_val_1}'
                     f'dataset shape in the 2nd HDF5: {shape_val_2}')


def compare_rtc_hdf5_files(file_1, file_2):
    '''
    Compare the two RTC products (in HDF5) if they are equivalent
    within acceptable difference

    Parameters:
    -----------
    file_1, file_2: str
        Path to the RTC products (in HDF5)

    Return:
    -------
    _: bool
        `True` if the two products are equivalent; `False` otherwise

    '''

    with h5py.File(file_1,'r') as hdf5_in_1, h5py.File(file_2,'r') as hdf5_in_2:
        list_dataset_1, list_attrs_1 = get_list_dataset_attrs_keys(hdf5_in_1)
        set_dataset_1 = set(list_dataset_1)
        set_attrs_1 = set(list_attrs_1)

        list_dataset_2, list_attrs_2 = get_list_dataset_attrs_keys(hdf5_in_2)
        set_dataset_2 = set(list_dataset_2)
        set_attrs_2 = set(list_attrs_2)

        # Check the dataset
        print('Checking the dataset.')
        intersection_set_dataset = set_dataset_1.intersection(set_dataset_2)
        flag_identical_dataset_structure = \
            (len(intersection_set_dataset) == len(set_dataset_1) and
             len(intersection_set_dataset) == len(set_dataset_2))

        # Proceed with checking the values in dataset,
        # regardless of the agreement of their structure.
        list_flag_identical_dataset = [None] * len(intersection_set_dataset)
        for id_flag, key_dataset in enumerate(intersection_set_dataset):
            list_flag_identical_dataset[id_flag] = \
                compare_hdf5_elements(hdf5_in_1,
                                      hdf5_in_2,
                                      key_dataset,
                                      is_attr=False,
                                      id_key=id_flag,
                                      total_key=len(intersection_set_dataset),
                                      print_passed_element=True)

        print('\nChecking the attributes.')
        # Check the attribute
        intersection_set_attrs = set_attrs_1.intersection(set_attrs_2)

        flag_identical_attrs_structure = \
            (len(intersection_set_attrs) == len(set_attrs_1) and
             len(intersection_set_attrs) == len(set_attrs_2))

        # Proceed with checking the values in attributes,
        # regardless of the agreement of their structure.

        list_flag_identical_attrs = [None] * len(intersection_set_attrs)
        for id_flag, key_attr in enumerate(intersection_set_attrs):
            list_flag_identical_attrs[id_flag] = \
                compare_hdf5_elements(hdf5_in_1,
                                      hdf5_in_2,
                                      key_attr,
                                      is_attr=True,
                                      id_key=id_flag,
                                      total_key=len(intersection_set_attrs),
                                      print_passed_element=False)

        flag_same_dataset = all(list_flag_identical_dataset)
        flag_same_attributes = all(list_flag_identical_attrs)

        final_result = (flag_identical_dataset_structure and
                        flag_identical_attrs_structure and
                        flag_same_dataset and
                        flag_same_attributes)

        print('\n\nTest completed.')

        if flag_identical_dataset_structure:
            print('\n\033[32mPASSED\033[00m: Dataset structure identical.')
        else:
            print('\n\033[91mFAILED\033[00m: Dataset structure not identical.')
            print('In the 1st HDF5, not in the 2nd data:')
            print('\n'.join(list(set_dataset_1 - set_dataset_2)))
            print('In the 2st HDF5, not in the 1nd data:')
            print('\n'.join(list(set_dataset_2 - set_dataset_1)))

        if flag_identical_attrs_structure:
            print('\033[32mPASSED\033[00m: Attribute structure identical.')

        else:
            flag_identical_attrs_structure = False
            print('\n\033[91mFAILED\033[00m: '
                  'Attribute structure not identical.')
            print('In the 1st HDF5, not in the 2nd data:')
            print('\n'.join(list(set_dataset_1 - set_dataset_2)))
            print('In the 2nd HDF5, not in the 1st data:')
            print('\n'.join(list(set_dataset_2 - set_dataset_1)))

        if all(list_flag_identical_dataset):
            print( '\033[32mPASSED\033[00m: '
                   'The datasets of the two HDF files are the same within the'
                  f'threshold of {RTC_S1_PRODUCTS_ERROR_TOLERANCE}')
        else:
            print( '\033[91mFAILED\033[00m: '
                  f'{sum(~np.array(list_flag_identical_dataset))} datasets '
                  f'out of {len(intersection_set_dataset)}'
                   'did not pass the test.')

        if all(list_flag_identical_attrs):
            print( '\033[32mPASSED\033[00m: '
                   'The attributes of the two HDF files are the same within '
                  f'the threshold of {RTC_S1_PRODUCTS_ERROR_TOLERANCE}\n')
        else:
            print(f'\033[91mFAILED\033[00m: '
                  f' {sum(~np.array(list_flag_identical_attrs))} '
                  f'attributes out of {len(intersection_set_attrs)} '
                   'did not pass the test.\n')

        return final_result


def main():
    '''
    main function of the RTC product comparison script
    '''
    parser = _get_parser()

    args = parser.parse_args()

    file_1 = args.input_file[0]
    file_2 = args.input_file[1]

    compare_rtc_hdf5_files(file_1, file_2)


if __name__ == '__main__':
    main()
