'''
Compare two RTC products if they are equivalent
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


def get_list_dataset_attrs_keys(hdf_obj_1: h5py.Group,
                                key_in: str='/',
                                list_dataset_so_far: list=None,
                                list_attrs_so_far: list=None):

    '''
    Recursively traverse the Dataset and Attributes in the input HDF5 object.
    Returns the list of keys for the dataset and the attributes.

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
    _ : 0 if the two HDF are identical; 1 otherwise

    '''

    #default values for the lists
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


def compare_dataset_attr(hdf5_obj_1, hdf5_obj_2, str_key, is_attr=False):
    '''
    Compare the dataset or attribute defined by `str_key`
    NOTE: For attributes, the path and the key are separated by newline character ('\n')

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

    Return:
    -------
    _: True when the dataset are identical; False otherwise
    '''

    # Prepare to comapre the data in the HDF objects
    if is_attr:
        path_attr, key_attr = str_key.split('\n')
        val_1 = hdf5_obj_1[path_attr].attrs[key_attr]
        val_2 = hdf5_obj_2[path_attr].attrs[key_attr]
        if not isinstance(val_1,np.ndarray):
            val_1 = np.array(val_1)

        if not isinstance(val_2,np.ndarray):
            val_2 = np.array(val_2)

    else:
        val_1 = np.array(hdf5_obj_1[str_key])
        val_2 = np.array(hdf5_obj_2[str_key])

    shape_val_1 = np.array(val_1).shape
    shape_val_2 = np.array(val_2).shape

    if shape_val_1 != shape_val_2:
        # Dataset or attribute shape does not match
        return False

    if len(shape_val_1)==0 and len(shape_val_2)==0:
        # Scalar value
        if issubclass(val_1.dtype.type, np.number) and issubclass(val_2.dtype.type, np.number):
            return np.array_equal(val_1, val_2, equal_nan=True)
        return np.array_equal(val_1, val_2)

    if len(shape_val_1)==1 and len(shape_val_2)==1:
        # 1d vector

        # Dereference if val_1 and val_2 have HDF5 objstc reference.
        # Convert the 1d numpy array into list to differentiate the comparison process
        if 'shape' in dir(val_1[0]):
            if isinstance(val_1[0], np.void) or\
            ((len(val_1[0].shape) == 1) and (isinstance(val_1[0][0], h5py.h5r.Reference))):

                # Example:
                # attribute `REFERENCE_LIST` in
                # /science/CSAR/RTC/grids/frequencyA/xCoordinates'
                # attribute `DIMENSION_LIST` in
                # /science/CSAR/RTC/grids/frequencyA/VH
                list_val_1 = list(itertools.chain.from_iterable(val_1))
                val_1_new = [None] * len(list_val_1)
                for i_val, element_1 in enumerate(list_val_1):
                    if isinstance(element_1, h5py.h5r.Reference):
                        val_1_new[i_val] = hdf5_obj_1[element_1]
                    else:
                        val_1_new[i_val] = element_1
                val_1 = val_1_new

        # Repeat the same process for `val_2`
        if 'shape' in dir(val_2[0]):
            if isinstance(val_2[0], np.void) or\
            ((len(val_2[0].shape) == 1) and (isinstance(val_2[0][0], h5py.h5r.Reference))):

                list_val_2 = list(itertools.chain.from_iterable(val_2))
                val_2_new = [None] * len(list_val_2)
                for i_val, element_2 in enumerate(list_val_2):
                    if isinstance(element_2, h5py.h5r.Reference):
                        val_2_new[i_val] = hdf5_obj_2[element_2]
                    else:
                        val_2_new[i_val] = element_2

                val_2 = val_2_new

        if isinstance(val_1, list) and isinstance(val_2, list):
            # dereferenced val_1 and val_2
            if len(val_1) != len(val_2):
                # List shape does not match
                return False

            for id_element, element_1 in enumerate(val_1):
                element_2 = val_2[id_element]
                if element_1.shape != element_2.shape:
                    return False

                if not np.allclose(element_1,
                                   element_2,
                                   RTC_S1_PRODUCTS_ERROR_TOLERANCE,
                                   equal_nan=True):
                    return False

            # Went through all elements in the list,
            # and passed the closeness test in the for loop
            return True

        if issubclass(val_1.dtype.type, np.number) and issubclass(val_2.dtype.type, np.number):
            # val_1 and val_2 are numeric numpy array
            return np.array_equal(val_1, val_2, equal_nan=True)

        # All other cases, including the npy array with bytes
        return np.array_equal(val_1, val_2)

    if len(shape_val_1)>=2 and len(shape_val_2)>=2:
        return np.allclose(val_1,
                           val_2,
                           RTC_S1_PRODUCTS_ERROR_TOLERANCE,
                           equal_nan=True)

    # If the processing has reached here, that means
    # val_1 and val_2 cannot be compated due to their shape difference
    print('Detected an issue on the dataset shapes: ',
            f'Dataset key: {str_key}, '
            'dataset shape in the 1st HDF5: ', shape_val_1,
            'dataset shape in the 2nd HDF5: ', shape_val_2)
    return False


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

    with h5py.File(file_1,'r') as hdf5_in_1, h5py.File(file_2,'r') as hdf5_in_2:
        list_dataset_1, list_attrs_1 = get_list_dataset_attrs_keys(hdf5_in_1)
        set_dataset_1 = set(list_dataset_1)
        set_attrs_1 = set(list_attrs_1)

        list_dataset_2, list_attrs_2 = get_list_dataset_attrs_keys(hdf5_in_2)
        set_dataset_2 = set(list_dataset_2)
        set_attrs_2 = set(list_attrs_2)

        # Check the dataset
        union_set_dataset = set_dataset_1.union(set_dataset_2)
        if (len(union_set_dataset) == len(set_dataset_1) and
            len(union_set_dataset) == len(set_dataset_2)):
            print('\nDataset structure identical.')
            flag_identical_dataset_structure = True

        else:
            flag_identical_dataset_structure = False
            print('\nDataset structure not identical.')
            print('In the 1st HDF5, not in the 2nd data:')
            print('\n'.join(list(set_dataset_1 - set_dataset_2)))
            print('In the 2st HDF5, not in the 1nd data:')
            print('\n'.join(list(set_dataset_2 - set_dataset_1)))

        # Proceed with checking the values in dataset,
        # regardless of the agreement of their structure.

        list_flag_identical_dataset = [None] * len(union_set_dataset)
        for id_flag, key_dataset in enumerate(union_set_dataset):
            list_flag_identical_dataset[id_flag] = compare_dataset_attr(hdf5_in_1,
                                                                        hdf5_in_2,
                                                                        key_dataset,
                                                                        is_attr=False)
            if list_flag_identical_dataset[id_flag]:
                print(f'{id_flag+1:03d} / {len(union_set_dataset):03d} : '
                      f'PASSED. key: {key_dataset}')
            else:
                print(f'\033[91m{id_flag+1:03d} / {len(union_set_dataset):03d} : '
                      f'FAILED. key: {key_dataset}\033[00m')

        # Check the attribute
        union_set_attrs = set_attrs_1.union(set_attrs_2)

        if (len(union_set_attrs) == len(set_attrs_1) and
            len(union_set_attrs) == len(set_attrs_2)):
            flag_identical_attrs_structure = True
            print('\nAttribute structure identical.')

        else:
            flag_identical_attrs_structure = False
            print('\nAttribute structure not identical.')
            print('In the 1st HDF5, not in the 2nd data:')
            print('\n'.join(list(set_dataset_1 - set_dataset_2)))
            print('In the 2st HDF5, not in the 1nd data:')
            print('\n'.join(list(set_dataset_2 - set_dataset_1)))

        # Proceed with checking the values in dataset,
        # regardless of the agreement of their structure.

        list_flag_identical_attrs = [None] * len(union_set_attrs)
        for id_flag, key_attr in enumerate(union_set_attrs):
            list_flag_identical_attrs[id_flag] = compare_dataset_attr(hdf5_in_1,
                                                                      hdf5_in_2,
                                                                      key_attr,
                                                                      is_attr=True)
            str_printout = key_attr.replace('\n',' - ')
            if list_flag_identical_attrs[id_flag]:
                print(f'{id_flag+1:03d} / {len(union_set_attrs):03d} : '
                      f'PASSED. path - key: {str_printout}')
            else:
                print(f'\033[91m{id_flag+1:03d} / {len(union_set_attrs):03d} : '
                      f'FAILED. path - key: {str_printout}\033[00m')


        # Print out the test summary:
        print('\n\n****************** Test summary ******************')
        print(f'1st HDF FILE                 : {file_1}')
        print(f'2nd HDF FILE                 : {file_2}')
        print(f'Value tolerance              : {RTC_S1_PRODUCTS_ERROR_TOLERANCE}')
        print(f'\nIdentical dataset structure  : {flag_identical_dataset_structure}')
        print(f'Identical Attribute structure: {flag_identical_attrs_structure}\n')

        if all(list_flag_identical_dataset):
            print('All dataset passed the test')
        else:
            print('Dataset below did not pass the test:')
            for id_dataset, key_dataset in enumerate(union_set_dataset):
                if not list_flag_identical_dataset[id_dataset]:
                    print(key_dataset)

        if all(list_flag_identical_attrs):
            print('All attributes passed the test')
        else:
            print('\nAttributes below did not pass the test:')
            for id_attr, key_attr in enumerate(union_set_attrs):
                if not list_flag_identical_attrs[id_attr]:
                    token_key_attr=key_attr.split('\n')
                    print(f'{token_key_attr[1]} {token_key_attr[0]}')

if __name__ == '__main__':
    main()
