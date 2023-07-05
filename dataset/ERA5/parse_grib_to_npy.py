# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

from glob import glob
import os
import re
import sys
import datetime
import pandas as pd
import numpy as np
import pygrib
from typing import Callable, Dict, List, Union, Optional
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt


def parse_grb_file(file_path: str, valid_date=None, columns: Optional[List] = None):
    """
    Parse single grib file using dict. For each stepRange, the data will be flattened into lists, one for each feature,
    and merge with the previous dict.

    Args:
        file_path: path to grib file.
        columns: the specific columns in raw grib file.

    Returns: dict of the form {feature_name: [data]}.
    """
    grb = pygrib.open(file_path)
    data = grb.read()
    res_dict = {}
    times = []

    for attr in tqdm(data):
        name = attr.name
        if not name in res_dict.keys():
            res_dict[name] = []

        attr_value = attr.values
        res_dict[name] += [attr_value]

        valid_dt = [attr.validDate]
        if attr.name == '2 metre temperature':
            times.extend(valid_dt)

    vs = []
    for k, v in res_dict.items():
        # print(k)
        res_dict[k] = np.array(v)
        vs += [v]
    vs = np.stack(vs, axis=-1)
    times = np.array(times)

    return vs, times, data


if __name__ == '__main__':
    grib_path = './raw/china_raw/'
    npy_path = './china/'

    files = os.listdir(grib_path)
    files = [i for i in files if re.match('.*.grib', i)]
    print('grib files to parse: {}'.format(files))
    for file in files:
        print('='*20)
        file_path = os.path.join(grib_path, file)
        print('processing grib file: {}'.format(file_path))
        data_npy, time_npy, data = parse_grb_file(file_path)

        new_npy_path = os.path.join(npy_path, file.split('.')[0] + '.npy')
        np.save(new_npy_path, data_npy)

        new_time_path = os.path.join(npy_path, 'time_' + file.split('.')[0] + '.npy')
        np.save(new_time_path, time_npy)
        print('npy file saved: {} with shape {}'.format(new_npy_path, data_npy.shape))

        a = 1