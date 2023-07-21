# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import oss2
import os
from datetime import datetime, timedelta
from glob import glob
import re
import pandas as pd

today = datetime.strftime(datetime.now(), '%Y%m%d')
today = datetime.now()
print("today is: ")
print(today)


print('====== oss transfer ec ==========')
access_key_upload = 'LTAI5tBvR8AYWhVaaT_yourkey'
access_secret_upload = 'SC7b7M3Zea8oFNsTs0bk5q_yourpw'
end_point_upload = 'https://oss-cn-beijing.aliyuncs.com'
bucket_name_upload = 'libra-ec-data'

auth = oss2.Auth(access_key_upload, access_secret_upload)
bucket = oss2.Bucket(auth, end_point_upload, bucket_name_upload)

path = './processed/'

for i, obj in enumerate(oss2.ObjectIteratorV2(bucket, prefix='/Earthformer/dataset/moving_mnist/processed/')):
    f_name = obj.key.split('/')[-1]
    try:
        bucket.get_object_to_file(obj.key, path + '/{}'.format(f_name))
        print(f_name)
        print('='*20)
    except:
        print('wrong ', f_name)
        print('='*20)
