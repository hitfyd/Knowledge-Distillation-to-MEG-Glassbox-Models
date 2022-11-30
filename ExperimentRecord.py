import csv
import os
import sys
import time

import numpy


def get_project_path():
    """得到项目路径"""
    project_path = os.path.dirname(__file__)
    return project_path


def create_dir(dir_path):
    os.makedirs(os.path.dirname(dir_path), exist_ok=True)  # 确保路径存在


# 定义模型存储路径和运行记录存储路径
# checkpoint_dir = get_project_path() + '/checkpoint/'
# record_dir = get_project_path() + '/record/'
# create_dir(checkpoint_dir)
# create_dir(record_dir)


class ExperimentRecord(object):
    def __init__(self, mark=''):
        self.mark = mark
        # 获取当前执行的py文件绝对路径
        self.run_py_path = os.path.abspath(sys.argv[0])
        # 提取当前执行的py文件名称
        self.run_py_name = os.path.split(self.run_py_path)[-1].split(".")[0]
        self.time = time.strftime("%Y%m%d%H%M%S")

        if self.mark == '':
            self.record_path = '{}{}/{}.csv'.format(record_dir, self.run_py_name, self.time)
        else:
            self.record_path = '{}{}/{}_{}.csv'.format(record_dir, self.run_py_name, self.mark, self.time)
        self.record_name = self.record_path[:-4]
        # 确保记录目录存在
        os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
        print(self.run_py_path, self.record_path)

    def append(self, row_record):
        assert isinstance(row_record, str) or isinstance(row_record, list) \
               or isinstance(row_record, numpy.ndarray) or isinstance(row_record, dict)
        # 打开记录文件并追加记录
        with open(self.record_path, 'a', newline='', encoding='utf-8') as record:
            writer = csv.writer(record)
            if isinstance(row_record, numpy.ndarray):
                writer.writerow(row_record.squeeze().tolist())
            elif isinstance(row_record, dict):
                writer.writerow(list(row_record.keys()))
                writer.writerow(list(row_record.values()))
            else:
                writer.writerow(row_record)
        print(row_record)
