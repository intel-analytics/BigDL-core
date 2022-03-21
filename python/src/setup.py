#!/usr/bin/env python

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

bigdl_core_home = os.environ['BIGDL_CORE_DIR']
bigdl_core_jar = os.environ['BIGDL_CORE_JAR_PATH']
print(bigdl_core_jar)
VERSION = open(os.path.join(bigdl_core_home, 'python/version.txt'), 'r').read().strip()

idx = 0
for arg in sys.argv:
    if arg == "--plat-name":
        break
    else:
        idx += 1

if idx >= len(sys.argv):
    raise ValueError("Cannot find --plat-name argument. bigdl-core requires --plat-name to build.")

verbose_plat_name = sys.argv[idx + 1]

valid_plat_names = {"macosx_10_11_x86_64", "manylinux2010_x86_64"}
verbose_plat_names_to_plat_name = {"macosx_10_11_x86_64": "darwin-x86_64",
                                   "manylinux2010_x86_64": "linux-x86_64"}
if verbose_plat_name not in valid_plat_names:
    raise ValueError(f"--plat-name is not valid. --plat-name should be one of {valid_plat_names}"
                      f" but got {verbose_plat_name}")

plat_name = verbose_plat_names_to_plat_name[verbose_plat_name]

def setup_package():
    packages_name = "bigdl.share.core.lib"

    metadata = dict(
        name='bigdl-core',
        version=VERSION,
        description='BigDL Core Library for bigdl-dllib',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        include_package_data=True,
        packages=["bigdl.share.core.lib"],
        package_data={packages_name: [bigdl_core_jar]}
    )
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
