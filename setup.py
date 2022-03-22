import os
from setuptools import setup, find_packages

path = os.path.dirname(os.path.realpath(__file__))
requirements_path = path + "/requirements.txt"

install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        lst = f.read().splitlines()

    for x in lst:
        install_requires.append(x)

setup(name="masterarbeit",
      packages=find_packages(),
      package_data={"": [".txt", "*.xlsx", "*.csv"]},
      install_requires = install_requires)
