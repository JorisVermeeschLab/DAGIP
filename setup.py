import os

from setuptools import setup


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = os.path.join(ROOT, 'dagip')

PROJECT_NAME = 'dagip'

packages = [
    f'{PROJECT_NAME}',
    f'{PROJECT_NAME}.benchmark',
    f'{PROJECT_NAME}.correction',
    f'{PROJECT_NAME}.da',
    f'{PROJECT_NAME}.ichorcna',
    f'{PROJECT_NAME}.nipt',
    f'{PROJECT_NAME}.nn',
    f'{PROJECT_NAME}.optimize',
    f'{PROJECT_NAME}.retraction',
    f'{PROJECT_NAME}.stats',
    f'{PROJECT_NAME}.tools',
    f'{PROJECT_NAME}.transport',
    f'{PROJECT_NAME}.validation'
]

setup(
    name=PROJECT_NAME,
    version='0.0.1',
    description='',
    url='https://github.com/AntoinePassemiers/DAGIP',
    author='Antoine Passemiers',
    packages=packages
)
