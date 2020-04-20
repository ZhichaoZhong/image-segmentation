try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='seglib',
    version='0.0.1',
    url='https://github.com/ZhichaoZhong/image-segmentation',
    author='ZhichaoZhong',
    author_email='zzhong@wehkamp.nl',
    description='Segmentation toolbox',
    packages=["seglib"],
    python_requires='>=3.4',
)