from setuptools import setup, find_packages

setup(
    name='tbotlib',
    version='0.1.0',
    author='Simon Harms',
    author_email='harms.simon759@mail.kyutech.jp',
    packages=find_packages(include=['tbotlib', 'tbotlib.*']),
    install_requires=[
        "qpsolvers ==3.4.0",
        "numpy-quaternion ==3.5",
        "screeninfo ==0.8.1",
        "alphashape ==1.3.1",
        "open3d ==0.13.0",
        "numpy ==1.24",
        "networkx ==3.1",
        "matplotlib ==3.1.2"
    ]
)