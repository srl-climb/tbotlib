from setuptools import setup, find_packages

setup(
    name='tbotlib',
    version='0.1.0',
    author='Simon Harms',
    author_email='harms.simon759@mail.kyutech.jp',
    packages=find_packages(include=['tbotlib', 'tbotlib.*'])
)