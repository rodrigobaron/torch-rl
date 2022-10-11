import re

from setuptools import find_namespace_packages, find_packages, setup


setup(
    name="torch-rl",
    version='0.0.1',
    description="PyTorch RL re-implementations",
    author="Rodrigo Baron",
    author_email="baron.rodrigo0@gmail.com",
    url="https://github.com/rodrigobaron/torch-rl",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    data_files=[('', ['README.md', 'LICENSE'])],
    include_package_data=True,
    python_requires='>=3.7',
)