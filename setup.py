from setuptools import setup, find_packages

setup(
    name="crims2s",
    author="CRIM",
    author_email="david.landry@crim.ca",
    install_requires=["dask", "hydra-core"],
    entry_points={"console_scripts": ["s2s_mldataset = crims2s.mldataset:cli"]},
    packages=find_packages(exclude="test"),
    include_package_data=True,
)
