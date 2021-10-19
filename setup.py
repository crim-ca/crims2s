from setuptools import setup, find_packages

setup(
    name="crims2s",
    author="CRIM",
    author_email="david.landry@crim.ca",
    install_requires=[
        "dask",
        "hydra-core",
        "pytorch-lightning==1.4.*",
        "hydra-submitit-launcher",
    ],
    entry_points={
        "console_scripts": [
            "s2s_infer = crims2s.training.infer:cli",
            "s2s_mldataset = crims2s.mldataset:cli",
            "s2s_train = crims2s.training.train:cli",
            "s2s_split_plev = crims2s.splitter.splitter:cli",
        ]
    },
    packages=find_packages(exclude="test"),
    include_package_data=True,
)
