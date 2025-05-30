from setuptools import setup, find_packages

setup(
    name="transformers",               # 强制覆盖官方 transformers 包
    version="4.39.0.post0",           
    packages=find_packages(where="transformers"),
    package_dir={"": "transformers"},
)
