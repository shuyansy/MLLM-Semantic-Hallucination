from setuptools import setup, find_packages

setup(
    name="qwen-transformers",
    version="0.1.0",
    description="Qwen-customized Transformers fork.",
    author="shuyansy",
    packages=find_packages(where="transformers"),
    package_dir={"": "transformers"},
    install_requires=[
        "torch>=1.13",
        "sentencepiece",
        "tqdm"
    ],
    python_requires=">=3.8",
)
