import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="loretta",
    version="0.1.0",
    author="Yifan Yang, Jiajun Zhou, Ngai Wong, Zheng Zhang",
    author_email="yifanycc@gmail.com",
    description="Official implementation for the LoRETTA adapters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yifanycc/loretta",
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers>=4.0.0',
        'torch>=2.0.0'
    ],
    python_requires='>=3.6',
)