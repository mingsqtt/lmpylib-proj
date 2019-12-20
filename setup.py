import setuptools
import lmpylib.version

print("Building version: {}".format(lmpylib.version.__version__))

version = {}
with open("lmpylib/version.py") as fp:
    exec(fp.read(), version)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lmpylib-mingsqtt",
    version=version['__version__'],
    author="Li Ming",
    author_email="mingsqtt@hotmail.com",
    description="This is a personal library.",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/mingsqtt/lmpylib-proj",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)