"""Setup configuration for DJI OSV Converter package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from __init__.py
init_file = Path(__file__).parent / "dji_osv_converter" / "__init__.py"
version = "0.1.0-dev"
for line in init_file.read_text().split('\n'):
    if line.startswith('__version__'):
        version = line.split('"')[1]
        break

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text()

setup(
    name="dji-osv-converter",
    version=version,
    description="Convert DJI Osmo 360 OSV files to equirectangular 360° MP4 for DaVinci Resolve",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DJI OSV Converter Contributors",
    url="https://github.com/zindello/djiclireframe",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "protobuf>=3.20.0",
        "av>=10.0.0",
        "numpy>=1.24.0",
        "opencv-contrib-python>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "dji-resolve=dji_osv_converter.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="dji osmo 360 video converter equirectangular davinci resolve",
    project_urls={
        "Bug Reports": "https://github.com/zindello/djiclireframe/issues",
        "Source": "https://github.com/zindello/djiclireframe",
    },
)
