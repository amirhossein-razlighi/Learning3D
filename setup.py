from setuptools import setup, find_packages
import re

# Read version from __init__.py
with open("torch3dr/__init__.py", "r") as f:
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

setup(
    name="torch3dr",
    version=version,
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.2",
        "tqdm>=4.64.0",
        "pillow~=11.1.0",
        "imageio~=2.37.0",
        "click~=8.1.8",
        "pytorch3d",
    ],
    author="AmirHossein Razlighi",
    author_email="arazlighi@gmail.com",
    description="A PyTorch-based library for 3D Vision Research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/amirhossein-razlighi/Torch3DR",
    project_urls={
        "Bug Tracker": "https://github.com/amirhossein-razlighi/Torch3DR/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
