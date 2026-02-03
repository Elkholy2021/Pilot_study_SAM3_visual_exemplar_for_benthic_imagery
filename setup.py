from setuptools import setup, find_packages

setup(
    name="Evaluation_SAM3",
    version="0.1.0",
    description="Evaluation toolkit for SAM3 benthic imagery analysis",
    author="Hassan Elkholy",
    author_email="hassan.elkholy@ntnu.no",
    url="https://github.com/Elkholy2021/Evaluation_SAM3",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Add your dependencies here
        # "numpy>=1.20.0",
        # "torch>=1.9.0",
        # "opencv-python>=4.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)