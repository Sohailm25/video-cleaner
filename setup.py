from setuptools import setup, find_packages

setup(
    name="video-cleaner",
    version="0.1.0",
    description="Detect and redact PII from screen recordings",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8.0",
        "easyocr>=1.7.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "video-cleaner=video_cleaner.cli:main",
        ],
    },
)
