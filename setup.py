from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="catastrophe",
    version="1.0.0",
    author="maximizemaxwell",
    description="Autoencoder-based code vulnerability detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/catastrophe",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "huggingface-hub>=0.20.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "catastrophe-train=catastrphe.train:train",
            "catastrophe-predict=catastrphe.predict:main",
        ],
    },
)
