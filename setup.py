import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    line.strip()
    for line in open("requirements.txt", "r")
    if not line.strip().startswith("#")
]

setuptools.setup(
    name="Perspectival",
    version="0.1",
    author="Philipp Blandfort",
    description="Python-based toolkit for comparing transformers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blandfort/perspectival",
    packages=setuptools.find_packages(include=["perspectival", "perspectival.*"]),
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "pylint",
            "pre-commit",
            "pytest",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
