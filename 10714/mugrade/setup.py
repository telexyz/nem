from setuptools import setup, find_packages

setup(
    name="mugrade",
    version="1.2",
    author="Zico Kolter",
    author_email="zkolter@cs.cmu.edu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mugrade = mugrade.__main__:main"
        ]
    },
    description="Interface library for minimalist autograding site",
    python_requires=">=3.5",
    url="http://github.com/locuslab/mugrade",
    install_requires=["numpy >= 1.15", "pytest", "argparse >= 1.1", "requests"],
    zip_safe=False,
)

