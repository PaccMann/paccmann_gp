import io
import re

from setuptools import setup, find_packages

# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            requirement
            for requirement in (line.strip() for line in fp)
            if requirement
            and not requirement.startswith("#")
            and vcs.search(requirement)
        ]
except FileNotFoundError:
    # requires verbose flags
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []


match = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("paccmann_gp/__init__.py", encoding="utf_8_sig").read(),
)
if match is None:
    raise SystemExit("Version number not found.")
__version__ = match.group(1)

setup(
    name="paccmann_gp",
    version=__version__,
    author="PaccMann team",
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_data={"polyrxn": ["py.typed"]},
    install_requires=[
        "loguru",
        "torch>=1.0,<=1.12.1",
        "numpy>=1.16.5,<1.24.0",
        "scipy>=1.0.0,<=1.11.0",
        "pytoda",
        "paccmann_generator",
        "paccmann_chemistry",
        "scikit-optimize>=0.9,<0.11",
    ],
    extras_require={"vcs": VCS_REQUIREMENTS},
)
