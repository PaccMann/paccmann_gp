import io
import re

from setuptools import setup, find_packages
from pkg_resources import parse_requirements

# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []
print(VCS_REQUIREMENTS)


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
    install_requires=["loguru", "torch", "numpy", "scipy", "pytoda", "paccmann_generator", "paccmann_chemistry", "scikit-optimize"],
    extras_require={"vcs": VCS_REQUIREMENTS}
)
