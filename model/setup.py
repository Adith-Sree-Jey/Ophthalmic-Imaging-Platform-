from setuptools import setup


setup(
    name="ophthalmic-imaging-model",
    version="0.1.0",
    packages=["model"],
    package_dir={"": ".."},
    install_requires=["timm==1.0.25"],
)
