from setuptools import setup


setup(
    name="ophthalmic-imaging-glaucoma",
    version="0.1.0",
    packages=["glaucoma", "glaucoma.src"],
    package_dir={"": ".."},
)
