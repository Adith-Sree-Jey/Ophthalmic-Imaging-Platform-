from setuptools import setup


setup(
    name="ophthalmic-imaging-retina-segmentation",
    version="0.1.0",
    packages=["retina_segmentation", "retina_segmentation.src"],
    package_dir={"": ".."},
)
