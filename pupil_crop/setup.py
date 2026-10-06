from setuptools import setup


setup(
    name="ophthalmic-imaging-pupil-crop",
    version="0.1.0",
    packages=[
        "pupil_crop",
        "pupil_crop.Yolo",
        "pupil_crop.detector",
        "pupil_crop.processor",
        "pupil_crop.utils",
    ],
    package_dir={"": ".."},
)
