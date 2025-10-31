from setuptools import find_packages, setup

setup(
    name="robust-xai-medimg",
    version="0.1.0",
    description="Adversarially robust and concept-grounded XAI for medical imaging",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
