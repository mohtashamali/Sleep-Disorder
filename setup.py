from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name="sleep_disorder_detection",
    version="1.0.0",
    author="Mohd Mohtasham Ali",
    author_email="your_email@example.com",
    description="A machine learning package for detecting sleep disorders using physiological and behavioral data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
    python_requires=">=3.8",
    license="MIT",
    url="https://github.com/yourusername/sleep_disorder_detection", 
)
