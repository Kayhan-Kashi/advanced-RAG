from setuptools import setup, find_packages

setup(
    name="shared",
    version="1.0.0",
    author="Your Team",
    description="Shared common library for microservices",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "confluent-kafka>=2.3.0",
        "pydantic>=2.5.0",
    ],
)