from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="portfolio-management-system",
    version="0.1.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="An automated system for managing a diversified investment portfolio with dynamic rebalancing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/portfolio-management-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    py_modules=["main"],
    entry_points={
        "console_scripts": [
            "portfolio-manager=main:main",
        ],
    },
)