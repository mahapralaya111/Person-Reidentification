"""
抗恶劣环境行人重识别系统 - 安装脚本

使用方式:
    pip install -e .          # 开发模式安装
    python setup.py install   # 标准安装
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh
                   if line.strip() and not line.startswith("#")
                   and not line.startswith("-")]

setup(
    name="person-reid-system",
    version="2.0.0",
    author="mahapralaya111",
    author_email="mahapralaya111@users.noreply.github.com",
    description="面向灾害应急搜救的抗恶劣环境行人重识别系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    
    packages=find_packages(),
    include_package_data=True,
    
    python_requires=">=3.7",
    install_requires=requirements,
    
    entry_points={
        "console_scripts": [
            "reid=main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    
    keywords="person-reid, reid, deep-learning, pytorch, computer-vision",
)
