from setuptools import setup, find_packages






setup(
    name='tupi-ai-realtime-OD',        # Name of your package
    version='0.1.0',                 # Initial release version
    author='Do Thanh Binh Nguyen - GEOCOUNTS Australia',              # Your name or organization
    author_email='nguyendotb2112@gmail.com',  # Your email
    description='A short description of your project',
    long_description=open('README.md').read(),  # Read the content of README.md
    long_description_content_type='text/markdown',  # Format of long description
    url='https://github.com/yourusername/your_project',  # URL for your project
    packages=find_packages(),        # Automatically find all packages and subpackages
    classifiers=[                   # Classifiers help users find your project
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: AGPL 3.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',         # Minimum Python version required
    install_requires=[               # List of dependencies
        "absl-py==2.1.0", "accelerate==1.1.1", "aiofiles==23.2.1", "aiohappyeyeballs==2.4.3", "aiohttp==3.10.10", "aiosignal==1.3.1", "altgraph==0.17.4", "annotated-types==0.7.0", "antlr4-python3-runtime==4.9.3", "anyio==4.6.2.post1", "asttokens==2.4.1", "astunparse==1.6.3", "attrs==24.2.0", "blinker==1.9.0", "certifi==2024.8.30", "cfgv==3.4.0", "charset-normalizer==3.4.0", "click==8.1.7", "colorama==0.4.6", "contourpy==1.3.0", "coverage==7.6.4", "cycler==0.12.1", "Cython==3.0.11", "cython_bbox==0.1.5", "decorator==5.1.1", "deep-sort-realtime==1.3.2", "defusedxml==0.7.1", "distlib==0.3.9", "docker-pycreds==0.4.0", "einops==0.8.0", "executing==2.1.0", "fastapi==0.115.4", "ffmpy==0.4.0", "filelock==3.16.1", "Flask==3.1.0", "flatbuffers==24.3.25", "fonttools==4.54.1", "frozenlist==1.5.0", "fsspec==2024.10.0", "gast==0.6.0", "gitdb==4.0.11", "GitPython==3.1.43", "google-pasta==0.2.0", "gradio==5.5.0", "gradio_client==1.4.2", "graphviz==0.20.3", "grpcio==1.67.1", "h11==0.14.0", "h5py==3.12.1", "httpcore==1.0.6", "httpx==0.27.2", "huggingface-hub==0.26.2", "hydra-core==1.3.2", "identify==2.6.2", "idna==3.10", "iniconfig==2.0.0", "ipython==8.29.0", "itsdangerous==2.2.0", "jedi==0.19.2", "Jinja2==3.1.4", "joblib==1.4.2", "keras==3.6.0", "kiwisolver==1.4.7", "labelImg==1.8.6", "lapx==0.5.11", "libclang==18.1.1", "lightning==2.4.0", "lightning-utilities==0.11.8", "loguru==0.7.2", "lxml==5.3.0", "Markdown==3.7", "markdown-it-py==3.0.0", "MarkupSafe==2.1.5", "matplotlib==3.9.2", "matplotlib-inline==0.1.7", "mdurl==0.1.2", "ml-dtypes==0.4.1", "mpmath==1.3.0", "multidict==6.1.0", "namex==0.0.8", "networkx==3.4.2", "nodeenv==1.9.1", "Nuitka==2.5.6", "numpy", "omegaconf==2.3.0", "opencv-python==4.10.0.84", "opt_einsum==3.4.0", "optree==0.13.0", "ordered-set==4.1.0", "orjson==3.10.11", "packaging==24.2", "pandas==2.2.3", "parso==0.8.4", "pefile==2023.2.7", "peft==0.13.2", "pillow==11.0.0", "platformdirs==4.3.6", "pluggy==1.5.0", "pre_commit==4.0.1", "prompt_toolkit==3.0.48", "propcache==0.2.0", "protobuf==5.28.3", "psutil==6.1.0", "pure_eval==0.2.3", "py-cpuinfo==9.0.0", "pycocotools==2.0.8", "pydantic==2.9.2", "pydantic_core==2.23.4", "pydub==0.25.1", "Pygments==2.18.0", "pyinstaller==6.11.1", "pyinstaller-hooks-contrib==2024.10", "pyparsing==3.2.0", "PyQt5==5.15.11", "PyQt5-Qt5==5.15.2", "PyQt5_sip==12.15.0", "pytest==8.3.3", "pytest-cov==6.0.0", "python-dateutil==2.9.0.post0", "python-multipart==0.0.12", "pytorch-lightning==2.4.0", "pytz==2024.2", "pywin32-ctypes==0.2.3", "PyYAML==6.0.2", "regex==2024.11.6", "requests==2.32.3", "rich==13.9.4", "ruff==0.7.3", "safehttpx==0.1.1", "safetensors==0.4.5", "scikit-learn==1.5.2", "scipy==1.14.1", "seaborn==0.13.2", "semantic-version==2.10.0", "sentry-sdk==2.18.0", "setproctitle==1.3.3", "shapely==2.0.6", "shellingham==1.5.4", "six==1.16.0", "smmap==5.0.1", "sniffio==1.3.1", "stack-data==0.6.3", "starlette==0.41.2", "supervision==0.25.0", "sympy==1.13.1", "tensorboard==2.18.0", "tensorboard-data-server==0.7.2", "tensorflow==2.18.0", "termcolor==2.5.0", "thop==0.1.1.post2209072238", "threadpoolctl==3.5.0", "timm==1.0.11", "tokenizers==0.20.3", "tomlkit==0.12.0", "torchmetrics==1.5.2", "tqdm==4.67.0", "traitlets==5.14.3", "transformers==4.46.2", "typer==0.13.0", "typing_extensions==4.12.2", "tzdata==2024.2", "ultralytics==8.3.32", "ultralytics-thop==2.0.11", "urllib3==2.2.3", "uvicorn==0.32.0", "virtualenv==20.27.1", "wandb==0.18.6", "wcwidth==0.2.13", "websockets==12.0", "Werkzeug==3.1.3", "win32-setctime==1.1.0", "wrapt==1.16.0", "yarl==1.17.1", "zstandard==0.23.0"
    ],
    entry_points={                   # Define entry points (e.g., CLI commands)
        'console_scripts': [
            'infer_live=infer_realtime:main_function',
        ],
    },
)
