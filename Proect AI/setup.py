from setuptools import setup, find_packages

setup(
    name="informer_trading",
    version="0.1.0",
    description="Скальпинг ETH/USDT на OKX с использованием нейросетей",
    author="Informer AI",
    packages=find_packages(),
    install_requires=[
        "ccxt",
        "numpy",
        "pandas",
        "pyyaml",
        "websockets",
        "aiohttp",
        "asyncio",
        "python-telegram-bot",
        "fastapi",
        "uvicorn",
        "scikit-learn",
        "torch",
        "sqlalchemy",
        "matplotlib",
        "dash"
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 