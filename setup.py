from setuptools import setup, find_packages
import platform


assert platform.system() in ('Linux', ), f"{platform.system()} is not supported."


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='embodied_llm',
    packages=[package for package in find_packages()],
    version='0.0.1',
    license='MIT',
    description='A chatbot that speaks and sees',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Yann Bouteiller',
    url='https://github.com/yannbouteiller/embodied-LLM',
    download_url='',
    keywords=['LLM', 'embodied', 'vision', 'speech', 'ASR', 'TTS', 'STT', 'AI', 'chatbot'],
    install_requires=[
        'uvicorn',
        'starlette',
        'fastapi',
        'sse_starlette',
        'starlette_context',
        'pydantic_settings',
        'eclipse-zenoh',
        'opencv-python',
        'transformers',
        'llama-cpp-python',
        'piper-tts',
        # 'RealtimeTTS',
        # 'RealtimeSTT'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Linux',
        'Programming Language :: Python',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    extras_require={},
    scripts=[],
)
