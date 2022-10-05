from setuptools import find_packages, setup

setup(
    name='verax_pipeline',
    version='1.0',
    packages=find_packages(),

    author='Kevin Freeman',
    author_email='kevin.freeman@dsm.com',

    # project dependencies
    install_requires=["sklearn", "requests", "numpy<1.22,>=1.18", "xgboost==1.5",
        "pandas", "shap", "holoviews", "botocore", "boto3", "awswrangler",
        "nbconvert==5.6.1","ipython_genutils", "jinja2==3.0", "pandoc"]

)
