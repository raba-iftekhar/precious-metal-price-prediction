import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "flask",          # Web framework
    "pandas",         # Data manipulation
    "numpy",          # Numerical computations
    "matplotlib",     # Plotting
    "statsmodels",    # ARIMA time series models
    "scikit-learn",   # Random Forest and other ML models
    "tensorflow",     # LSTM deep learning
    "Jinja2",         # Templating (usually comes with Flask)
    "Werkzeug",       # Flask dependency for server utilities
]

for pkg in packages:
    install(pkg)
