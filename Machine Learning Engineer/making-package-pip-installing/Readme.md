# Convert modularized code into a Python package

Put code into a folder, e.g. "python_package" in the workspace. Inside the folder, you'll need to create a few folders and files:

* a `setup.py` file, which is required in order to use pip install
* a folder called 'distributions', which is the name of the Python package
* inside the 'distributions' folder, you'll need the `Gaussiandistribution.py` file, `Generaldistribution.py` and an `__init__.py` file.

Set up a virtual environment first. A virtual environment is a siloed Python installation apart from your main Python installation. That way you can easily delete the virtual environment without affecting your Python installation.

Open a new terminal window in the workspace by clicking 'NEW TERMINAL' and type:

* this command to create a virtual environment `python3 -m venv venv_name` where `venv_name` is the name you want to give to your virtual environment. You'll see a new folder appear with the Python installation named `venv_name`.
* In the terminal, type `source venv_name/bin/activate`. You'll notice that the command line now shows (`venv_name`) at the beginning of the line to indicate you are using the venv_name virtual environment.
* Now, you can type `pip install python_package/.` That should install your distributions Python package.
* Try using the package in a program to see if everything works!

Start the python interpreter from the terminal typing:

* `python3`

Then within the Python interpreter, you can use the distributions package:

* `from distributions import Gaussian`
* `gaussian_one = Gaussian(25, 2)`
* `gaussian_one.mean`
* `gaussian_one + gaussian_one`

etcetera... In other words, you can import and use the Gaussian class because the distributions package is now officially installed as part of your Python installation.
