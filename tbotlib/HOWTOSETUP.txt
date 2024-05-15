How to install python packages

1) Switch into the folder where the setup.py file of the packages is located
2) In terminal: 
	pip install -e .
	conda develop .
   (Do not forget the dot and activating the conda environment)

For uninstalling
1) Switch into the folder where the setup.py file of the packages is located
2) In terminal: 
	pip uninstall <name>
	conda develop -u .

   
Details: https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
