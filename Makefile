install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
format:
	black *.py
lint:
	pylint *.py
#test:
#	python -m pytest 
all: install format