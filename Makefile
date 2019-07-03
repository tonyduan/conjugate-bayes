all:
	python3 -m readme2tex --branch master --nocdn --readme READMATH.md --output README.md
pkg:
	python3 setup.py sdist bdist_wheel
clean:
	rm -r build dist conjugate_bayes.egg-info
upload:
	twine upload dist/*


