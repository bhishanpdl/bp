# Notes
- Make sure are all files inside `tests/` ends with `_test.py`.
  I have used pre-commit hook with config `name-tests-test`
  so it needs test files ending with required format.
- To run pre-commit
  + first install module [pre-commit](https://pre-commit.com/) `pip install pre-commit`
  + create a .pre-commit-config.yaml file in root of repo
  + create .git/hooks/pre-commit executable file `pre-commit install`
  + run pre-commit `pre-commit run --all-files`



# To run unit tests
```bash
# go to the conda environment where the moduel bp is installed,
# then run unittest

python -m unittest test_bp.py

# aliter
nosetests
```

# Best Practices
While writing test cases, make test function names `test_` plus the name of the functions.
We can use various unit test assertions.

- [Python unit test official examples](https://docs.python.org/3/library/unittest.html)
- [Pandas official utility functions](https://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html)
