import mugrade
import sys
import os
import pytest
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run mugrade')
    parser.add_argument("operation", nargs=1, choices = ["submit", "publish"],
        help = "operation: 'submit' or 'publish'")
    parser.add_argument("key", nargs=1, help = "Your mugrade grader key")
    args, pytest_args = parser.parse_known_args()
    os.environ["MUGRADE_KEY"] = args.key[0]
    os.environ["MUGRADE_OP"] = args.operation[0]
    print(os.environ["MUGRADE_OP"])
    pytest.main(pytest_args + ["-s", "-o", "python_functions='submit_*'"], plugins=[mugrade])
    return 0

if __name__ == "__main__":
    sys.exit(main())

