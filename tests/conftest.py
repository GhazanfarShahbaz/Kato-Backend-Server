import sys

def pytest_configure():
    root_dir: str = "/Users/ghaz/Documents/projects/Kato-Backend-Server"
    sys.path.append(root_dir)
