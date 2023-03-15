from pycuda.compiler import SourceModule

def load_from_file(file):
    with open(file) as f:
        source = f.read()
        return SourceModule(source)
    