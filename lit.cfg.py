import lit.formats

root_dir = os.path.dirname(__file__)
bin_dir = os.path.join(root_dir, "bin")

config.name = "Silicon"
config.test_format = lit.formats.ShTest()
config.suffixes = [".si", ".mlir"]
config.test_exec_root = os.path.join(root_dir, "build")
config.substitutions += [
    ("silc", os.path.join(bin_dir, "silc")),
    ("silicon-opt", os.path.join(bin_dir, "silicon-opt")),
]
python_path = root_dir
if env := os.environ.get("PYTHONPATH"):
  python_path += ":" + env
config.environment["PYTHONPATH"] = python_path
