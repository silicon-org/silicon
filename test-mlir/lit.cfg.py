import lit.formats
from lit.llvm import llvm_config

config.name = "Silicon"
config.test_format = lit.formats.ShTest()
config.suffixes = [".si", ".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.silicon_binary_dir, 'test')

llvm_config.add_tool_substitutions(["silicon-opt"], [config.silicon_tools_dir])
