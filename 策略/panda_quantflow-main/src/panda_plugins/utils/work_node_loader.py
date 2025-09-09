import os
import importlib.util
import sys
import pathlib
from common.logging.system_logger import logging, setup_logging

logger = logging.getLogger(__name__)

# 这里定义相对路径部分
INTERNAL_REL = "internal"
CUSTOM_REL = "custom"

def load_all_nodes():
    """
    Load all work nodes from both internal and custom folders.
    
    从内部和自定义文件夹加载所有工作节点。
    """
    current_file = pathlib.Path(__file__).resolve()
    plugins_root = current_file.parent.parent
    
    internal_folder = os.path.join(plugins_root, INTERNAL_REL)
    custom_folder = os.path.join(plugins_root, CUSTOM_REL)
    
    if not os.path.exists(internal_folder):
        os.makedirs(internal_folder, exist_ok=True)
    else:
        load_all_nodes_from_folder(internal_folder)
    if not os.path.exists(custom_folder):
        os.makedirs(custom_folder, exist_ok=True)
    else:
        load_all_nodes_from_folder(custom_folder)
    
    # 防止插件加载时，日志配置被覆盖
    setup_logging()

def load_all_nodes_from_folder(folder_path: str):
    """
    Load all work node modules from the folder and its subfolders.
    
    从文件目录及其所有子目录动态导入所有工作节点模块.
    """
    sys.path.append(folder_path)
    
    loaded_count = 0
    failed_count = 0

    for root, _, files in os.walk(folder_path):
        # Add current subdirectory to sys.path
        # 添加当前子目录到sys.path
        if root != folder_path:
            sys.path.append(root)
            
        for fname in files:
            if fname.endswith(".py") and not fname.startswith("__"):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, folder_path)
                modulename = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                
                if modulename in sys.modules:
                    continue

                spec = importlib.util.spec_from_file_location(modulename, fpath)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                        loaded_count += 1
                        logger.debug(f"Loaded {modulename} from {folder_path}/{fname}")
                    except Exception as e:
                        logger.error(f"Failed to load {modulename} from {folder_path}/{fname}: {e}")
                        failed_count += 1
    
    logger.debug(f"From {folder_path}: Loaded {loaded_count} nodes, Failed {failed_count} nodes")

# TODO @cgt 调试函数
def load_work_node_from_db(obj_id: str):
    """
    Load a work node module from database by object ID.
    
    从数据库通过对象ID加载工作节点模块。
    
    Args:
        obj_id: 数据库中模块的对象ID
        
    Returns:
        module: 加载成功的模块对象，失败时返回None
    """
    try:
        # 伪代码：从数据库读取Python代码字符串
        # python_code = db.get_module_code_by_id(obj_id)
        # 这里使用伪代码模拟从数据库读取
        python_code = """
# 这是从数据库读取的示例代码
def example_function():
    return "Hello from database module"

class ExampleNode:
    def __init__(self):
        self.name = "db_loaded_node"
    
    def process(self):
        return "Processing from database loaded module"
"""
        
        if not python_code:
            logger.error(f"No code found for object ID: {obj_id}")
            return None
            
        # 生成模块名称
        module_name = f"user_module_{obj_id}"
        
        # 检查模块是否已经加载
        if module_name in sys.modules:
            logger.debug(f"Module {module_name} already loaded, returning existing module")
            return sys.modules[module_name]
        
        # 创建模块规范 - 使用自定义加载器
        spec = importlib.util.spec_from_loader(
            module_name, 
            loader=None,  # 将手动处理加载过程
            origin=f"database://{obj_id}"  # 虚拟的来源标识
        )
        
        if not spec:
            logger.error(f"Failed to create module spec for {module_name}")
            return None
            
        # 创建模块对象
        module = importlib.util.module_from_spec(spec)
        
        # 设置模块基本属性
        module.__name__ = module_name
        module.__file__ = f"<database:{obj_id}>"
        module.__loader__ = spec.loader
        module.__package__ = None
        module.__spec__ = spec
        
        # 编译代码字符串
        try:
            code_object = compile(python_code, f"<database:{obj_id}>", 'exec')
        except SyntaxError as e:
            logger.error(f"Syntax error in module {module_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to compile module {module_name}: {e}")
            return None
            
        # 执行模块代码
        try:
            exec(code_object, module.__dict__)
            
            # 将模块添加到sys.modules缓存
            sys.modules[module_name] = module
            
            logger.debug(f"Successfully loaded module {module_name} from database (ID: {obj_id})")
            
            # 调整
            return module
            
        except Exception as e:
            logger.error(f"Failed to execute module {module_name}: {e}")
            # 清理可能的部分加载状态
            if module_name in sys.modules:
                del sys.modules[module_name]
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error loading module from database (ID: {obj_id}): {e}")
        return None

# TODO @cgt 调试函数
def unload_work_node_from_db(obj_id: str, force: bool = False):
    """
    Unload a work node module that was loaded from database.
    
    卸载从数据库加载的工作节点模块。
    
    Args:
        obj_id: 数据库中模块的对象ID
        force: 是否强制卸载，即使有其他引用
        
    Returns:
        bool: 卸载是否成功
    """
    try:
        module_name = f"user_module_{obj_id}"
        
        # 检查模块是否存在
        if module_name not in sys.modules:
            logger.warning(f"Module {module_name} not found in sys.modules")
            return False
        
        module = sys.modules[module_name]
        
        # 尝试调用模块的清理方法（如果存在）
        if hasattr(module, '__cleanup__'):
            try:
                module.__cleanup__()
                logger.debug(f"Called cleanup method for module {module_name}")
            except Exception as e:
                logger.warning(f"Error during cleanup of module {module_name}: {e}")
        
        # 获取模块的引用计数（仅在非强制模式下检查）
        if not force:
            import gc
            refs = gc.get_referrers(module)
            # 过滤掉sys.modules和当前函数局部变量的引用
            external_refs = [ref for ref in refs if ref is not sys.modules and ref is not locals()]
            if external_refs:
                logger.warning(f"Module {module_name} has {len(external_refs)} external references, use force=True to unload anyway")
                return False
        
        # 从sys.modules中删除模块
        del sys.modules[module_name]
        
        # 清理模块的属性（减少内存泄漏）
        if hasattr(module, '__dict__'):
            module.__dict__.clear()
        
        logger.debug(f"Successfully unloaded module {module_name} from database (ID: {obj_id})")
        return True
        
    except Exception as e:
        logger.error(f"Error unloading module from database (ID: {obj_id}): {e}")
        return False



if __name__ == "__main__":
    load_all_nodes()