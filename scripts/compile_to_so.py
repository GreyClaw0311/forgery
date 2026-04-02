#!/usr/bin/env python3
"""
Python 代码加密编译脚本 - 将 .py 编译为 .so 文件

用法:
    python compile_to_so.py

功能:
    1. 使用 Cython 将 release/ 目录下的 .py 文件编译成 .so 文件
    2. 保持原有目录结构
    3. 复制非 Python 文件 (模型、配置、shell脚本等)
    4. 输出到 release_so/ 目录
    5. 自动生成启动脚本

依赖:
    pip install cython

注意:
    - 需要在 Linux 系统上运行 (生成 .so 文件)
    - 需要 gcc 编译器
    - Python 版本需要与目标运行环境一致

作者: 灰 (上坤商业帝国首席CTO)
日期: 2026-04-02
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

# ==================== 配置 ====================
SOURCE_DIR = "release"
OUTPUT_DIR = "release_so"

# 需要编译的 Python 文件 (相对路径)
# 注意: __init__.py 文件通常只有导入语句，可以选择性编译
PY_FILES_TO_COMPILE = [
    "server_forgrey.py",
    "debug_service.py",
    "debug_response.py",
    "test_service.py",
    # "algorithms/__init__.py",  # 跳过编译，有复杂导入语句，创建简化版
    "algorithms/ela_detector.py",
    "algorithms/dct_detector.py",
    "algorithms/fusion_detector.py",
    "algorithms/ml_detector.py",
    "algorithms/features.py",
    # "utils/__init__.py",  # 跳过编译，只有注释
    "utils/postprocess.py",
]

# 需要创建简化版 __init__.py 的文件 (不编译，复制简化内容)
# 这些文件保留必要的导入/导出，但不包含完整代码
INIT_FILES_TO_COPY = {
    "__init__.py": '''"""
图像篡改检测服务 - Release 模块 (编译版)
"""

__version__ = "1.0.0"
''',
    "algorithms/__init__.py": '''"""
图像篡改检测算法模块 (编译版)
"""

# 编译版: 导入编译后的模块
from .ela_detector import ELADetector
from .dct_detector import DCTBlockDetector
from .fusion_detector import AdaptiveFusion
from .features import extract_all_features, PixelFeatureExtractor

# 兼容性别名
DCTDetector = DCTBlockDetector
FusionDetector = AdaptiveFusion

__all__ = [
    'ELADetector', 
    'DCTBlockDetector', 
    'DCTDetector',
    'AdaptiveFusion',
    'FusionDetector',
    'extract_all_features',
    'PixelFeatureExtractor'
]
''',
    "utils/__init__.py": '''"""
工具模块 (编译版)
"""

from .postprocess import PostProcessor, AdaptivePostProcessor, post_process

__all__ = ['PostProcessor', 'AdaptivePostProcessor', 'post_process']
''',
}

# 需要直接复制的文件 (不编译)
COPY_FILES = [
    "requirements.txt",
    "start.sh",
    "stop.sh",
]

# 需要复制的目录 (模型文件等，如果存在)
COPY_DIRS = [
    "models/gb_classifier",
    "models/pixel_segmentation",
]

# 模型目录 (即使为空也要创建)
MODEL_DIRS = [
    "models",
    "models/gb_classifier",
    "models/pixel_segmentation",
]

# ==================== 编译模板 ====================

# setup.py 模板 (用于编译单个模块)
SETUP_TEMPLATE = '''
from distutils.core import setup
from Cython.Build import cythonize
import numpy

module_name = "{module_name}"
source_file = "{source_file}"

setup(
    ext_modules=cythonize(
        source_file,
        compiler_directives={{
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }},
        include_path=[numpy.get_include()],
    ),
)
'''

# ==================== 工具函数 ====================

def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    
    # 检查 Cython
    try:
        import Cython
        print(f"✓ Cython 已安装: {Cython.__version__}")
    except ImportError:
        print("✗ Cython 未安装，请运行: pip install cython")
        sys.exit(1)
    
    # 检查 gcc
    result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ gcc 已安装")
    else:
        print("✗ gcc 未安装，请运行: apt-get install gcc")
        sys.exit(1)
    
    # 检查 numpy
    try:
        import numpy
        print(f"✓ numpy 已安装: {numpy.__version__}")
    except ImportError:
        print("✗ numpy 未安装，请运行: pip install numpy")
        sys.exit(1)
    
    # 检查源目录 (使用绝对路径)
    source_dir_abs = os.path.abspath(SOURCE_DIR)
    if not os.path.exists(source_dir_abs):
        print(f"✗ 源目录不存在: {source_dir_abs}")
        sys.exit(1)
    print(f"✓ 源目录存在: {source_dir_abs}")
    
    print()


def clean_output_dir():
    """清理输出目录"""
    # 使用绝对路径
    output_dir_abs = os.path.abspath(OUTPUT_DIR)
    
    if os.path.exists(output_dir_abs):
        print(f"清理输出目录: {output_dir_abs}")
        shutil.rmtree(output_dir_abs)
    os.makedirs(output_dir_abs)
    print(f"✓ 创建输出目录: {output_dir_abs}")
    print()


def compile_py_to_so(py_file: str, source_dir: str, output_dir: str) -> bool:
    """
    将单个 .py 文件编译为 .so 文件
    
    Args:
        py_file: Python 文件相对路径
        source_dir: 源目录
        output_dir: 输出目录
    
    Returns:
        是否成功
    """
    # 使用绝对路径
    source_dir_abs = os.path.abspath(source_dir)
    output_dir_abs = os.path.abspath(output_dir)
    source_path = os.path.join(source_dir_abs, py_file)
    
    if not os.path.exists(source_path):
        print(f"⚠ 源文件不存在: {source_path}")
        return False
    
    print(f"编译: {py_file}")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 关键修复: 将源文件复制到临时目录，保持相同的目录结构
        tmp_source_path = os.path.join(tmp_dir, py_file)
        tmp_source_dir = os.path.dirname(tmp_source_path)
        
        # 创建临时目录结构
        if tmp_source_dir and not os.path.exists(tmp_source_dir):
            os.makedirs(tmp_source_dir)
        
        # 复制源文件到临时目录
        shutil.copy2(source_path, tmp_source_path)
        
        # 生成 setup.py (使用临时目录中的源文件)
        module_name = Path(py_file).stem
        setup_content = SETUP_TEMPLATE.format(
            module_name=module_name,
            source_file=tmp_source_path  # 使用临时目录中的源文件
        )
        setup_path = os.path.join(tmp_dir, "setup.py")
        with open(setup_path, 'w') as f:
            f.write(setup_content)
        
        # 运行编译 (在临时目录执行 setup.py)
        result = subprocess.run(
            [sys.executable, setup_path, "build_ext", "--inplace"],
            cwd=tmp_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"✗ 编译失败: {py_file}")
            print(f"错误: {result.stderr}")
            return False
        
        # 关键修复: 在源文件所在目录查找 .so 文件 (不是临时目录根目录)
        so_search_dir = tmp_source_dir if tmp_source_dir else tmp_dir
        so_files = list(Path(so_search_dir).glob(f"{module_name}*.so"))
        
        if not so_files:
            # 如果在子目录没找到，尝试在整个临时目录搜索
            so_files = list(Path(tmp_dir).rglob(f"{module_name}*.so"))
        
        if not so_files:
            print(f"✗ 未找到编译输出: {module_name}.so")
            return False
        
        # 复制 .so 文件到输出目录 (保持目录结构)
        so_file = so_files[0]
        output_path = os.path.join(output_dir_abs, py_file.replace('.py', '.so'))
        output_subdir = os.path.dirname(output_path)
        
        if output_subdir and not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        shutil.copy2(str(so_file), output_path)
        print(f"✓ 输出: {output_path}")
        
        return True


def copy_init_files(output_dir: str):
    """
    复制简化版 __init__.py 文件
    
    这些文件不编译，直接复制简化内容（保留必要的导入）
    """
    print("\n创建简化版 __init__.py 文件...")
    
    output_dir_abs = os.path.abspath(output_dir)
    
    for init_file, content in INIT_FILES_TO_COPY.items():
        output_path = os.path.join(output_dir_abs, init_file)
        output_subdir = os.path.dirname(output_path)
        
        if output_subdir and not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"✓ 创建: {init_file}")


def copy_non_py_files(source_dir: str, output_dir: str):
    """复制非 Python 文件"""
    print("\n复制非 Python 文件...")
    
    # 使用绝对路径
    source_dir_abs = os.path.abspath(source_dir)
    output_dir_abs = os.path.abspath(output_dir)
    
    # 创建模型目录结构 (即使源目录为空)
    for model_dir in MODEL_DIRS:
        dst = os.path.join(output_dir_abs, model_dir)
        if not os.path.exists(dst):
            os.makedirs(dst)
            print(f"✓ 创建目录: {model_dir}")
    
    # 复制单个文件
    for file_path in COPY_FILES:
        src = os.path.join(source_dir_abs, file_path)
        dst = os.path.join(output_dir_abs, file_path)
        
        if os.path.exists(src):
            dst_dir = os.path.dirname(dst)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.copy2(src, dst)
            print(f"✓ 复制: {file_path}")
        else:
            print(f"⚠ 文件不存在: {file_path}")
    
    # 复制目录 (包括模型文件)
    for dir_path in COPY_DIRS:
        src = os.path.join(source_dir_abs, dir_path)
        dst = os.path.join(output_dir_abs, dir_path)
        
        # 目标目录已创建，复制文件内容
        if os.path.exists(src):
            # 复制目录中的文件
            for item in os.listdir(src):
                src_item = os.path.join(src, item)
                dst_item = os.path.join(dst, item)
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                    print(f"✓ 复制: {dir_path}/{item}")
                elif os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                    print(f"✓ 复制目录: {dir_path}/{item}")
        else:
            print(f"⚠ 目录不存在 (已创建空目录): {dir_path}")


def create_launcher_script(output_dir: str):
    """
    创建启动脚本
    
    编译后的 .so 文件可以通过 import 运行，
    但需要创建一个 Python 入口脚本来启动服务
    """
    print("\n创建启动脚本...")
    
    # 使用绝对路径
    output_dir_abs = os.path.abspath(output_dir)
    
    # 创建 Python 入口脚本
    launcher_content = '''#!/usr/bin/env python3
"""
图像篡改检测服务启动入口

编译版 (使用 .so 文件)
"""

import sys
import os

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入编译后的服务模块
from server_forgrey import main

if __name__ == '__main__':
    main()
'''
    
    launcher_path = os.path.join(output_dir_abs, "launcher.py")
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print(f"✓ 创建启动入口: launcher.py")
    
    # 更新 start.sh (使用 launcher.py 启动)
    start_sh_path = os.path.join(output_dir_abs, "start.sh")
    
    if os.path.exists(start_sh_path):
        with open(start_sh_path, 'r') as f:
            content = f.read()
        
        # 替换启动命令
        content = content.replace(
            'python3 server_forgrey.py',
            'python3 launcher.py'
        )
        
        with open(start_sh_path, 'w') as f:
            f.write(content)
        
        print(f"✓ 更新 start.sh")


def verify_compilation(output_dir: str) -> bool:
    """
    验证编译结果
    
    检查:
    1. 所有 .so 文件是否存在
    2. __init__.py 文件是否存在
    3. 模型文件是否复制
    4. 启动脚本是否正确
    """
    print("\n验证编译结果...")
    
    # 使用绝对路径
    output_dir_abs = os.path.abspath(output_dir)
    
    # 检查 .so 文件
    so_count = 0
    for py_file in PY_FILES_TO_COMPILE:
        so_file = py_file.replace('.py', '.so')
        so_path = os.path.join(output_dir_abs, so_file)
        
        if os.path.exists(so_path):
            so_count += 1
            print(f"✓ {so_file}")
        else:
            print(f"✗ {so_file} 不存在")
    
    print(f"\n编译文件统计: {so_count}/{len(PY_FILES_TO_COMPILE)}")
    
    # 检查 __init__.py 文件
    init_count = 0
    for init_file in INIT_FILES_TO_COPY.keys():
        init_path = os.path.join(output_dir_abs, init_file)
        
        if os.path.exists(init_path):
            init_count += 1
            print(f"✓ {init_file}")
        else:
            print(f"✗ {init_file} 不存在")
    
    print(f"\n__init__.py 文件统计: {init_count}/{len(INIT_FILES_TO_COPY)}")
    
    # 检查模型目录
    models_dir = os.path.join(output_dir_abs, "models")
    if os.path.exists(models_dir):
        print(f"✓ 模型目录存在")
    else:
        print(f"✗ 模型目录不存在")
        return False
    
    # 检查启动脚本
    launcher_path = os.path.join(output_dir_abs, "launcher.py")
    start_sh_path = os.path.join(output_dir_abs, "start.sh")
    
    if os.path.exists(launcher_path) and os.path.exists(start_sh_path):
        print(f"✓ 启动脚本存在")
    else:
        print(f"✗ 启动脚本不存在")
        return False
    
    return so_count == len(PY_FILES_TO_COMPILE) and init_count == len(INIT_FILES_TO_COPY)


def test_import(output_dir: str) -> bool:
    """
    测试导入编译后的模块
    
    验证 .so 文件是否可以正确导入
    """
    print("\n测试模块导入...")
    
    # 使用绝对路径
    output_dir_abs = os.path.abspath(output_dir)
    
    test_script = '''
import sys
import os
sys.path.insert(0, "{output_dir}")

try:
    # 测试导入主要模块
    import server_forgrey
    print("✓ server_forgrey 导入成功")
    
    # 测试导入算法模块
    from algorithms import ELADetector, DCTBlockDetector
    print("✓ algorithms 模块导入成功")
    
    # 测试导入工具模块
    from utils import postprocess
    print("✓ utils 模块导入成功")
    
    print("\\n所有模块导入测试通过!")
    
except ImportError as e:
    print(f"✗ 导入失败: {{e}}")
    sys.exit(1)
'''
    
    test_path = os.path.join(output_dir_abs, "test_import.py")
    with open(test_path, 'w') as f:
        f.write(test_script.format(output_dir=output_dir_abs))
    
    print(f"✓ 创建测试脚本: test_import.py")
    print("  请在目标服务器运行: python test_import.py")
    
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("Python 代码加密编译脚本")
    print("将 .py 编译为 .so 文件")
    print("=" * 60)
    print()
    
    # 1. 检查依赖
    check_dependencies()
    
    # 2. 清理输出目录
    clean_output_dir()
    
    # 获取绝对路径
    source_dir_abs = os.path.abspath(SOURCE_DIR)
    output_dir_abs = os.path.abspath(OUTPUT_DIR)
    
    print(f"源目录 (绝对路径): {source_dir_abs}")
    print(f"输出目录 (绝对路径): {output_dir_abs}")
    print()
    
    # 3. 编译 Python 文件
    print("开始编译 Python 文件...")
    print()
    
    success_count = 0
    for py_file in PY_FILES_TO_COMPILE:
        if compile_py_to_so(py_file, source_dir_abs, output_dir_abs):
            success_count += 1
    
    print()
    print(f"编译完成: {success_count}/{len(PY_FILES_TO_COMPILE)}")
    
    # 4. 创建简化版 __init__.py 文件
    copy_init_files(output_dir_abs)
    
    # 5. 复制非 Python 文件
    copy_non_py_files(source_dir_abs, output_dir_abs)
    
    # 6. 创建启动脚本
    create_launcher_script(output_dir_abs)
    
    # 7. 验证编译结果
    if verify_compilation(output_dir_abs):
        print("\n" + "=" * 60)
        print("✓ 编译成功!")
        print("=" * 60)
        print(f"\n输出目录: {output_dir_abs}")
        print("\n启动服务:")
        print(f"  cd {output_dir_abs}")
        print("  bash start.sh")
        print()
    else:
        print("\n" + "=" * 60)
        print("✗ 编译失败，请检查日志")
        print("=" * 60)
        sys.exit(1)
    
    # 8. 创建导入测试脚本
    test_import(output_dir_abs)


if __name__ == '__main__':
    main()