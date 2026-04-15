#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容性测试脚本

此脚本用于测试打包后的图像自动拼接系统在不同环境下的兼容性，
检查应用程序是否能正常启动、加载资源和执行基本功能。
"""

import os
import sys
import subprocess
import time
import shutil
from datetime import datetime

# 配置参数
APP_NAME = '图像自动拼接系统—sigmoid.exe'
APP_DIR = os.path.join('.', 'dist', 'img_stitcher')
APP_PATH = os.path.join(APP_DIR, APP_NAME)
TEST_DIR = os.path.join('.', 'compatibility_test')
LOG_FILE = os.path.join(TEST_DIR, 'compatibility_test.log')
TEST_DURATION = 10  # 应用程序启动测试持续时间（秒）

# 创建测试目录和日志文件
def setup_test_environment():
    """设置测试环境"""
    print("正在设置测试环境...")
    
    # 创建测试目录
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    
    # 初始化日志文件
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"兼容性测试日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
    
    log_message(f"测试环境已设置在 {TEST_DIR}")

# 记录日志
def log_message(message, print_to_console=True):
    """记录消息到日志文件"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry + "\n")
    
    if print_to_console:
        print(log_entry)

# 检查应用程序文件是否存在
def check_application_files():
    """检查应用程序及其相关文件是否存在"""
    log_message("开始检查应用程序文件...")
    
    # 检查应用程序可执行文件
    if not os.path.exists(APP_PATH):
        log_message(f"错误: 找不到应用程序文件 {APP_PATH}")
        return False
    else:
        log_message(f"找到应用程序文件: {APP_PATH}")
        # 获取文件信息
        file_size = os.path.getsize(APP_PATH) / (1024 * 1024)  # 转换为MB
        log_message(f"应用程序文件大小: {file_size:.2f} MB")
    
    # 检查图标文件夹
    icon_dir = os.path.join(APP_DIR, 'ico')
    if os.path.exists(icon_dir):
        log_message(f"找到图标文件夹: {icon_dir}")
        # 检查图标文件
        cat_ico = os.path.join(icon_dir, 'cat.ico')
        if os.path.exists(cat_ico):
            log_message("找到主图标文件: cat.ico")
        else:
            log_message("警告: 找不到cat.ico图标文件")
    else:
        log_message(f"警告: 找不到图标文件夹 {icon_dir}")
    
    # 检查示例图片文件夹
    imgs_dir = os.path.join(APP_DIR, 'Imgs')
    if os.path.exists(imgs_dir):
        log_message(f"找到示例图片文件夹: {imgs_dir}")
        # 列出示例图片
        img_files = [f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))]
        log_message(f"找到 {len(img_files)} 个示例图片文件")
        if len(img_files) > 0:
            log_message(f"示例图片列表: {', '.join(img_files[:5])}{'...' if len(img_files) > 5 else ''}")
    else:
        log_message(f"警告: 找不到示例图片文件夹 {imgs_dir}")
    
    # 检查主要DLL文件是否存在
    required_dlls = ['opencv_world', 'numpy', 'scipy', 'python']
    dll_found = False
    
    for file in os.listdir(APP_DIR):
        if file.endswith('.dll'):
            dll_found = True
            for req_dll in required_dlls:
                if req_dll.lower() in file.lower():
                    log_message(f"找到依赖DLL: {file}")
    
    if not dll_found:
        log_message("警告: 没有找到DLL文件")
    
    log_message("应用程序文件检查完成")
    return True

# 测试应用程序启动
def test_application_startup():
    """测试应用程序是否能正常启动"""
    log_message("开始测试应用程序启动...")
    
    try:
        # 使用subprocess启动应用程序
        log_message(f"尝试启动应用程序: {APP_PATH}")
        process = subprocess.Popen(
            [APP_PATH],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待一段时间让应用程序启动
        log_message(f"等待 {TEST_DURATION} 秒让应用程序启动...")
        time.sleep(TEST_DURATION)
        
        # 检查进程是否仍在运行
        if process.poll() is None:
            log_message("应用程序启动成功，进程仍在运行")
            # 正常终止进程
            process.terminate()
            process.wait(timeout=5)
            log_message("应用程序已正常关闭")
            return True
        else:
            exit_code = process.returncode
            log_message(f"错误: 应用程序启动后立即退出，退出码: {exit_code}")
            
            # 尝试获取输出信息
            stdout, stderr = process.communicate(timeout=5)
            if stdout:
                log_message(f"标准输出:\n{stdout[:500]}..." if len(stdout) > 500 else f"标准输出:\n{stdout}")
            if stderr:
                log_message(f"错误输出:\n{stderr[:500]}..." if len(stderr) > 500 else f"错误输出:\n{stderr}")
            return False
            
    except Exception as e:
        log_message(f"错误: 启动应用程序时发生异常: {str(e)}")
        return False

# 测试系统兼容性
def test_system_compatibility():
    """测试系统兼容性"""
    log_message("开始测试系统兼容性...")
    
    # 获取系统信息
    import platform
    system_info = {
        "系统": platform.system(),
        "版本": platform.version(),
        "架构": platform.architecture(),
        "处理器": platform.processor(),
        "Python版本": platform.python_version(),
        "Windows版本": platform.win32_ver() if platform.system() == "Windows" else "不适用"
    }
    
    # 记录系统信息
    log_message("系统信息:")
    for key, value in system_info.items():
        log_message(f"  {key}: {value}", print_to_console=False)
    
    # 检查环境变量
    path_env = os.environ.get("PATH", "")
    log_message(f"PATH环境变量包含 {len(path_env.split(';'))} 个路径")
    
    # 检查Microsoft Visual C++ Redistributable是否存在（Windows特有）
    if platform.system() == "Windows":
        log_message("检查Microsoft Visual C++ Redistributable...")
        # 简单的检查方式 - 检查常见的DLL路径
        vc_redist_paths = [
            os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "System32", "vcruntime140.dll"),
            os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "System32", "msvcp140.dll")
        ]
        
        for dll_path in vc_redist_paths:
            if os.path.exists(dll_path):
                log_message(f"找到Visual C++ DLL: {os.path.basename(dll_path)}")
            else:
                log_message(f"未找到Visual C++ DLL: {os.path.basename(dll_path)}")
    
    log_message("系统兼容性测试完成")

# 检查Python依赖项
def check_python_dependencies():
    """检查Python依赖项"""
    log_message("开始检查Python依赖项...")
    
    # 检查关键依赖库是否可用
    required_libraries = [
        ('tkinter', 'tkinter'),
        ('PIL', 'PIL'),
        ('numpy', 'numpy'),
        ('cv2', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy')
    ]
    
    for import_name, display_name in required_libraries:
        try:
            __import__(import_name)
            log_message(f"依赖库可用: {display_name}")
        except ImportError:
            log_message(f"警告: 依赖库不可用: {display_name}")
        except Exception as e:
            log_message(f"警告: 检查依赖库 {display_name} 时出错: {str(e)}")
    
    log_message("Python依赖项检查完成")

# 创建测试报告
def generate_test_report(results):
    """生成测试报告"""
    log_message("\n生成兼容性测试报告...")
    
    report = "\n" + "="*60 + "\n"
    report += "图像自动拼接系统兼容性测试报告\n"
    report += "="*60 + "\n"
    report += f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # 汇总测试结果
    report += "测试结果:\n"
    all_passed = True
    
    for test_name, passed in results.items():
        status = "通过" if passed else "失败"
        report += f"- {test_name}: {status}\n"
        if not passed:
            all_passed = False
    
    # 总体评估
    report += "\n总体评估: "
    if all_passed:
        report += "所有测试通过，应用程序兼容性良好"
    else:
        report += "部分测试失败，请检查日志获取详细信息"
    
    report += "\n\n详细信息请查看完整日志文件"
    
    # 记录报告到日志文件
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(report)
    
    # 输出报告到控制台
    print(report)
    
    # 创建单独的报告文件
    report_file = os.path.join(TEST_DIR, 'compatibility_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    log_message(f"测试报告已保存到 {report_file}")

# 主函数
def main():
    """主函数"""
    print("图像自动拼接系统兼容性测试")
    print("="*60)
    
    # 设置测试环境
    setup_test_environment()
    
    # 测试结果字典
    results = {}
    
    # 执行各项测试
    results["检查应用程序文件"] = check_application_files()
    
    # 只有当应用程序文件检查通过后才测试启动
    if results["检查应用程序文件"]:
        results["应用程序启动测试"] = test_application_startup()
    else:
        results["应用程序启动测试"] = False
    
    # 系统兼容性测试总是执行
    test_system_compatibility()
    check_python_dependencies()
    
    # 生成测试报告
    generate_test_report(results)
    
    print(f"\n兼容性测试完成，请查看日志文件了解详细信息: {LOG_FILE}")

if __name__ == "__main__":
    main()
