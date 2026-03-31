import sys
import ast
from pathlib import Path

print("=" * 50)
print("测试UI界面文件 (语法检查)")
print("=" * 50)

files_to_check = [
    "stitcher/ui/main_window.py",
    "stitcher/workers/stitching_worker.py"
]

all_good = True

for file_path in files_to_check:
    full_path = Path(__file__).parent / file_path
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source, filename=file_path)
        print(f"✓ {file_path} - 语法正确")
    except SyntaxError as e:
        print(f"✗ {file_path} - 语法错误: {e}")
        all_good = False
    except Exception as e:
        print(f"✗ {file_path} - 无法读取: {e}")
        all_good = False

print()

if all_good:
    print("=" * 50)
    print("所有UI文件语法检查通过！")
    print("=" * 50)
    print()
    print("新增UI功能:")
    print("  1. 特征检测器选择下拉菜单")
    print("  2. 自动排序开关")
    print("  3. 拼接顺序预览功能")
    print("  4. 详细的进度显示")
    print()
    print("使用方法:")
    print("  1. 选择图像后，点击'预览顺序'查看推荐的拼接顺序")
    print("  2. 在参数设置中选择合适的特征检测器")
    print("  3. 根据需要启用或禁用自动排序")
    print("  4. 点击'开始拼接'执行拼接")
    print()
    sys.exit(0)
else:
    print("=" * 50)
    print("部分UI文件语法检查失败，请修复错误！")
    print("=" * 50)
    print()
    sys.exit(1)
