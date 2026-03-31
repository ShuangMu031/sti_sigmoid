import sys
import ast
from pathlib import Path

print("=" * 50)
print("测试优化后的图像拼接系统 (语法检查)")
print("=" * 50)

files_to_check = [
    "stitcher/config/settings.py",
    "stitcher/config/__init__.py",
    "stitcher/algorithms/feature_registration.py",
    "stitcher/algorithms/image_sorter.py",
    "stitcher/algorithms/__init__.py",
    "stitcher/pipeline/stitching_pipeline.py",
    "scripts/demo_cli.py"
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
    print("所有语法检查通过！")
    print("=" * 50)
    print()
    print("新增功能:")
    print("  1. 支持多种特征检测器 (ORB/SIFT/AKAZE)")
    print("  2. 支持FLANN快速匹配器")
    print("  3. 图像自动排序功能")
    print("  4. 可配置的拼接参数")
    print()
    print("修改的文件:")
    for file_path in files_to_check:
        print(f"  - {file_path}")
    print()
    print("使用示例:")
    print("  python scripts/demo_cli.py img1.jpg img2.jpg --detector ORB")
    print("  python scripts/demo_cli.py img1.jpg img2.jpg img3.jpg --no-auto-sort")
    print("  python scripts/demo_cli.py img1.jpg img2.jpg --detector AKAZE --seam-band 15")
    print()
    sys.exit(0)
else:
    print("=" * 50)
    print("部分文件语法检查失败，请修复错误！")
    print("=" * 50)
    print()
    sys.exit(1)
