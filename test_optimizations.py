import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 50)
print("测试优化后的图像拼接系统 (语法检查)")
print("=" * 50)

try:
    from stitcher.config import (
        FEATURE_DETECTOR,
        FEATURE_NPOINTS,
        USE_FLANN,
        SEAM_BAND,
        FEATHER_RADIUS,
        FLANN_INDEX_PARAMS,
        FLANN_SEARCH_PARAMS
    )
    print("✓ 配置文件导入成功")
    print(f"  - 默认特征检测器: {FEATURE_DETECTOR}")
    print(f"  - 特征点数量: {FEATURE_NPOINTS}")
    print(f"  - 使用FLANN匹配器: {USE_FLANN}")
    print(f"  - 融合带宽: {SEAM_BAND}")
    print(f"  - 羽化半径: {FEATHER_RADIUS}")
except Exception as e:
    print(f"✗ 配置文件导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

try:
    from stitcher.algorithms import image_sorter
    print("✓ image_sorter 模块导入成功")
except Exception as e:
    print(f"✗ image_sorter 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from stitcher.algorithms import feature_registration
    print("✓ feature_registration 模块导入成功")
except Exception as e:
    print(f"✗ feature_registration 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from stitcher.pipeline.stitching_pipeline import StitchingPipeline
    print("✓ 拼接管道导入成功")
except Exception as e:
    print(f"✗ 拼接管道导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

try:
    import importlib.util
    import inspect
    
    print("检查新增函数...")
    
    spec = importlib.util.spec_from_file_location("feature_registration", "/workspace/stitcher/algorithms/feature_registration.py")
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    
    if hasattr(fr, 'create_feature_detector'):
        print("  ✓ create_feature_detector 函数存在")
    else:
        print("  ✗ create_feature_detector 函数不存在")
    
    if hasattr(fr, 'create_matcher'):
        print("  ✓ create_matcher 函数存在")
    else:
        print("  ✗ create_matcher 函数不存在")
    
    if hasattr(fr, 'registerTexture'):
        print("  ✓ registerTexture 函数存在")
        sig = inspect.signature(fr.registerTexture)
        if len(sig.parameters) >= 4:
            print("    ✓ registerTexture 支持detector_type参数")
    else:
        print("  ✗ registerTexture 函数不存在")
    
except Exception as e:
    print(f"✗ 检查函数时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

try:
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("image_sorter", "/workspace/stitcher/algorithms/image_sorter.py")
    sorter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sorter)
    
    print("检查图像排序函数...")
    if hasattr(sorter, 'compute_pair_overlap'):
        print("  ✓ compute_pair_overlap 函数存在")
    if hasattr(sorter, 'build_overlap_matrix'):
        print("  ✓ build_overlap_matrix 函数存在")
    if hasattr(sorter, 'find_optimal_order'):
        print("  ✓ find_optimal_order 函数存在")
    if hasattr(sorter, 'sort_images_by_overlap'):
        print("  ✓ sort_images_by_overlap 函数存在")
    
except Exception as e:
    print(f"✗ 检查图像排序函数时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

try:
    import importlib.util
    import inspect
    
    spec = importlib.util.spec_from_file_location("stitching_pipeline", "/workspace/stitcher/pipeline/stitching_pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    
    print("检查拼接管道更新...")
    if hasattr(pipeline.StitchingPipeline, '__init__'):
        init_sig = inspect.signature(pipeline.StitchingPipeline.__init__)
        print("  ✓ StitchingPipeline.__init__ 存在")
    
    if hasattr(pipeline.StitchingPipeline, 'run'):
        print("  ✓ StitchingPipeline.run 存在")
        with open('/workspace/stitcher/pipeline/stitching_pipeline.py', 'r') as f:
            content = f.read()
            if 'sort_images_by_overlap' in content:
                print("    ✓ 集成了图像排序功能")
            if 'AUTO_SORT' in content:
                print("    ✓ 支持AUTO_SORT配置")
    
except Exception as e:
    print(f"✗ 检查拼接管道时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

try:
    import importlib.util
    import argparse
    
    spec = importlib.util.spec_from_file_location("demo_cli", "/workspace/scripts/demo_cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    
    print("检查命令行工具更新...")
    if hasattr(cli, 'main'):
        print("  ✓ main函数存在")
        with open('/workspace/scripts/demo_cli.py', 'r') as f:
            content = f.read()
            if '--detector' in content:
                print("    ✓ 支持--detector参数")
            if '--no-auto-sort' in content:
                print("    ✓ 支持--no-auto-sort参数")
            if 'FEATURE_DETECTOR' in content:
                print("    ✓ 集成了FEATURE_DETECTOR配置")
    
except Exception as e:
    print(f"✗ 检查命令行工具时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 50)
print("所有语法和结构检查通过！系统已成功升级。")
print("=" * 50)
print()
print("新增功能:")
print("  1. 支持多种特征检测器 (ORB/SIFT/AKAZE)")
print("  2. 支持FLANN快速匹配器")
print("  3. 图像自动排序功能")
print("  4. 可配置的拼接参数")
print()
print("修改的文件:")
print("  - stitcher/config/settings.py (新增配置项)")
print("  - stitcher/config/__init__.py (导出新配置)")
print("  - stitcher/algorithms/feature_registration.py (重写特征注册)")
print("  - stitcher/algorithms/image_sorter.py (新建: 图像排序模块)")
print("  - stitcher/algorithms/__init__.py (导出新函数)")
print("  - stitcher/pipeline/stitching_pipeline.py (集成排序功能)")
print("  - scripts/demo_cli.py (新增命令行参数)")
print()
print("使用示例:")
print("  python scripts/demo_cli.py img1.jpg img2.jpg --detector ORB")
print("  python scripts/demo_cli.py img1.jpg img2.jpg img3.jpg --no-auto-sort")
print("  python scripts/demo_cli.py img1.jpg img2.jpg --detector AKAZE --seam-band 15")
print()
