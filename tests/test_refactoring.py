import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("测试重构后的图像拼接系统 (导入检查)")
print("=" * 60)

all_passed = True

def test_import(module_name, description):
    global all_passed
    try:
        __import__(module_name)
        print(f"✓ {description} - 导入成功")
        return True
    except Exception as e:
        print(f"✗ {description} - 导入失败: {str(e)}")
        all_passed = False
        return False

test_import('stitcher.config', '配置模块')
test_import('stitcher.config.settings', '配置文件')
test_import('stitcher.algorithms', '算法模块')
test_import('stitcher.algorithms.feature_registration', '特征配准')
test_import('stitcher.algorithms.seam_graphcut', '图割接缝')
test_import('stitcher.algorithms.local_poisson_blend', '泊松融合')
test_import('stitcher.pipeline', '流程模块')
test_import('stitcher.pipeline.stitching_pipeline', '拼接流程')

print("\n" + "=" * 60)
if all_passed:
    print("所有模块导入检查通过！")
else:
    print("部分模块导入检查失败")
print("=" * 60)

print("\n重构总结:")
print("1. 配置文件已更新，新增了 GC_ROI_MARGIN, GC_OBJECT_THRESH, AKAZE_THRESHOLD 等参数")
print("2. seam_graphcut.py 已重构，返回 select_img1_mask 而不是 label_map")
print("3. stitching_pipeline.py 已重构，拆分为多个阶段函数")
print("4. local_poisson_blend.py 已优化，使用数组索引，支持矩阵分解复用")
print("5. feature_registration.py 已重构，拆分为多个小函数")

sys.exit(0 if all_passed else 1)
