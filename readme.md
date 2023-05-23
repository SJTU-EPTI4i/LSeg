# DRAC 2022 Task 1 Based on MT-UNet
操作：
- 数据集：`data/Miccai/A. Segmentation`
- 预处理：`cd data/Miccai && python make_nagetive_trainings.py`
- 创建数据集csv：`cd data/Miccai && python make_csv.py`
- 训练（子任务1/3）：`python train_mtunet_Miccai_1_3.py`
- 训练（子任务2）：`python train_mtunet_Miccai.py`
- 测试（子任务1/3）：`python validate_mtunet_Miccai.py --target 0`
- 测试（子任务2）：`python validate_mtunet_Miccai.py --target 1`
- 生成nii.gz：`python makenii.py`