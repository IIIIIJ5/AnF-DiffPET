"""Experiment settings shared by train.py and test.py.
实验配置：train.py 与 test.py 共用。做消融实验时只需修改下面的模块开关。
"""

# ============================== 模块开关 / module switches ==============================
use_afg = True     # AFG：MedDINOv3 提取 CT 解剖先验，经两级 FCA 注入 U-Net 编码器
use_msctr = True   # MSCTR：用通道交叉 Transformer 联合重建多尺度跳连特征
use_fchm = True    # FCHM：频域对比难例挖掘损失，只在训练时使用

# ============================== 数据 / data ==============================
data_dir = 'data/lists'   # 存放 {train,test}_{hd,ld,ct}.txt 的文件夹
dataroot = ''             # 拼接在列表中每条路径前面的公共前缀
image_size = 128          # 切片统一缩放到 image_size x image_size

# ============================== 训练 / training ==============================
results_dir = 'results/anf_diffpet'
batch_size = 4
grad_accum = 2            # 梯度累积步数
lr = 2e-4
num_steps = 100000
ema_decay = 0.995
fchm_weight = 1.0         # lambda_FCHM
fchm_k = 2                # 难负样本个数 k
eval_every = 5000         # 每隔多少步在测试集上评估一次
save_last_every = 5000    # 每隔多少步更新 last.pt（用于断点续训）
save_every = 100000       # 每隔多少步另存一个 model-<step>.pt
num_workers = 4
seed = 10

# ============================== 推理 / inference ==============================
sampling_steps = 10       # DDIM 反向采样步数（论文设置：10 步，eta = 0），训练中评估与测试共用
test_batch_size = 16
