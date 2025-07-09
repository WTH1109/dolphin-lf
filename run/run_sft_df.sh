set -x

if [ ! -d "/mnt/nlp-ali" ]; then
    mkdir -p /mnt/nlp-ali
    cd /mnt/nlp-ali
    ln -s /mnt/ali-sh-1/dataset/redone usr
    ln -s /mnt/ali-sh-1/dataset/redone nas
    cd /mnt/nlp-ali/usr
fi

nvidia-smi topo -m  # 查看 GPU 间连接方式（NVLink/PCIe）
ifconfig           # 确认主网络接口（如 eth0/ib0)
ibstat

ori_checkpoint_name="llama-3.1-70b-instruct"
ori_checkpoint_path="/mnt/ali-sh-1/dataset/redone/hc/model/ll"
ds_zero=3
cutoff_len=10240
step=1
lr=2e-5
packing=false
neat_packing=false
logging_steps=10
save_steps=3000
eval_steps=3000
per_device_train_batch_size=1
gradient_accumulation_steps=4
save_total_limit=3
num_train_epochs=2.0
warmup_ratio=0.1
val_size=0.001
per_device_eval_batch_size=1
datetime=$(date +"%Y%m%d%H%M%S")
run_name="V19"
tokenized="V19"


dataset="llava_sharegpt4o,llava_textocr(gpt4v),llava_Geometry3K,llava_chartqa,llava_scienceqa,18_medical_talk,laolao77/MMDU,benchmark_qa_03,benchmark_qa_04_detection,benchmark_qa_09_detection,benchmark_qa_10,benchmark_qa_10_report,benchmark_qa_13_detection_new,benchmark_qa_16_detection_new,benchmark_qa_17_detection,benchmark_qa_18_detection_new,benchmark_qa_18_measurement,benchmark_qa_18_new,benchmark_qa_21,benchmark_qa_23_detection_new,benchmark_qa_23_new,benchmark_qa_25,benchmark_qa_27_measurement,benchmark_qa_28_class_new2,benchmark_qa_28_new2,benchmark_qa_31_detection,benchmark_qa_31_measurement,benchmark_qa_32,benchmark_qa_32_detection,benchmark_qa_37,benchmark_qa_37_detection,benchmark_qa_38_detection,benchmark_qa_40,benchmark_qa_42,benchmark_qa_44,benchmark_qa_47_detection_new,benchmark_qa_49_detection,benchmark_qa_50_detection_new,benchmark_qa_50_measurement,benchmark_qa_50_new,benchmark_qa_52_detection_new,benchmark_qa_53,benchmark_qa_53_detection,benchmark_qa_64_detection,benchmark_qa_66,benchmark_qa_67_detection,benchmark_qa_69_new,benchmark_qa_70,benchmark_qa_74_normal,benchmark_qa_74_visible,benchmark_qa_75,benchmark_qa_99_locattion,benchmark_qa_99_section,WeThink_Multimodal_Reasoning_120K,teaching-material-01_illustration-v2,teaching-material-02-illustration-v2,teaching-material-02-illustration-v3,teaching-material-03-illustration-v3,teaching-material-03_illustration-v2,teaching-material-04-case-v3,teaching-material-04_case-v1,teaching-material-04_illustration-v1,teaching-material-05_illustration-v1,teaching-material-05_illustration-v2,teaching-material-06_illustration-v1,teaching-material-07-illustration-v3,teaching-material-07_illustration-v1,teaching-material-07_illustration-v2,teaching-material-08_illustration-v1,teaching-material-09_illustration-v1,teaching-material-10-illustration-v3,teaching-material-10_illustration-v1,teaching-material-10_illustration-v2,teaching-material-11-illustration-v3,teaching-material-11_illustration-v2,teaching-material-12-illustration-v3,teaching-material-12_illustration-v2,teaching-material-14-anatomical-v1,benchmark_qa_03_cn,benchmark_qa_10_cn,benchmark_qa_10_report_cn,benchmark_qa_18_cn,benchmark_qa_18_measurement_cn,benchmark_qa_21_cn,benchmark_qa_23_cn,benchmark_qa_23_detection_cn,benchmark_qa_25_cn,benchmark_qa_27_measurement_cn,benchmark_qa_28_class_cn,benchmark_qa_28_cn,benchmark_qa_31_measurement_cn,benchmark_qa_32_cn,benchmark_qa_40_cn,benchmark_qa_42_cn,benchmark_qa_44_cn,benchmark_qa_50_cn,benchmark_qa_50_detection_cn,benchmark_qa_50_measurement_cn,benchmark_qa_53_cn,benchmark_qa_53_detection_cn,benchmark_qa_64_detection_cn,benchmark_qa_66_cn,benchmark_qa_67_detection_cn,benchmark_qa_69_cn_new,benchmark_qa_70_cn,benchmark_qa_74_normal_cn,benchmark_qa_74_visible_cn,benchmark_qa_75_cn,benchmark_qa_99_location_cn,Pascal_01_new_part_0,Pascal_01_new_part_1,Pascal_01_new_part_2,Pascal_01_new_part_3,Pascal_02_new_part_0,Pascal_02_new_part_1,Pascal_02_new_part_2,Pascal_02_new_part_3,Pascal_03_new_part_0,Pascal_03_new_part_1,Pascal_03_new_part_2,Pascal_03_new_part_3,benchmark_qa_04_detection_s,benchmark_qa_09_detection_s,benchmark_qa_10_report_s,benchmark_qa_18_s,benchmark_qa_21_s,benchmark_qa_23_s,benchmark_qa_27_measurement_s_new,benchmark_qa_31_measurement_s_new,benchmark_qa_32_s,benchmark_qa_38_detection_s,benchmark_qa_40_s,benchmark_qa_50_s,benchmark_qa_52_detection_s,benchmark_qa_53_s,inc_data_79_detection_,inc_data_79_detection__cn,inc_data_82_classification_,inc_data_82_classification__cn,inc_data_83_classification_,inc_data_83_classification__cn,inc_data_83_detection_,inc_data_83_detection__cn,inc_data_84_classification_,inc_data_84_classification__cn,inc_data_84_detection_,inc_data_84_detection__cn,inc_data_86_classification_,inc_data_86_classification__cn,inc_data_86_detection_,inc_data_86_detection__cn,inc_data_87_classification_,inc_data_87_classification__cn,inc_data_87_detection_,inc_data_87_detection__cn,inc_data_92_classification_,inc_data_92_classification__cn,inc_data_92_detection_,inc_data_92_detection__cn,inc_data_92is_healthy_classification_,inc_data_92is_healthy_classification__cn,inc_data_93_detection_,inc_data_93_detection__cn,benchmark_qa_18_measurement_cn_s,benchmark_qa_23_cn_s,benchmark_qa_25_cn_s,benchmark_qa_27_measurement_cn_s,benchmark_qa_28_cn_s,benchmark_qa_31_measurement_cn_s,benchmark_qa_32_cn_s,benchmark_qa_32_detection_cn_s,benchmark_qa_42_cn_s,benchmark_qa_44_cn_s,benchmark_qa_49_detection_cn_s,benchmark_qa_50_cn_s,benchmark_qa_50_measurement_cn_s,benchmark_qa_53_cn_s,benchmark_qa_66_cn_s,benchmark_qa_70_cn_s,benchmark_qa_74_normal_cn_s,cp500/ultrasound-samples,Ka4on/ultrasound_train,identity,autoif_4omini,alpaca_gpt4_zh,ruozhiba_gpt4,DolphinAI/SelfCognition,r1_distillation_v1_new,r1_distillation_v2_new,r1_distillation_v3,r1_distillation_zh_v1,Chinese-DeepSeek-R1-Distill-data-110k,medical-o1-reasoning-SFT_en,medical-o1-reasoning-SFT_zh,sharegpt-zh-en_new,teaching-material-00-report-v1,teaching-material-01_scan-qa-v2,teaching-material-15-ab-v1"
echo $dataset

output_dir=/mnt/ali-sh-1/dataset/redone/hc/model/llamatrain/checkpoints/sft/${run_name}_${ori_checkpoint_name}_sft_do_${datetime}

# 判断output_dir路径是否存在且是目录
#if [ -d "$output_dir" ]; then
#    echo "警告：output_dir $output_dir 已存在！！！"
#    exit 1
#    cd ~ || exit 1
#else
#    # 若不存在 output_dir，则切换到当前路径下的 output 目录
#    cd /mnt/nlp-ali/usr/zhaofei5/projects/unimodel_sft/sft/LLaMA-Factory
#fi

source /mnt/nlp-ali/usr/envs/anaconda3/etc/profile.d/conda.sh

conda activate /mnt/nlp-ali/usr/envs/anaconda3/envs/llamafactorydo


export WANDB_API_KEY="86887cdf03cfcbc5040f4c80e957ec20e42f4c5f"

echo $datetime
echo $(date +"%Y%m%d%H%M%S")
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_P2P_LEVEL=5  # 设置点对点通信级别
# export NCCL_P2P_LEVEL=nvl     # 优先使用 NVLink
# export NCCL_IB_DISABLE=0      # 默认值，启用 InfiniBand（或明确设置=0）
# export NCCL_SOCKET_IFNAME=ib0
# export WANDB_SOCKET=eth0
# export GLOO_SOCKET_IFNAME=eth0  # 或 enp, bond, 等主要接口

# export NCCL_IB_DISABLE=1 # 禁用 InfiniBand
# export NCCL_NET_GDR_LEVEL=0 # 关闭 GPU Direct RDMA



cd /mnt/nlp-ali/usr/hc/code/llm_train/llamda_factore_do


yaml_filename=${ori_checkpoint_name}_full_sft_ds_do_${ds_zero}_step${step}_${cutoff_len}_${lr}_packing_${packing}_neat_packing_${neat_packing}

new_yaml_path="/mnt/nlp-ali/usr/hc/code/llm_train/llamda_factore_do/train_script/dolphin/tmp/${yaml_filename}_${datetime}.yaml"

# 直接写出 YAML 到 $new_yaml_path
cat <<EOF > "$new_yaml_path"
### model
model_name_or_path: ${ori_checkpoint_path}

### method
stage: sft
do_train: true
finetuning_type: full
use_fast_tokenizer: true
deepspeed: examples/deepspeed/ds_z${ds_zero}_config.json
enable_liger_kernel: true
use_unsloth_gc: true

### dataset
dataset: $dataset
template: qwen2_vl
cutoff_len: $cutoff_len
max_samples: 100000000
overwrite_cache: true
preprocessing_num_workers: 64
packing: ${packing}
neat_packing: ${neat_packing}
image_max_pixels: 331776

tokenized_path: saves/tokenized/${tokenized}

### output
output_dir: $output_dir
logging_steps: $logging_steps
save_steps: $save_steps
plot_loss: true
overwrite_output_dir: true
include_num_input_tokens_seen: true
save_total_limit: $save_total_limit
load_best_model_at_end: true
metric_for_best_model: loss
greater_is_better: false
save_only_model: false

### train
per_device_train_batch_size: $per_device_train_batch_size
gradient_accumulation_steps: $gradient_accumulation_steps
learning_rate: $lr
num_train_epochs: $num_train_epochs
lr_scheduler_type: cosine
warmup_ratio: $warmup_ratio
bf16: true
ddp_timeout: 18000000
flash_attn: auto
use_liger_kernel: true
report_to: none
run_name: ${yaml_filename}_${datetime}
max_grad_norm: 1.0

### eval
val_size: $val_size
per_device_eval_batch_size: $per_device_eval_batch_size
eval_strategy: steps
eval_steps: $eval_steps

EOF

# 查看生成的 YAML
echo "======= Generated $new_yaml_path ======="
cat "$new_yaml_path"

# 然后执行训练
FORCE_TORCHRUN=1 \
    NNODES=$WORLD_SIZE \
    NPROC_PER_NODE=$GPU_NUM \
    NODE_RANK=$RANK \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    llamafactory-cli train "$new_yaml_path"