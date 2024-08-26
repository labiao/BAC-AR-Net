# NEED TO SET
DATASET_ROOT=./Dataset/RRDSD
#GPU=1,2,3
GPU=2

# 1. train a classification network and compute refined seed
# 1.1 train a classification network
# resnet50_cam ----> res50_cam_vai.pth 
# RepVGG_cam  ---->  repvgg_cam_vai.pth
# 1.2 generate, evaluate init_cams, generate ir_label
# 1.3 BAC module
for nc in 20
do
 CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
      --voc12_root ${DATASET_ROOT} \
      --train_list img_labels_rrdsd/train_aug.txt \
      --num_workers 8 \
      --cam_weights_name sess_rrdsd/repvgg_cam  \
      --cam_learning_rate 0.01 --cam_batch_size 16 --cam_num_epoches 20 \
      --cam_network net.RepVGG_cam \
      --train_cam_pass False \
      --make_cam_pass True --cam_out_dir result_rrdsd/gradcam \
      --eval_cam_pass False \
#      --work_space result_rrdsd --num_cluster ${nc} \
#      --lpcam_out_dir result_cluster/lpcam_mask_${nc}_repvgg \
#      --make_lpcam_pass False \
#      --cam_out_dir result_cluster/lpcam_mask_${nc}_repvgg \
#      --eval_cam_pass False \
#      --conf_fg_thres 0.25 \
#      --conf_bg_thres 0.15 \
#      --cam_to_ir_label_pass False
done


# 2.1. train a AR module
#for lmd in 0.2 0.4 0.6 0.8
#for lmd in 1
#do
# CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
#     --voc12_root ${DATASET_ROOT} \
#     --train_list img_labels_rrdsd/train_aug.txt \
#     --infer_list img_labels_rrdsd/train_aug_building.txt \
#     --num_workers 8 \
#     --amn_crop_size 256 \
#     --amn_weights_name sess_rrdsd/repvgg_lpcam_l${lmd}_cam_mask_1_0.pth \
#     --amn_num_epoches 10 \
#     --amn_network net.RepVGG_amn \
#     --Lambda1 ${lmd} --Lambda2 0.0 \
#     --train_amn_pass False \
#     --make_amn_cam_pass False \
#     --eval_amn_cam_pass True
#done


