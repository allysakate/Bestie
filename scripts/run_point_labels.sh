# Training BESTIE with point labels.

ROOT=/home/allysakate/Videos/cvat_2023
SUP=point
NUM_CLASS=10
REFINE_WARMUP=0
SIZE=416
BATCH=2
WORKERS=4
TRAIN_ITERS=50000
BACKBONE=hrnet48 # [resnet50, resnet101, hrnet32, hrnet48]
VAL_IGNORE=False
DATASET=coco
PROCESS_NO=1

CUDA_VISIBLE_DEVICES=0 && python main.py --dataset ${DATASET} --num_classes ${NUM_CLASS} \
--root_dir ${ROOT} --sup ${SUP} --batch_size ${BATCH} --num_workers ${WORKERS} --crop_size ${SIZE} \
--train_iter ${TRAIN_ITERS} --refine True --refine_iter ${REFINE_WARMUP} \
--val_freq 1000 --val_thresh 0.1 --val_ignore ${VAL_IGNORE} --val_clean False --val_flip True \
--seg_weight 1.0 --center_weight 200.0 --offset_weight 0.01 \
--lr 5e-5 --backbone ${BACKBONE} --random_seed 3407 \
--save_folder checkpoints/${DATASET}_${BACKBONE}_${PROCESS_NO}
