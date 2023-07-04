# first stage
cd first_stage/irn
python run_sample.py --train_cam_pass True  --make_cam_pass True --eval_cam_pass True
python run_sample.py --num_worker 8 --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True
python gen_mask.py

# second stage
CUDA_VISIBLE_DEVICES=0 python main.py -dist --logging_tag test
CUDA_VISIBLE_DEVICES=0 python main.py --test --logging_tag test --ckpt data/logging/test/best_ckpt.pth

# second stage debug
CUDA_VISIBLE_DEVICES=0 python main.py --logging_tag test
