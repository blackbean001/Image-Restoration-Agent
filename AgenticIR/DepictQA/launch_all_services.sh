CUDA_VISIBLE_DEVICES=4 nohup python src/app_comp.py >> log_app_comp.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python src/app_eval.py >> log_app_eval.log 2>&1 &
