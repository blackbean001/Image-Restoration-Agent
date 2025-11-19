CUDA_VISIBLE_DEVICE=0
python -m src.insert_emb_to_postgresql \
  --imgs-dir "../../../AgenticIR/output"  \
  --combiner-path "./models/combiner_trained_on_imgres_RN50x4/saved_models/combiner_arithmetic.pt"  \
  --combining-function 'combiner' \
  --clip-model-path "./models/clip_finetuned_on_imgres_RN50x4/saved_models/tuned_clip_arithmetic.pt"  \
  --projection-dim 2560 \
  --hidden-dim 5120 \
  --transform "targetpad"  \
  --target-ratio 1.25  \
