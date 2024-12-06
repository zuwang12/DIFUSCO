export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# # shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp50" \
  --diffusion_type "categorical" \
  --do_test true \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "" \
  --training_split "data/tsp/tsp50_test_concorde1.txt" \
  --validation_split "data/tsp/tsp50_test_concorde1.txt" \
  --test_split "data/tsp/tsp50_test_concorde1.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --ckpt_path "ckpt/tsp50_categorical.ckpt" \
  > ./logs/difusco50.log 2>&1