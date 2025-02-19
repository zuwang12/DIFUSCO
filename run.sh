export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set the number of cities
num_cities=100
constraint_type='basic'
now=$(date +"%Y%m%d_%H%M%S")
f_name=difusco_${constraint_type}_tsp${num_cities}_${now}

export WANDB_DISABLED=true

nohup python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp${num_cities}" \
  --diffusion_type "categorical" \
  --do_test true \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "" \
  --training_split "data/tsp/tsp${num_cities}_test_concorde.txt" \
  --validation_split "data/tsp/tsp${num_cities}_test_concorde.txt" \
  --test_split "data/tsp/tsp${num_cities}_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --ckpt_path "ckpt/tsp${num_cities}_categorical.ckpt" \
  --f_name ${f_name} \
  --use_ddp true \
  > logs/${f_name}.log 2>&1 &
