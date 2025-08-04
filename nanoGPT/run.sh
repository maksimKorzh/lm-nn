# Prepare dataset
python prepare.py

# Override config params and run custom train
python train.py                          \
       config.py                         \
       --device=cpu                      \
       --compile=False                   \
       --eval_iters=20                   \
       --log_interval=1                  \
       --block_size=12                   \
       --batch_size=12                   \
       --n_layer=6                       \
       --n_head=6                        \
       --n_embd=384                      \
       --max_iters=500                  \
       --lr_decay_iters=2000             \
       --dropout=0.0
