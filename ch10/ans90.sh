onmt_preprocess -train_src data/kyoto-train.ja -train_tgt data/kyoto-train.en -valid_src data/kyoto-dev.ja -valid_tgt data/kyoto-dev.en -save_data data/data -src_vocab_size 10000 -tgt_vocab_size 10000
onmt_train  \
  -data data/data  \
  -save_model data/demo-model  \
  -train_steps 100000  \
  -world_size 1  \
  -gpu_ranks 0
