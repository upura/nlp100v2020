onmt_translate  \
  -model data/demo-model_step_100000.pt  \
  -src data/kyoto-test.ja  \
  -output pred.txt  \
  -replace_unk  \
  -verbose  \
  -gpu 0
