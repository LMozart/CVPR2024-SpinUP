#!/bin/bash
# run quick testing on precomputed models
CUDA_NUM="0"
TESTING="True"
QUICKTESTING="True"   # Change QUICKTESTING to "False" for more visualization results
echo cuda:$CUDA_NUM/Testing:$TESTING/quick_testing:$QUICKTESTING
echo "Start initialize rand-light"

for f in ./data/Sync_rand/shape_*
do
  echo $f
  python code/preprocess/light_init/init_env_map_gray_fixlobe_sh.py --data_path $f --gpu $CUDA_NUM
done

for f in ./data/Sync_rand/sv_*
do
  echo $f
  python code/preprocess/light_init/init_env_map_gray_fixlobe_sh.py --data_path $f --gpu $CUDA_NUM
done