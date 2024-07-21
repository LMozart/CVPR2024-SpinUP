#!/bin/bash
# run quick testing on precomputed models
CUDA_NUM="0"
TESTING="True"
QUICKTESTING="True"   # Change QUICKTESTING to "False" for more visualization results
echo cuda:$CUDA_NUM/Testing:$TESTING/quick_testing:$QUICKTESTING
echo "Start testing rand-light"

for f in ./configs/sync-rand-light/light_*.yml
do
  echo $f
  python code/train.py --config $f --cuda $CUDA_NUM
done

for f in ./configs/sync-rand-light/ref_*.yml
do
  echo $f
  python code/train.py --config $f --cuda $CUDA_NUM
done