#!/bin/bash
# run quick testing on precomputed models
CUDA_NUM="0"
TESTING="True"
QUICKTESTING="True"   # Change QUICKTESTING to "False" for more visualization results
echo cuda:$CUDA_NUM/Testing:$TESTING/quick_testing:$QUICKTESTING
echo "Start testing rand-light"

for f in ./configs/sync-rand-light/*.yml
do
  echo $f
  python code/test.py --config $f --cuda $CUDA_NUM --testing $TESTING --quick_testing $QUICKTESTING
done