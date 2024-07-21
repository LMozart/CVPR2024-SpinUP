#!/bin/bash
# run quick testing on precomputed models
CUDA_NUM="1"
TESTING="True"
QUICKTESTING="True"   # Change QUICKTESTING to "False" for more visualization results
echo cuda:$CUDA_NUM/Testing:$TESTING/quick_testing:$QUICKTESTING
echo "Start training real-world dataset"

for f in ./configs/real/outdoor/*.yml
do
  echo $f
  python code/train.py --config $f --cuda $CUDA_NUM
done