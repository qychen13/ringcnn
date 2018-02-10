#!/usr/bin/env bash
python trainval.py -gs 0 1 -b 16 -nw 2 -ilog 30 -en 'trainval-test' -model 'deeplabv3-resnet50-4blks' -nc 21 -dir /media/jamesch/Dataset -cifrec 5000 -cefrec 5 -cpath checkpoints -e 300 -lr 0.007 -iter 0 -epo 0 -tl 1