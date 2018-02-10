#!/usr/bin/env bash
python trainval.py -gs 0 1 -b 100 -nw 8 -ilog 30 -en 'trainval-test' -model 'resnet50' -nc 1000 -dir /media/jamesch/Dataset/ImageNet -cifrec 5000 -cefrec 5 -cpath checkpoints -e 300 -lr 0.1 -iter 0 -epo 0
