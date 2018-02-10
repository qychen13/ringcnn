#!/usr/bin/env bash
python test.py -gs 0 1 -b 30 -nw 2 -en 'trainval-test' -model 'deeplabv3-resnet50-4blks' -nc 21 -dir /media/jamesch/Dataset -tm 'checkpoints/e60t18720.pth.tar' -lp 'logs'
