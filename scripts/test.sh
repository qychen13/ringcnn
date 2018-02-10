#!/usr/bin/env bash
python test.py -gs 0 -b 128 -nw 8 -en 'trainval-test' -model 'resnet56' -nc 100 -dir /media/jamesch/Dataset -tm 'checkpoints/init_model.pth.tar'