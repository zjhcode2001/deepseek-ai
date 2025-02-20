#!/bin/bash
source activate Qwen2.5

mv /root/DeepSeek-R1-Distill-Qwen-7B /root/autodl-tmp
python /root/deepseek/inference/推理代码.py