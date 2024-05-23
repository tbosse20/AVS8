#!/bin/sh

srun --time=1-00 singularity run --nv /home/student.aau.dk/lk83xy/pytorch-24.01 python tts_pipeline.py --checkpoint_run=/home/student.aau.dk/lk83xy/avs8/AVS8/runs/baseline_24_4k/ --test
