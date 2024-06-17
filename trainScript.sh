#!/bin/sh

srun --time=5-00 singularity run --nv /home/student.aau.dk/lk83xy/pytorch-24.01 python tts_pipeline.py --checkpoint_run="/home/student.aau.dk/lk83xy/avs8/AVS8/runs/run-May-09-2024_10+07AM-7a6b52b/" --train
