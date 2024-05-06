#!/bin/sh

srun --time=1-00 --cpus-per-task=16 singularity run --nv /home/student.aau.dk/lk83xy/pytorch-24.01 python tts_pipeline.py
