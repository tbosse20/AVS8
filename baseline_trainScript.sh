#!/bin/sh

srun --time=1-00 singularity run --nv /home/student.aau.dk/lk83xy/pytorch-24.01 python baseline_tts_pipeline.py -n "baseline" --train --base
