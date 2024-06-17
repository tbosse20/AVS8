#!/bin/sh

srun --time=5-00 singularity run --nv /home/student.aau.dk/lk83xy/pytorch-24.01 python baseline_tts_pipeline.py -n "baseline chpnt" --train --base --checkpoint_run "/home/student.aau.dk/lk83xy/avs8/AVS8/runs/run-May-10-2024_06+16PM-c65a463"
