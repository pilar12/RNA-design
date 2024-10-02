#!/bin/bash

for ds in syn_hk syn_ns syn_multiplet
do 
    for v in 0 1 2
    do
        python eval.py --path runs --v $v --ds $ds
        python eval.py --path runs --v $v --ds $ds --gc True
    done
done

python eval_antarna.py
python eval.py --path runs/learna_suite/learna/ --ds syn_ns
python eval.py --path runs/learna_suite/liblearna/ --ds syn_ns
python eval.py --path runs/learna_suite/liblearna_gc/ --ds syn_ns --gc True
python eval.py --path runs/learna_suite/meta_learna/ --ds syn_ns
python eval.py --path runs/learna_suite/meta_learna_adapt/ --ds syn_ns
python eval.py --path runs/samfeo/ --ds syn_ns