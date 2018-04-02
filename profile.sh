#!/bin/bash

for indexID in {0..1}
    do
        tfileName="vgg_time${indexID}.log"
        nvprof --csv --log-file ${tfileName} --print-gpu-trace python vgg_generator.py $indexID
        mfileName="vgg_mem${indexID}.log"
        nvprof --csv --log-file ${mfileName} --print-gpu-trace --metrics dram_write_transactions,dram_read_transactions python vgg_generator.py $indexID

        sfileName="vgg_sample${indexID}.log"

        python genSample.py $tfileName $mfileName $sfileName
    done

