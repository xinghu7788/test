for indexID in {0..2999}
     do
         tfileName="log/vgg_time${indexID}.log"
         CUDA_VISIBLE_DEVICES=1 nvprof --csv --log-file ${tfileName} --print-gpu-trace python vgg_generator_RT.py $indexID
         mfileName="log/vgg_mem${indexID}.log"
         CUDA_VISIBLE_DEVICES=1 nvprof --csv --log-file ${mfileName} --print-gpu-trace --metrics dram_write_transactions,dram_read_transactions python vgg_generator_RT.py $indexID
         sfileName="log/vgg_sample${indexID}.log"

         #python genSample.py $tfileName $mfileName $sfileName
     done

