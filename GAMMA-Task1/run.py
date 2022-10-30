import os
import time

start_time = time.time()

print('需要修改inference_grading_final.py中的testset_root')

cmd = 'CUDA_VISIBLE_DEVICES=1 python inference_grading_final.py --model_mode x50 --sub1dataset_mode "orig" --eval_kappa "8993"'
print(cmd)
os.system(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python inference_grading_final.py --model_mode 34 --sub1dataset_mode "crop" --eval_kappa "8438"'
print(cmd)
os.system(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python inference_grading_final.py --model_mode x50 --sub1dataset_mode "orig" --eval_kappa "8625"'
print(cmd)
os.system(cmd)

cmd = 'python emsemble.py'
print(cmd)
os.system(cmd)


print('time cost:' + str(int(time.time()-start_time)) + 's')


