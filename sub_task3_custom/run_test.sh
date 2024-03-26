# $1: --exp_name
# $2: experiment name of $1
# $3: --task_ids
# $4: task ids of training and evaluation data of $3
# $5: --in_file_tst_dialog
# $6: filename of $5
# $7: --model_path
# $8: path for loading trained model of $7

CUDA_VISIBLE_DEVICES="0" python3 ./main.py --seed 2023 \
                                   --mode eval \
                                   --task_id 1 \
                                   --in_file_fashion ../data/mdata.wst.txt.2023.08.23 \
                                   --subWordEmb_path ../data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                   --model_file gAIa-final.pt \
                                   --mem_size 8 \
                                   --key_size 300 \
                                   --hops 3 \
                                   --eval_node [6000,3000,1000,500,200][2000] \
                                   --batch_size 100 \
                                   $1 $2 \
                                   $3 $4 \
                                   $5 $6 \
                                   $7 $8
