# $1: --in_file_trn_dialog
# $2: filename of $1
# $3: --in_file_tst_dialog
# $4: filename of $3
# $5: --model_path
# $6: path for saving trained model
# $7: --model_file
# $8: filename of $7 (loaded file after task#1)

CUDA_VISIBLE_DEVICES="0" python3 ./main.py --seed 2023 \
                                     --mode train \
                                     --in_file_fashion /home/suyeongp7/data/mdata.wst.txt.2023.08.23 \
                                     --subWordEmb_path /home/suyeongp7/data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                     --mem_size 8 \
                                     --key_size 300 \
                                     --hops 3 \
                                     --eval_node [6000,3000,1000,500,200][2000] \
                                     --epochs 40 \
                                     --save_freq 80 \
                                     --batch_size 100 \
                                     --learning_rate 0.005 \
                                     --max_grad_norm 20.0 \
                                     --use_dropout True \
                                     --zero_prob 0.25 \
                                     --permutation_iteration 6 \
                                     --num_augmentation 3 \
                                     --corr_thres 0.99 \
                                     $1 $2 \
                                     $3 $4 \
                                     $5 $6 \
                                     $7 $8
