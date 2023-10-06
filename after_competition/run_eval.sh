# $1: --in_file_tst_dialog
# $2: filename of $1
# $3: --model_path
# $4: path for loading trained model

CUDA_VISIBLE_DEVICES="0" python3 ./main.py --exp_group_name baseline7 \
                                   --seed 2023 \
                                   --mode test \
                                   --in_file_fashion /home/suyeongp7/data/item_metadata/mdata.wst.txt.2023.08.23 \
                                   --subWordEmb_path /home/suyeongp7/data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                   --model_file gAIa-final.pt \
                                   --mem_size 8 \
                                   --key_size 300 \
                                   --hops 3 \
                                   --eval_node [6000,3000,1000,500,200][2000] \
                                   --batch_size 100 \
                                   $1 $2 \
                                   $3 $4
