### train task#1 ###
sh run_train.sh --in_file_trn_dialog ../data/dialogue/tr/task1.ddata.wst.txt --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task1.wst.dev --model_path ./gAIa_CL_model
### eval task#1 ###
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task1.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task2.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task3.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task4.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task5.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task6.wst.dev --model_path ./gAIa_CL_model

### train task#2 ###
sh run_train.sh --in_file_trn_dialog ../data/dialogue/tr/task2.ddata.wst.txt --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task2.wst.dev --model_path ./gAIa_CL_model --model_file gAIa-final.pt
### eval task#2 ###
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task1.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task2.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task3.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task4.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task5.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task6.wst.dev --model_path ./gAIa_CL_model

### train task#3 ###
sh run_train.sh --in_file_trn_dialog ../data/dialogue/tr/task3.ddata.wst.txt --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task3.wst.dev --model_path ./gAIa_CL_model --model_file gAIa-final.pt
### eval task#3 ###
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task1.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task2.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task3.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task4.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task5.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task6.wst.dev --model_path ./gAIa_CL_model

### train task#4 ###
sh run_train.sh --in_file_trn_dialog ../data/dialogue/tr/task4.ddata.wst.txt --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task4.wst.dev --model_path ./gAIa_CL_model --model_file gAIa-final.pt
### eval task#4 ###
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task1.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task2.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task3.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task4.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task5.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task6.wst.dev --model_path ./gAIa_CL_model

### train task#5 ###
sh run_train.sh --in_file_trn_dialog ../data/dialogue/tr/task5.ddata.wst.txt --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task5.wst.dev --model_path ./gAIa_CL_model --model_file gAIa-final.pt
### eval task#5 ###
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task1.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task2.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task3.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task4.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task5.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task6.wst.dev --model_path ./gAIa_CL_model

### train task#6 ###
sh run_train.sh --in_file_trn_dialog ../data/dialogue/tr/task6.ddata.wst.txt --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task6.wst.dev --model_path ./gAIa_CL_model --model_file gAIa-final.pt
### eval task#6 ###
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task1.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task2.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task3.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task4.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task5.wst.dev --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/eval/cl_eval_task6.wst.dev --model_path ./gAIa_CL_model

### test all ###
sh run_test.sh --in_file_tst_dialog ../data/dialogue/tt/cl_eval_task1.wst.tst --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/tt/cl_eval_task2.wst.tst --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/tt/cl_eval_task3.wst.tst --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/tt/cl_eval_task4.wst.tst --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/tt/cl_eval_task5.wst.tst --model_path ./gAIa_CL_model
sh run_test.sh --in_file_tst_dialog ../data/dialogue/tt/cl_eval_task6.wst.tst --model_path ./gAIa_CL_model