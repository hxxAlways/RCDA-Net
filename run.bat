@REM python train.py ^
@REM --model_params cda ^
@REM --net_name cda ^
@REM --weight_save_name cda ^
@REM --loss_function DiceLoss ^
@REM --epochs 150

python eval.py ^
--weight_name wpn rcda ^
--model_name wpn rcda ^
--threshold_list 0.5 0.6 ^
--UID_list 665 94 ^
--date_list 2016-07-15 2015-06-24

@REM python eval.py --weight_name rcda --model_name rcda