# Script to run train and test the TLM models from the EMNLP'18 paper.
# "da,sa" indicates the S-TLM model, and "dc,sc" indicates the C-TLM
# Evaluation F-scores can be found in the output log files

date='DD-MM-YYYY' # date of today (for naming the output folder)
cuda=-1 # gpu number to use (-1 uses no gpus)

#--------------------------- TD data split

split='TD'
simplification=0
prefix=$date'/'$split'/'
train='data/S-'$split'/Train/'
test='data/S-'$split'/Test/'

for loss in Lt Ldce Ldh; do
 for model_struct in "da,sa" "dc,sc"; do # 2 * 3 = 6 exps
  exp=$prefix"TLM-"$loss"-"$model_struct
  python main.py -exp_id $exp -train $train -test $test -cuda $cuda -simplification $simplification -loss_func $loss #&
  exit
  sleep 2
 done
done

#--------------------------- TBAQ+TD+VC-TE3 data split

split='TE3'
prefix=$date'/'$split'/'
train='data/S-'$split'/Train/'
test='data/S-'$split'/Test/'

for loss in Lt Ldce Ldh;do 
 for model_struct in "dc,sc" "da,sa"; do # 2 * 3 = 6 exps
  exp=$prefix"TLM-"$loss"-"$model_struct
  python main.py -exp_id $exp -train $train -test $test -cuda $cuda -subtasks $model_struct &
  sleep 2
 done
done


