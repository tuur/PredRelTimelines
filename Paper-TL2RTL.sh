# Script obtain the LT2RTL timelines from the EMNLP'18 paper.

date='DD-MM-YYYY' # date of today (for naming the output folder)

num_epochs=10000

for lossfunc in Lt Ldh Ldce L*; do

  # ----------------- TE3 data split

  prefix=$date'/TE3/'
  testdata="data/S-TE3/Test-normalized/"
  data="data/S-TE3/ning2017-table4-line6-normalized/"
  exp=$prefix"TL2RTL-"$lossfunc

  python timeml_to_timeline.py -timeml $data -pointwise_loss $pointwiseloss -loss_func $lossfunc -test $testdata -num_epochs $num_epochs -exp_id $exp &
  sleep 2
  exit

  # ----------------- TD data split

  prefix=$date'/TD/'
  testdata="data/S-TD/Test-normalized/"
  data="data/ning2017-table4-line13-normalized/"
  exp=$prefix"TL2RTL-"$lossfunc

  python timeml_to_timeline.py -timeml $data -pointwise_loss $pointwiseloss -loss_func $lossfunc -test $testdata -num_epochs $num_epochs -exp_id $exp &
  sleep 2

done
