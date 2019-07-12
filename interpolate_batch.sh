for i in {1..8}
do
  python interpolate.py --prev ./images/interpolate/Camera00$i.png --succ ./images/interpolate/Camera00$((i+1)).png --dest ./images/interpolate/Camera0$((i+12))i.png --model ./out_1_L1_kernel51/model_epoch_30.pth
done

python interpolate.py --prev ./images/interpolate/Camera009.png --succ ./images/interpolate/Camera010.png --dest ./images/interpolate/Camera021i.png --model ./out_1_L1_kernel51/model_epoch_30.pth

python interpolate.py --prev ./images/interpolate/Camera010.png --succ ./images/interpolate/Camera011.png --dest ./images/interpolate/Camera022i.png --model ./out_1_L1_kernel51/model_epoch_30.pth

python interpolate.py --prev ./images/interpolate/Camera011.png --succ ./images/interpolate/Camera012.png --dest ./images/interpolate/Camera023i.png --model ./out_1_L1_kernel51/model_epoch_30.pth

python interpolate.py --prev ./images/interpolate/Camera012.png --succ ./images/interpolate/Camera001.png --dest ./images/interpolate/Camera024i.png --model ./out_1_L1_kernel51/model_epoch_30.pth
