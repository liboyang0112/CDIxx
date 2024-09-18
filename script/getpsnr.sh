getpsnr(){
  cdi_run ../cdi.cfg > /dev/null
  for y in {-1..1} ;
  do
    for x in {-1..1} ;
    do
      takepsnr_run ../input0.png recon.bin $x $y tp 1 0.35 | grep -E "ssim=|psnr=" | awk -F '=' '{print $2}' | sed -z 's/\n/ /' >> tmp
    done
  done

  for y in {-1..1} ;
  do
    for x in {-1..1} ;
    do
      takepsnr_run ../input0.png recon.bin $x $y p 1 0.35 | grep -E "ssim=|psnr=" | awk -F '=' '{print $2}' | sed -z 's/\n/ /' >> tmp
    done
  done
  cat tmp | sort -n | tail -1
  rm tmp
}

rm data.dat

for x in {0..10}
do
  getpsnr >> data.dat
done

take_avg.py data.dat
