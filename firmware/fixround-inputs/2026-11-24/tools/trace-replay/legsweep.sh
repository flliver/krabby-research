#!/bin/sh
# inside container: sweep hip start distances for a hip curl trace
for hip in f1 realsw128 realsw128b realsw150; do
  for loop in 1 2 5; do
    max=$(awk -v h=$hip '$1==h {getline; n=split($0,a," "); print a[n-1]}' out/legdata.txt)
    [ "$max" -gt 420 ] && max=420
    starts=$(seq 0 2 $max)
    out/legsim $hip $loop 1 $starts < out/legdata.txt > out/legsweep_${hip}_${loop}.txt
  done
done
