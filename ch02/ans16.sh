n=`wc -l ch02/popular-names.txt | awk '{print $1}'`
ln=`expr $n / $1`
split -l $ln ch02/popular-names.txt ch02/ans16_
