cut -f1 -d$'\t' ch05/ans47.txt >> ch05/col1.txt
cut -f2 -d$'\t' ch05/ans47.txt >> ch05/col2.txt
paste ch05/col1.txt ch05/col2.txt >> ch05/tmp.txt
sed -e 's/[[:cntrl:]]/ /g' ch05/tmp.txt >> ch05/tmp2.txt
cut -f1 -d$'\t' ch05/tmp2.txt | LANG=C sort | uniq -c | sort -r -k 2 -t ' '
