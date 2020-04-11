cut -f1 -d$'\t' ch05/ans47.txt | LANG=C sort | uniq -c | sort -r -k 2 -t ' '
