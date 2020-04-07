with open('ch02/popular-names.txt', 'r') as f:
    fileText = f.read()
    after = fileText.replace('\t', ' ')

with open('ch02/popular-names_11p.txt', mode='w') as f:
    f.write(after)
