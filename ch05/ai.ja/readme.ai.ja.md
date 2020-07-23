このzipファイルに含まれる `ai.ja.txt` は，[日本語Wikipedia](https://ja.wikipedia.org/)の「[人工知能](https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD)」に関する記事から，テキスト部分を取り出したものです．

このテキストファイルの作成のために，[WikiExtractor](https://github.com/attardi/wikiextractor)を利用しました．ファイルを作成した大まかな手順は以下のとおりです．

```
# Extract text from ai.ja.xml (the article in XML format).
$ python WikiExtractor.py ai.ja.xml
# Prepare ai.ja.txt by editing the output from the tool, e.g., text/AA/wiki_00
```

このテキストファイルは，[クリエイティブ・コモンズ 表示-継承 3.0 非移植](https://creativecommons.org/licenses/by-sa/3.0/legalcode)のライセンスで配布されています．
