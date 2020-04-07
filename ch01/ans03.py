rawText = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
text = rawText.replace('.', '').replace(',', '')
ans = [len(w) for w in text.split()]
print(ans)
