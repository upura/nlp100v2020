def cipher(text):
    encodeText = [chr(219 - ord(w)) if 97 <= ord(w) <= 122 else w for w in text]
    return encodeText


text = 'this is a message.'
ans = cipher(text)
print(''.join(ans))
ans = cipher(ans)
print(''.join(ans))
