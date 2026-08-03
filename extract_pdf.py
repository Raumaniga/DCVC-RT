import fitz
doc = fitz.open(r'e:\LAB\DCVC RT\toward.pdf')
with open(r'e:\LAB\DCVC RT\toward_text.txt', 'w', encoding='utf-8') as f:
    for page in doc:
        f.write(page.get_text())
        f.write('\n---PAGE BREAK---\n')
print("Done!")
