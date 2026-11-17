"""tab.py PMCxxxx -> print every table of the article (caption + rows), from the NCBI full-text XML"""
import sys, re, html, urllib.request
x = urllib.request.urlopen(urllib.request.Request('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=%s&rettype=xml' % sys.argv[1].replace('PMC', ''), headers={'User-Agent': 'Mozilla/5.0 (research reader)'}), timeout=60).read().decode('utf-8', 'replace')
cl = lambda t: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', t))).strip()
tw = re.findall(r'<table-wrap\b.*?</table-wrap>', x, flags=re.S)
print(len(tw), 'tables')
for t in tw:
    lab = re.search(r'<label>(.*?)</label>', t, flags=re.S); cap = re.search(r'<caption>(.*?)</caption>', t, flags=re.S)
    print('\n#### ', cl(lab.group(1)) if lab else '', '-', cl(cap.group(1))[:300] if cap else '')
    for r in re.findall(r'<tr\b.*?</tr>', t, flags=re.S):
        print(' | '.join(cl(c) for c in re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', r, flags=re.S)))
    ft = re.search(r'<table-wrap-foot>(.*?)</table-wrap-foot>', t, flags=re.S)
    if ft: print('FOOT:', cl(ft.group(1))[:600])
