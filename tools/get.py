"""get.py find "<title words>"      -> list Europe PMC hits (title, year, pmcid, open access)
   get.py pmc PMC1234567 <outdir>   -> full text as plain text (Europe PMC XML), prints sections with word counts
   get.py url <url> <outdir>        -> fetch an html page (arxiv html / ar5iv) and strip to text"""
import sys, re, json, os, urllib.request, urllib.parse, html
UA = {'User-Agent': 'Mozilla/5.0 (research reader)'}
def fetch(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read().decode('utf-8', 'replace')
def strip(x):
    x = re.sub(r'<(script|style|xref|sup|table-wrap|ref-list|fig)[^>]*>.*?</\1>', ' ', x, flags=re.S)
    x = re.sub(r'<title>(.*?)</title>', lambda m: '\n\n## ' + re.sub(r'<[^>]+>', '', m.group(1)) + '\n', x, flags=re.S)
    x = re.sub(r'<h[1-4][^>]*>(.*?)</h[1-4]>', lambda m: '\n\n## ' + re.sub(r'<[^>]+>', '', m.group(1)) + '\n', x, flags=re.S)
    x = re.sub(r'</p>', '\n', x); x = re.sub(r'<[^>]+>', '', x); x = html.unescape(x)
    x = re.sub(r'[ \t]+', ' ', x); x = re.sub(r'\n\s*\n+', '\n\n', x)
    return x
def save(txt, out, name):
    os.makedirs(out, exist_ok=True); p = os.path.join(out, name)
    open(p, 'w').write(txt)
    lines = txt.split('\n'); print(p, len(txt.split()), 'words', len(lines), 'lines')
    for i, l in enumerate(lines):
        if l.startswith('## '): print('  L%d %s' % (i + 1, l[:90]))
m = sys.argv[1]
if m == 'find':
    q = urllib.parse.quote(sys.argv[2])
    d = json.loads(fetch('https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%s&format=json&pageSize=%s&resultType=lite' % (q, sys.argv[3] if len(sys.argv) > 3 else '6')))
    for r in d.get('resultList', {}).get('result', []):
        print(r.get('pubYear'), '|', r.get('pmcid'), '| OA', r.get('isOpenAccess'), '| inEPMC', r.get('inEPMC'), '|', r.get('authorString', '')[:40], '|', r.get('title', '')[:110], '|', r.get('journalTitle', ''), '| cited', r.get('citedByCount'))
elif m == 'pmc':
    x = fetch('https://www.ebi.ac.uk/europepmc/webservices/rest/%s/fullTextXML' % sys.argv[2])
    b = re.search(r'<body.*</body>', x, flags=re.S); a = re.search(r'<abstract.*?</abstract>', x, flags=re.S)
    t = re.search(r'<article-title>(.*?)</article-title>', x, flags=re.S)
    save('# ' + (re.sub(r'<[^>]+>', '', t.group(1)) if t else '') + '\n\n## Abstract\n' + strip(a.group(0) if a else '') + '\n' + strip(b.group(0) if b else x), sys.argv[3], sys.argv[2] + '.txt')
elif m == 'ncbi':
    x = fetch('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=%s&rettype=xml' % sys.argv[2].replace('PMC', ''))
    b = re.search(r'<body.*</body>', x, flags=re.S); a = re.search(r'<abstract.*?</abstract>', x, flags=re.S)
    t = re.search(r'<article-title>(.*?)</article-title>', x, flags=re.S)
    if not b: print('NO BODY', sys.argv[2]); sys.exit(1)
    save('# ' + (re.sub(r'<[^>]+>', '', t.group(1)) if t else '') + '\n\n## Abstract\n' + strip(a.group(0) if a else '') + '\n' + strip(b.group(0)), sys.argv[3], sys.argv[2] + '.txt')
elif m == 'url':
    x = fetch(sys.argv[2]); b = re.search(r'<body.*</body>', x, flags=re.S)
    save(strip(b.group(0) if b else x), sys.argv[3], 'page.txt')
