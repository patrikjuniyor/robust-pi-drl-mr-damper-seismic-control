import json, urllib.request, urllib.parse, sys, time
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'refcheck/1.0 (mailto:example@example.org)'})
    return json.load(urllib.request.urlopen(req,timeout=25))
def fmt(m):
    au=m.get('author',[{}]); a0=au[0].get('family','?') if au else '?'
    yr=(m.get('issued',{}).get('date-parts',[[None]])[0][0])
    return f"{a0} ({len(au)} au) {yr} | {(m.get('title') or ['?'])[0][:90]} | {(m.get('container-title') or [''])[0][:40]} | v{m.get('volume','')}({m.get('issue','')}) p{m.get('page','')} | {m.get('DOI')}"
items=[l.rstrip('\n').split('|',1) for l in open(sys.argv[1]) if l.strip()]
out={}
for key,q in items:
    try:
        if q.startswith('10.'):
            m=get('https://api.crossref.org/works/'+urllib.parse.quote(q))['message']
        else:
            m=get('https://api.crossref.org/works?rows=1&query.bibliographic='+urllib.parse.quote(q))['message']['items'][0]
        print(key,'=>',fmt(m)); out[key]=m
    except Exception as e:
        print(key,'ERR',str(e)[:80])
    time.sleep(0.15)
json.dump({k:{f:v.get(f) for f in ('DOI','title','author','issued','container-title','volume','issue','page','publisher','type')} for k,v in out.items()},open(sys.argv[1]+'.json','w'))
