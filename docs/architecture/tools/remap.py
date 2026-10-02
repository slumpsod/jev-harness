"""Re-anchor file:line references after the code moved: each ref's line at <old-commit> is looked up
by content in today's file (the nearest identical line wins). Unresolved refs are listed for a human.

    python docs/architecture/tools/remap.py <old-commit> docs/architecture/*.md docs/architecture/tools/components.js
"""
import re,subprocess,sys
OLD=sys.argv[1]
def old_lines(f):
    return subprocess.run(["git","show",f"{OLD}:jevh/{f}"],capture_output=True,text=True).stdout.splitlines()
def new_lines(f):
    return open(f"jevh/{f}").read().splitlines()
cache={}
def remap(f,n):
    if f not in cache: cache[f]=(old_lines(f),new_lines(f))
    o,nw=cache[f]
    if n>len(o): return None
    target=o[n-1]
    hits=[i+1 for i,l in enumerate(nw) if l==target]
    if not hits: return None
    return min(hits,key=lambda h:abs(h-n))
RE=re.compile(r'((?:[\w]+/)*[\w]+\.py):(\d+)(?:-(\d+))?')
bad=[];changed=0
paths=sys.argv[2:]
for md in paths:
    s=open(md).read()
    def fix(m):
        global changed
        f=m.group(1); 
        import os
        if not os.path.exists("jevh/"+f): return m.group(0)
        a=remap(f,int(m.group(2)))
        b=remap(f,int(m.group(3))) if m.group(3) else None
        if a is None or (m.group(3) and b is None):
            bad.append((md,m.group(0))); return m.group(0)
        out=f"{f}:{a}"+(f"-{b}" if m.group(3) else "")
        if out!=m.group(0): changed+=1
        return out
    s2=RE.sub(fix,s)
    # bare ":NNN" continuation refs like `:367` after a file ref are rare; report them
    open(md,'w').write(s2)
print("changed",changed); print("unresolved",bad)
