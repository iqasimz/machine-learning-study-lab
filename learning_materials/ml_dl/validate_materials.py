"""Check authored course sources without downloads, installs or model fitting."""
from pathlib import Path
import json
import math
import re

ROOT = Path(__file__).resolve().parent
EXISTING = {
    "P1":"pre_llm/01_COMPUTERS_TENSORS_AND_SOFTWARE.md",
    "P2":"pre_llm/02_LINEAR_ALGEBRA_AND_CALCULUS.md",
    "P3":"pre_llm/03_PROBABILITY_INFORMATION_AND_STATISTICS.md",
    "P4":"pre_llm/04_LEARNING_AND_GENERALIZATION.md",
    "P5":"pre_llm/05_NEURAL_NETWORKS_AND_OPTIMIZATION.md",
    "T0":"llm_training/00_THE_COMPLETE_TRAINING_LOOP.md",
    "T3":"llm_training/03_TRANSFORMER_FROM_FIRST_PRINCIPLES.md",
    "R1":"post_training/01_SFT_AND_PARAMETER_EFFICIENT_ADAPTATION.md",
}

def check(condition,message):
    if not condition:raise AssertionError(message)

def load(name,key):
    obj=json.loads((ROOT/"metadata"/name).read_text(encoding="utf-8"))
    check(obj["schema_version"]==1,name)
    rows=obj[key]
    check(len({r["id"] for r in rows})==len(rows),f"Duplicate IDs: {name}")
    return rows

def close(value,expected,tolerance=1e-4):
    check(math.isclose(value,expected,rel_tol=0,abs_tol=tolerance),
          f"Arithmetic mismatch: {value} versus {expected}")

def main():
    lessons=load("lessons.json","lessons")
    params=load("parameters.json","parameters")
    datasets=load("datasets.json","datasets")
    sources=load("sources.json","sources")
    experiments=load("experiments.json","experiments")
    lesson_ids={x["id"] for x in lessons}
    param_ids={x["id"] for x in params}
    data_ids={x["id"] for x in datasets}
    source_ids={x["id"] for x in sources}
    experiment_ids={x["id"] for x in experiments}
    check(len(lessons)==30 and len(params)==98,"Intended coverage changed")
    check(sum(l["stage"]=="ml" for l in lessons)==16,"ML count")
    check(sum(l["stage"]=="dl" for l in lessons)==14,"DL count")
    for key,filename in EXISTING.items():
        check((ROOT.parent/filename).is_file(),f"Missing prerequisite {key}")
    previous=set(EXISTING);graph={};words=0;min_words=1000000
    for l in lessons:
        check(l["status"]=="integrated_course" and l["website_integrated"] is True,
              f"Integration status missing: {l['id']}")
        check(set(l["prerequisites"])<=previous,f"Prerequisite order: {l['id']}")
        previous.add(l["id"])
        graph[l["id"]]=[p for p in l["prerequisites"] if p in lesson_ids]
        for field,allowed in [("parameter_ids",param_ids),("dataset_ids",data_ids),
                              ("source_ids",source_ids),("experiment_ids",experiment_ids)]:
            check(set(l[field])<=allowed,f"Unknown {field}: {l['id']}")
        path=(ROOT/l["file"]).resolve()
        check(path.is_relative_to(ROOT) and path.is_file(),f"Unsafe/missing source: {l['id']}")
        body=path.read_text(encoding="utf-8")
        check(body.startswith(f"# {l['id']}. {l['title']}\n"),f"Title mismatch: {l['id']}")
        for marker in ["**Key terms.**","**Process.**","**Symbols and assumptions.**",
                       "## 2. A worked calculation",
                       "**Check.**","**Answer.**","## Sources"]:
            check(marker in body,f"Missing {marker}: {l['id']}")
        # Topic-specific headings replace the former identical section labels.
        # Preserve the numbered section contract used by the live note reader.
        sections = re.findall(r"^## (\d+)\. (.+)$", body, re.M)
        check(len(sections)>=4 and [int(number) for number, _ in sections] == list(range(1,len(sections)+1)),
              f"Missing, repeated or unordered lesson sections: {l['id']}")
        check(all(len(title.strip()) >= 8 for _, title in sections),
              f"Uninformative section title: {l['id']}")
        check(body.index("**Key terms.**")<body.index("**Process.**")<
              body.index("**Symbols and assumptions.**"),f"Definition order: {l['id']}")
        count=len(body.split());words+=count;min_words=min(min_words,count)
        check(count>=350,f"Lesson is only an outline: {l['id']} ({count} words)")
        check(not re.search(r"\b(?:TODO|TBD|lorem ipsum)\b",body,re.I),f"Placeholder: {l['id']}")
    visited=set()
    def visit(key,active):
        check(key not in active,f"Prerequisite cycle: {key}")
        if key in visited:return
        for before in graph[key]:visit(before,active|{key})
        visited.add(key)
    for key in graph:visit(key,set())
    for p in params:
        for field in ["label","role","methods","proposed_choices","definition",
                      "mathematical_meaning","expected_effect","interactions",
                      "failure_modes","source_ids","api_mapping"]:
            check(bool(p[field]),f"Empty parameter {field}: {p['id']}")
        check(p["status"]=="planned",f"Control not implemented: {p['id']}")
        check(set(p["source_ids"])<=source_ids,f"Control source: {p['id']}")
    for d in datasets:
        check(d["approval"] in {"approved_for_design","conditional","on_hold"},"Dataset approval")
        check(d["artifact_sha256"] is None,f"Invented hash: {d['id']}")
        for field in ["scope","license","split_policy","preprocessing","leakage_risks","preparation_status"]:
            check(bool(d[field]),f"Empty dataset {field}: {d['id']}")
        check(set(d["source_ids"])<=source_ids,f"Dataset source: {d['id']}")
    for s in sources:
        check(s["url"].startswith("https://"),f"Non-HTTPS source: {s['id']}")
        check(s["checked_on"]=="2026-11-02",f"Check date: {s['id']}")
        check(s["verification"] in {"page_accessed","landing_only","linked_pdf_accessed"},"Source evidence")
    first_lab_data=set()
    for e in experiments:
        check(set(e["dataset_ids"])<=data_ids,f"Experiment data: {e['id']}")
        check(set(e["lesson_ids"])<=lesson_ids,f"Experiment lessons: {e['id']}")
        check(e["status"]=="planned" and e["results"] is None,f"Invented result: {e['id']}")
        check(len(e["controls"])>=2 and e["evidence_required"],f"Weak contract: {e['id']}")
        if e["phase"]=="first_lab":first_lab_data.update(e["dataset_ids"])
    approved={d["id"] for d in datasets if d["approval"]=="approved_for_design"}
    check(first_lab_data<=approved,"First lab uses held/conditional data")
    links=0
    for path in ROOT.rglob("*.md"):
        body=path.read_text(encoding="utf-8")
        check(len(re.findall("^"+re.escape(chr(96)*3),body,re.M))%2==0,f"Unbalanced fences: {path}")
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)",body):
            if target.startswith(("https://","http://","mailto:","#")):continue
            check((path.parent/target.split("#")[0]).exists(),f"Broken link: {path}: {target}")
            links+=1

    # Independent calculations, not training runs.
    close(((5-3)**2+(5-7)**2)/2,4)
    close((2*1-1)*2,2)
    close(((2*.8-.1)-1)**2/2,.125)
    close(-math.log(.8),.22314355)
    close(.2*3**2/2,.9)
    close(.06/(.06+.016),.78947368)
    close(.5-(1-.75**2-.25**2),.125)
    close((.9**2+1.8**2)/2,2.025)
    close(2*.9*.6/(.9+.6),.72)
    close((.2**2+.3**2)/2,.065)
    close(math.sqrt(math.log(400)/2000),.0547333)
    close(2/1.25,1.6);close(1-1/1.25,.2)
    for x1,x2,expected in [(0,0,0),(1,0,1),(0,1,1),(1,1,0)]:
        close(max(0,x1+x2)-2*max(0,x1+x2-1),expected)
    close(2*3+3+3*2+2,17)
    close(1-.1*(.2/(1-.9))/math.sqrt(.004/(1-.999))-.1*.01,.899)
    close(math.floor((28+2-(3-1)-1)/2+1),14)
    close(16*1*3*3+16,160)
    close(160+(32*16*9+32)+(32*10+10),5130)
    close(784*64+64+64*10+10,50890)
    close(.9*2+.2*.5,1.9)
    a=math.exp(1/math.sqrt(2))/(math.exp(1/math.sqrt(2))+1)
    close(2*a,1.3395231);close(4*(1-a),1.3209538)
    close(.8+.6*.5,1.1)
    close(3/(7+5-3),1/3)
    close(math.ceil(1000/64)*10,160)
    close(math.ceil(math.ceil(1000/64)/4)*10,40)
    print(f"PASS: {len(lessons)} integrated lessons, {words:,} lesson words "
          f"(shortest {min_words}), {len(params)} control cards, {len(datasets)} data cards, "
          f"{len(experiments)} experiment contracts, {len(sources)} sources, "
          f"{links} local links, acyclic prerequisites and selected arithmetic.")
    print("This validator checks authored content only. Runtime checks are in the repository tests directory.")

if __name__=="__main__":
    main()
