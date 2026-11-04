"use strict";

const LabLearning = (() => {
  const $=id=>document.getElementById(id);
  const make=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const slug=text=>text.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu,"").trim().replace(/\s+/g,"-");
  let index,navigate,stage="basics",selectedLesson=null,selectedQuestion=null,serial=0,questionSerial=0;
  const cache=new Map(),remembered={},scenes=[];
  const questionCache=new Map(),questionButtons=new Map(),questionGroups=new Map();
  const stages=["basics","ml","dl","language","nlp","training","post"];
  const interviewPages=["interview","ml-interview","dl-interview","nlp-interview"];
  const modelFigurePlan={ML05:{"1":"selection"},ML08:{"1":"ensembles"},ML12:{"1":"pca"},DL02:{"1":"backprop"},DL05:{"1":"convolution"},DL07:{"1":"recurrence"}};
  function setPage(page,fromReader=false){
    if(!fromReader){serial++;questionSerial++;} // Stale reads must not undo later navigation.
    if(interviewPages.includes(page))serial++;
    if(stages.includes(page))questionSerial++;
    for(const name of stages){
      const menu=$("course-menu-"+name),toggle=$("course-toggle-"+name);
      if(stages.includes(page)||interviewPages.includes(page)){menu.hidden=name!==page;toggle.setAttribute("aria-expanded",String(name===page));}
      for(const button of menu.children){
        const active=page===name&&button.dataset.lesson===selectedLesson;
        button.classList.toggle("active",active);
        if(active)button.setAttribute("aria-current","page");else button.removeAttribute("aria-current");
      }
    }
    if(interviewPages.includes(page)||stages.includes(page)){
      for(const p of interviewPages){const prefix=p==="interview"?"":p.split("-")[0]+"-";$(prefix+"interview-menu").hidden=page!==p;$(prefix+"interview-toggle").setAttribute("aria-expanded",String(page===p));}
    }
    const current=index?.questions.find(q=>q.id===selectedQuestion);
    for(const [category,group] of questionGroups){
      if(interviewPages.includes(page)){group.menu.hidden=category!==current?.category;group.toggle.setAttribute("aria-expanded",String(!group.menu.hidden));}
    }
    for(const [id,button] of questionButtons){
      const active=interviewPages.includes(page)&&id===selectedQuestion;button.classList.toggle("active",active);
      if(active)button.setAttribute("aria-current","page");else button.removeAttribute("aria-current");
    }
  }
  function toggleMenu(name){
    const prefix=name==="interview"?"":name.split("-")[0]+"-",isQuestion=interviewPages.includes(name);
    const menu=$(isQuestion?prefix+"interview-menu":"course-menu-"+name);menu.hidden=!menu.hidden;
    $(isQuestion?prefix+"interview-toggle":"course-toggle-"+name).setAttribute("aria-expanded",String(!menu.hidden));
  }
  async function get(url){const r=await fetch(url),d=await r.json();if(!r.ok)throw Error(d.error||"Could not read this lesson.");return d;}
  function hide(){scenes.splice(0).forEach(s=>s?.destroy());}
  function formula(node,tex,display=false){
    if(typeof katex!=="undefined"){
      try{katex.render(tex,node,{displayMode:display,output:"mathml",trust:false,throwOnError:true,strict:"ignore"});return;}catch(error){node.className="math-error";node.textContent="Equation: "+tex;node.setAttribute("aria-label","Equation could not be typeset: "+tex);return;}
    }
    node.textContent=tex;node.className="math-error";
  }
  function localLink(target,label,source){
    if(/^https?:\/\//i.test(target)){const a=make("a",label);a.href=target;a.target="_blank";a.rel="noopener noreferrer";return a;}
    let url;try{url=new URL(target,"https://course.invalid/"+source);}catch{return make("span",label);}
    if(url.origin!=="https://course.invalid")return make("span",label);
    const match=[...(index?.lessons||[]),...(index?.resources||[])].find(l=>l.source===decodeURIComponent(url.pathname).slice(1));
    if(!match)return make("span",label);
    const a=make("a",label);a.href="#lesson-"+match.id;
    a.onclick=e=>{e.preventDefault();openDocument(match.id,decodeURIComponent(url.hash.slice(1)));};return a;
  }
  function inline(node,text,source){
    const pattern=/(`[^`]+`|\\\([\s\S]*?\\\)|\$[^$\n]+\$|\*\*[^*]+\*\*|!?\[[^\]]+\]\([^\s)]+\))/g;let last=0;
    for(const m of text.matchAll(pattern)){
      if(m.index>last)node.append(document.createTextNode(text.slice(last,m.index)));
      const t=m[0];
      if(t.startsWith("`"))node.append(make("code",t.slice(1,-1)));
      else if(t.startsWith("\\(" )||t.startsWith("$")){const span=make("span");formula(span,t.startsWith("$")?t.slice(1,-1):t.slice(2,-2));node.append(span);}
      else if(t.startsWith("**"))node.append(make("strong",t.slice(2,-2)));
      else{const p=t.match(/^(!?)\[([^\]]+)\]\(([^)]+)\)$/);node.append(p[1]?make("span",p[2]):localLink(p[3],p[2],source));}
      last=m.index+t.length;
    }
    if(last<text.length)node.append(document.createTextNode(text.slice(last)));
  }
  function markdown(text,target,source="",lessonId=null){
    target.replaceChildren();const lines=text.split("\n");let i=0,pendingFigure=null;
    const figurePlan=lessonId&&typeof LabVisuals!=="undefined"?LabVisuals.figuresForLesson(lessonId):null;
    function placeFigure(){
      if(!pendingFigure)return;
      const host=make("div");target.append(host);scenes.push(pendingFigure.explorer?ArchitectureExplorer.mount(host,pendingFigure.explorer):pendingFigure.model?ModelFigures.mount(host,pendingFigure.model):LabVisuals.mount(host,pendingFigure));pendingFigure=null;
    }
    while(i<lines.length){
      const line=lines[i];if(!line.trim()){i++;continue;}
      if(/^\s*\\\[/.test(line)||line.trim()==="$$"){
        const marker=line.trim()==="$$"?"$$":"\\]",body=[];
        let first=line.trim()==="$$"?"":line.replace(/^\s*\\\[/,"");i++;
        if(first.includes(marker)){const tex=first.slice(0,first.indexOf(marker));const block=make("div",undefined,"math-block");formula(block,tex,true);target.append(block);continue;}
        if(first)body.push(first);
        while(i<lines.length&&!lines[i].includes(marker))body.push(lines[i++]);
        if(i<lines.length){const prefix=lines[i].split(marker)[0];if(prefix)body.push(prefix);i++;}
        const block=make("div",undefined,"math-block");formula(block,body.join("\n"),true);target.append(block);continue;
      }
      if(line.startsWith("```")){
        const language=line.slice(3).trim(),body=[];i++;while(i<lines.length&&!lines[i].startsWith("```"))body.push(lines[i++]);i++;
        if(language==="mermaid")continue;
        if(["math","latex","tex"].includes(language)){const block=make("div",undefined,"math-block");formula(block,body.join("\n"),true);target.append(block);continue;}
        const pre=make("pre"),code=make("code",body.join("\n"));pre.append(code);target.append(pre);placeFigure();continue;
      }
      if(/^#{1,6} /.test(line)){
        placeFigure();
        const title=line.replace(/^#+ /,""),level=line.match(/^#+/)[0].length;
        if(lessonId&&level===1){i++;continue;} // The page H1 already names this full lesson.
        const h=make("h"+Math.min(6,lessonId?level:level+1));h.id="lesson-"+slug(title);inline(h,title,source);target.append(h);
        if(line.startsWith("## ")){const number=title.match(/^(\d+)\./)?.[1],model=modelFigurePlan[lessonId]?.[number],explorer=typeof ArchitectureExplorer!=="undefined"?(ArchitectureExplorer.placement[lessonId]?.[number]||(typeof ConceptLessons!=="undefined"?ConceptLessons.placement[lessonId]?.[number]:null)):null;pendingFigure=explorer?{explorer}:model?{model}:figurePlan?.[number]||null;}
        i++;continue;
      }
      if(i+1<lines.length&&line.includes("|")&&/^\s*\|?\s*:?-{3}/.test(lines[i+1])){
        const cells=row=>row.trim().replace(/^\||\|$/g,"").split(/(?<!\\)\|/).map(c=>c.trim().replace(/\\\|/g,"|"));
        const wrap=make("div",undefined,"table-scroll"),table=make("table"),head=make("thead"),hr=make("tr"),body=make("tbody");
        cells(line).forEach(c=>{const th=make("th");inline(th,c,source);hr.append(th);});head.append(hr);i+=2;
        while(i<lines.length&&lines[i].trim()&&lines[i].includes("|")){const tr=make("tr");cells(lines[i++]).forEach(c=>{const td=make("td");inline(td,c,source);tr.append(td);});body.append(tr);}
        table.append(head,body);wrap.append(table);target.append(wrap);continue;
      }
      if(/^\s*(?:[-*] |\d+\. )/.test(line)){
        const list=make(/^\s*\d+\. /.test(line)?"ol":"ul");
        while(i<lines.length&&/^\s*(?:[-*] |\d+\. )/.test(lines[i])){const li=make("li");inline(li,lines[i++].replace(/^\s*(?:[-*] |\d+\. )/,""),source);list.append(li);}target.append(list);continue;
      }
      if(/^>/.test(line)){const body=[];while(i<lines.length&&/^>/.test(lines[i]))body.push(lines[i++].replace(/^> ?/,""));const q=make("blockquote");inline(q,body.join(" "),source);target.append(q);continue;}
      if(/^---+$/.test(line.trim())){target.append(make("hr"));i++;continue;}
      const p=make("p"),body=[line];i++;
      while(i<lines.length&&lines[i].trim()&&!/^(?:#|```|>|\s*[-*] |\s*\d+\. |\s*\\\[|\$\$)/.test(lines[i])&&!(lines[i].includes("|")&&/^\s*\|?\s*:?-{3}/.test(lines[i+1]||"")))body.push(lines[i++]);
      const prose=body.join(" ");
      if(prose.startsWith("**Key terms.**"))p.className="topic-terms";
      if(prose.startsWith("**Process.**"))p.className="topic-flow";
      if(prose.startsWith("**Calculation.**"))p.className="topic-calculation";
      inline(p,prose,source);target.append(p);
      if(p.className!=="topic-terms")placeFigure();
    }
    placeFigure();
  }
  function renderLecture(doc,target){
    target.replaceChildren();
    if(!doc.reader){markdown(doc.markdown,target,doc.source);return;}
    const host=make("div",undefined,"lesson-explorer-host");target.append(host);
    const scene=ArchitectureExplorer.mount(host,doc.reader.explorer,{showHeading:false,notes:doc.reader.notes,source:doc.source,sources:doc.reader.sources,reading:doc.reader.reading});
    scenes.push(scene);
    if(doc.reader.related?.length){
      const related=make("nav",undefined,"lesson-related");related.setAttribute("aria-label","Related explanations");
      related.append(make("span","Related: "));
      for(const id of doc.reader.related){const title=index?.lessons.find(l=>l.id===id)?.title||id,b=make("button",title,"text-button");b.type="button";b.onclick=()=>openDocument(id);related.append(b);}
      target.append(related);
    }
    return scene;
  }
  async function openMetric(family="classification"){
    await openDocument("P8");
    const scene=scenes.find(s=>s.root?.dataset.architecture==="P8");scene?.configure({family});
  }
  async function openDocument(id,anchor=""){
    const request=++serial;$("course-error").textContent="";
    try{
      const doc=cache.get(id)||await get("/api/learning/document/"+encodeURIComponent(id));if(request!==serial)return;
      cache.set(id,doc);stage=doc.stage;remembered[stage]=id;selectedLesson=id;hide();navigate(stage,true);
      $("page-title").textContent=doc.title;document.title=doc.title+" · Machine Learning Study Lab";
      renderLecture(doc,$("course-content"));
      const destination=anchor?$("lesson-"+anchor):$("page-title");
      (destination||$("page-title")).scrollIntoView({block:"start"});
    }catch(error){if(request===serial)$("course-error").textContent=error.message;}
  }
  async function showStage(s){stage=s;const id=remembered[s]||index.lessons.find(l=>l.stage===s).id;await openDocument(id);}
  async function openQuestion(id){
    const request=++questionSerial;
    try{
      const q=questionCache.get(id)||await get("/api/learning/interview/"+encodeURIComponent(id));if(request!==questionSerial)return;
      questionCache.set(id,q);selectedQuestion=id;navigate(["ml","dl","nlp"].includes(q.category)?q.category+"-interview":"interview",true);
      const introductions={ml:"These questions use familiar data problems: leakage, rare events, model choice and deployment failures. Try a short answer first, then use the example and follow-up to check your reasoning.",dl:"These questions connect neural-network ideas to debugging and model choices. Explain the immediate problem first; use the equations and follow-ups when they help justify your answer.",nlp:"These scenarios cover text classification, retrieval, extraction and evaluation. Start with what the user needs and what the model can read. The course has numerical NLP examples, not a dedicated NLP benchmark-training lab."};
      $("interview-intro").textContent=introductions[q.category]||"These questions cover language-model training, applications and system design. The answers are examples of sound reasoning, not scripts or claims about your experience. Describe what you actually built, measured or tested.";
      $("question-title").textContent=q.title;markdown(q.markdown,$("question-answer"),q.source);$("question-error").textContent="";
    }catch(e){if(request===questionSerial)$("question-error").textContent=e.message;}
  }
  function showInterview(scope="llm"){const match=q=>scope==="llm"?!["ml","dl","nlp"].includes(q.category):q.category===scope;return openQuestion(index.questions.find(q=>q.id===selectedQuestion&&match(q))?.id||index.questions.find(match).id);}
  function renderSiteMap(){
    const groups=[
      {title:"Shared foundations",stages:["basics"]},
      {title:"Machine learning",stages:["ml"],scope:"ml",tools:[["ml-lab","Training lab"],["ml-results","Saved results"],["ml-compare","Compare models"]]},
      {title:"Deep learning",stages:["dl"],scope:"dl",tools:[["dl-lab","Training lab"],["dl-results","Saved results"],["dl-compare","Compare models"]]},
      {title:"Natural language processing",stages:["language","nlp"],scope:"nlp"},
      {title:"Large language models",stages:["training","post"],scope:"llm",tools:[["visuals","Model diagrams"],["train","Train a model"],["explore","Explore results"],["compare","Compare experiments"]]}
    ];
    const pageButton=(page,label)=>{const b=make("button",label,"text-button");b.type="button";b.dataset.mapPage=page;b.onclick=()=>navigate(page);return b;};
    $("home-sitemap").replaceChildren(...groups.map(group=>{
      const card=make("div",undefined,"map-subject");card.append(make("h3",group.title));
      for(const stage of group.stages){
        if(group.stages.length>1)card.append(make("h4",index.stages[stage].title));
        const list=make("ul",undefined,"map-lessons");
        for(const lesson of index.lessons.filter(l=>l.stage===stage)){
          const item=make("li"),details=make("details",undefined,"map-lesson"),summary=make("summary",lesson.title);
          const open=make("button","Open lesson","text-button");open.type="button";open.dataset.mapLesson=lesson.id;open.onclick=()=>openDocument(lesson.id);
          details.append(summary,open);
          const topics=make("ul",undefined,"map-topics");
          for(const title of lesson.topics)topics.append(make("li",title));
          details.append(topics);item.append(details);list.append(item);
        }
        card.append(list);
      }
      if(group.tools){const tools=make("div",undefined,"map-tools");tools.setAttribute("aria-label",group.title+" tools");for(const [page,label] of group.tools)tools.append(pageButton(page,label));card.append(tools);}
      if(group.scope){
        const questions=make("details",undefined,"map-questions");questions.append(make("summary","Interview questions"));
        const list=make("ul",undefined,"map-question-list");
        for(const q of index.questions.filter(q=>group.scope==="llm"?!["ml","dl","nlp"].includes(q.category):q.category===group.scope)){
          const li=make("li"),b=make("button",q.title,"text-button");b.type="button";b.dataset.mapQuestion=q.id;b.onclick=()=>openQuestion(q.id);li.append(b);list.append(li);
        }
        questions.append(list);card.append(questions);
      }
      return card;
    }));
  }
  async function init(go){
    navigate=go;index=await get("/api/learning");
    renderSiteMap();
    for(const name of stages){
      const buttons=index.lessons.filter(l=>l.stage===name).map(lesson=>{
        const button=make("button",lesson.title,"lesson-link");button.type="button";
        button.dataset.lesson=lesson.id;button.onclick=()=>openDocument(lesson.id);return button;
      });
      $("course-menu-"+name).replaceChildren(...buttons);
    }
    $("interview-menu").replaceChildren();
    $("ml-interview-menu").replaceChildren();$("dl-interview-menu").replaceChildren();$("nlp-interview-menu").replaceChildren();
    for(const [category,label] of new Map(index.questions.map(q=>[q.category,q.category_label]))){
      if(["ml","dl","nlp"].includes(category)){
        for(const q of index.questions.filter(q=>q.category===category)){const button=make("button",q.title,"lesson-link question-link");button.type="button";button.dataset.question=q.id;button.onclick=()=>openQuestion(q.id);$(category+"-interview-menu").append(button);questionButtons.set(q.id,button);}continue;
      }
      const wrapper=make("div",undefined,"question-group"),toggle=make("button",label,"course-toggle topic-toggle"),menu=make("div",undefined,"course-submenu question-list");
      toggle.type="button";toggle.id="question-topic-"+category;menu.id="question-menu-"+category;menu.hidden=true;
      toggle.setAttribute("aria-controls",menu.id);toggle.setAttribute("aria-expanded","false");menu.setAttribute("role","group");menu.setAttribute("aria-label",label+" questions");
      toggle.onclick=()=>{menu.hidden=!menu.hidden;toggle.setAttribute("aria-expanded",String(!menu.hidden));};
      for(const q of index.questions.filter(q=>q.category===category)){
        const button=make("button",q.title,"lesson-link question-link");button.type="button";button.dataset.question=q.id;button.onclick=()=>openQuestion(q.id);menu.append(button);questionButtons.set(q.id,button);
      }
      wrapper.append(toggle,menu);$("interview-menu").append(wrapper);questionGroups.set(category,{toggle,menu});
    }
  }
  return {init,showStage,openDocument,openMetric,renderLecture,openQuestion,showInterview,markdown,hide,formula,setPage,toggleMenu};
})();
