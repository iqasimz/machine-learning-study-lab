"use strict";

// Local practice history only. No score here is a qualification or server-side grade.
const CourseProgress=(()=>{
  const KEY="model-lab.interactive.v1";
  const fresh=()=>({version:1,cursors:{},visited:{},responses:{}});
  const record=v=>v&&typeof v==="object"&&!Array.isArray(v);
  function load(storage){
    try{const raw=storage.getItem(KEY);if(!raw)return {data:fresh(),warning:""};if(raw.length>2_000_000)throw Error("oversized");const v=JSON.parse(raw);if(v.version!==1||![v.cursors,v.visited,v.responses].every(record))throw Error("invalid");
      const data=fresh();for(const [k,r] of Object.entries(v.cursors))if(["basics","python","ml","dl","nlp","llm"].includes(k)&&record(r)&&typeof r.lesson==="string"&&typeof r.step==="string")data.cursors[k]={lesson:r.lesson,step:r.step};
      for(const [k,r] of Object.entries(v.visited))if(/^[A-Z0-9]+:[a-z0-9-]+$/.test(k)&&record(r)&&/^[a-f0-9]{64}$/.test(r.fingerprint))data.visited[k]={fingerprint:r.fingerprint};
      for(const [k,r] of Object.entries(v.responses))if(/^[A-Z0-9]+:check-[ab]$/.test(k)&&record(r)&&/^[a-f0-9]{64}$/.test(r.fingerprint)&&["correct","incorrect","skipped"].includes(r.status)&&Number.isInteger(r.attempts)&&r.attempts>=0&&r.attempts<=100000&&[r.first,r.last].every(x=>x===null||Number.isInteger(x)&&x>=0&&x<4)&&(r.status==="skipped"||r.attempts>0&&r.first!==null&&r.last!==null))data.responses[k]={fingerprint:r.fingerprint,status:r.status,first:r.first,last:r.last,attempts:r.attempts,everWrong:!!r.everWrong,everSkipped:!!r.everSkipped};
      return {data,warning:""};
    }catch{return {data:fresh(),warning:"Saved progress is unavailable or unreadable. You can continue in this tab; no training data or run was changed."};}
  }
  function save(storage,data){try{storage.setItem(KEY,JSON.stringify(data));return "";}catch{return "Progress could not be saved. This session still works, but closing or reloading may lose its history.";}}
  function response(data,step){const r=data.responses[step.id];return r?.fingerprint===step.fingerprint?r:null;}
  function mark(data,step){data.visited[step.id]={fingerprint:step.fingerprint};}
  function answer(data,step,choice){
    if(step.kind!=="question"||!Number.isInteger(choice)||choice<0||choice>=step.question.options.length)throw Error("Choose one answer.");
    const old=response(data,step),correct=choice===step.question.answer;
    data.responses[step.id]={fingerprint:step.fingerprint,status:correct?"correct":"incorrect",first:old?.first??choice,last:choice,attempts:Math.min(100000,(old?.attempts||0)+1),everWrong:!!old?.everWrong||!correct,everSkipped:!!old?.everSkipped};mark(data,step);return data.responses[step.id];
  }
  function skip(data,step){if(step.kind!=="question")throw Error("Only questions can be skipped.");const old=response(data,step);data.responses[step.id]={fingerprint:step.fingerprint,status:"skipped",first:old?.first??null,last:old?.last??null,attempts:old?.attempts||0,everWrong:!!old?.everWrong,everSkipped:true};mark(data,step);}
  function stats(data,path,units){
    const steps=path.lessons.flatMap(l=>units[l.id].steps),questions=steps.filter(s=>s.kind==="question");
    const seen=steps.filter(s=>data.visited[s.id]?.fingerprint===s.fingerprint).length;
    const correct=questions.filter(s=>response(data,s)?.status==="correct").length;
    const review=questions.filter(s=>["incorrect","skipped"].includes(response(data,s)?.status));
    const firstCorrect=questions.filter(s=>{const r=response(data,s);return r?.status==="correct"&&!r.everWrong&&!r.everSkipped&&r.attempts>=1;}).length;
    return {seen,total:steps.length,correct,questions:questions.length,review,firstCorrect};
  }
  function migrate(data,index){
    for(const id of ["ml","dl","nlp","llm"]){
      const old=data.cursors[id];
      if(old&&index.shared_foundations.includes(old.lesson)){data.cursors.basics??=old;delete data.cursors[id];}
      else if(old&&!index.paths.find(p=>p.id===id)?.lessons.some(l=>l.id===old.lesson)){
        const owner=index.paths.find(p=>p.lessons.some(l=>l.id===old.lesson));
        if(owner)data.cursors[owner.id]??=old;
        delete data.cursors[id];
      }
    }
    return data;
  }
  return {fresh,load,save,response,mark,answer,skip,stats,migrate};
})();

const GuidedCourse=(()=>{
  const $=id=>document.getElementById(id);
  const make=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  let index,navigate,storage,state,storageWarning="",path=null,deck=null,lessonIndex=0,stepIndex=0,serial=0,active=false,selected=null;
  const cache=new Map(),scenes=[];
  async function get(url){const r=await fetch(url),d=await r.json();if(!r.ok)throw Error(d.error||"Could not load the interactive course.");return d;}
  function persist(){const warning=CourseProgress.save(storage,state);if(warning)storageWarning=warning;$("interactive-storage").textContent=storageWarning;}
  function clean(){scenes.splice(0).forEach(s=>s?.destroy());LabLearning.hide();}
  function hide(){active=false;serial++;clean();}
  function counts(p){return CourseProgress.stats(state,p,index.units);}
  function summary(p){const s=counts(p);return `${s.seen} / ${s.total} steps visited · ${s.correct} / ${s.questions} checks currently correct · ${s.review.length} to review`;}
  function selector(){
    serial++;clean();$("interactive-loading").hidden=true;$("interactive-picker").hidden=false;$("interactive-journey").hidden=true;$("interactive-complete").hidden=true;
    $("page-title").textContent="Interactive mode";$("page-context").textContent="Guided courses";$("interactive-storage").textContent=storageWarning;
    $("interactive-paths").replaceChildren(...index.paths.map(p=>{
      const card=make("section",undefined,"path-card"),name=make("h2",p.title),description=make("p",p.description),s=counts(p);
      if(p.id==="basics")card.dataset.common="true";
      const details=make("p",p.lessons.length+" lessons · "+s.questions+" practice checks","plot-help");
      const progress=make("p",summary(p),"path-progress"),button=make("button",state.cursors[p.id]?"Continue "+p.title:"Begin "+p.title,"primary");button.dataset.path=p.id;button.onclick=()=>enter(p.id);
      card.append(name,description,details,progress,button);return card;
    }));
  }
  function show(){active=true;serial++;if(!index){$("interactive-error").textContent="Interactive content is still loading. Please try again in a moment.";return;}selector();}
  async function enter(id){
    path=index.paths.find(p=>p.id===id);if(!path)return;
    const cursor=state.cursors[id];let chosen=cursor?path.lessons.findIndex(l=>l.id===cursor.lesson):0;if(chosen<0)chosen=0;
    await open(chosen,cursor?.step);
  }
  async function open(chapter,stepId){
    if(!path||chapter<0||chapter>=path.lessons.length)return;
    const request=++serial;$("interactive-error").textContent="";$("interactive-loading").hidden=false;
    try{const id=path.lessons[chapter].id,loaded=cache.get(id)||await get("/api/course/lesson/"+encodeURIComponent(id));if(request!==serial||!active)return;
      cache.set(id,loaded);deck=loaded;lessonIndex=chapter;index.units[id]={title:loaded.title,fingerprint:loaded.fingerprint,steps:loaded.steps.map(s=>({id:s.id,kind:s.kind,title:s.title,fingerprint:s.fingerprint}))};const target=loaded.steps.findIndex(s=>s.id===stepId);stepIndex=target>=0?target:0;
      state.cursors[path.id]={lesson:deck.id,step:deck.steps[stepIndex].id};persist();render();
    }catch(e){if(request===serial)$("interactive-error").textContent=e.message;}finally{if(request===serial)$("interactive-loading").hidden=true;}
  }
  function outline(){
    const host=$("interactive-outline");host.replaceChildren(...path.lessons.map((lesson,i)=>{
      const item=make("div",undefined,"journey-map-lesson"),button=make("button",`${i+1}. ${lesson.title}`,"journey-lesson-link");button.onclick=()=>open(i);button.setAttribute("aria-current",i===lessonIndex?"step":"false");
      const meta=index.units[lesson.id],seen=meta.steps.filter(s=>state.visited[s.id]?.fingerprint===s.fingerprint).length;
      item.append(button,make("small",`${seen} / ${meta.steps.length} steps visited`));
      if(i===lessonIndex){const list=make("div",undefined,"journey-topics");deck.steps.forEach((s,j)=>{const b=make("button",s.title,"journey-topic-link");b.onclick=()=>move(j);b.setAttribute("aria-current",j===stepIndex?"step":"false");list.append(b);});item.append(list);}return item;
    }));
  }
  function feedback(step,r){
    const host=$("interactive-feedback");host.replaceChildren();if(!r)return;
    const q=step.question,label=r.status==="skipped"?"Skipped — available for review.":r.status==="correct"?(r.everWrong||r.everSkipped?"Correct after review.":"Correct."):"Not quite — compare the reasoning below.";
    host.append(make("strong",label),make("p",r.status==="skipped"?q.explanation:q.options[r.last].feedback));
    if(r.status==="incorrect")host.append(make("p",q.explanation));
    const alternatives=make("details");alternatives.append(make("summary","Reasoning for all four choices"));const list=make("ul");q.options.forEach((o,i)=>{const li=make("li");li.append(make("strong",(i===q.answer?"Correct: ":"")+o.text),make("p",o.feedback));list.append(li);});alternatives.append(list);host.append(alternatives);
  }
  function question(step){
    const host=$("interactive-body"),q=step.question,r=CourseProgress.response(state,step),fieldset=make("fieldset",undefined,"course-question"),legend=make("legend",q.prompt);
    selected=r?.last??null;fieldset.append(legend);
    q.options.forEach((o,i)=>{const label=make("label",undefined,"course-option"),radio=make("input");radio.type="radio";radio.name="interactive-choice";radio.id="interactive-choice-"+i;radio.value=String(i);radio.checked=selected===i;
      radio.onchange=()=>{selected=i;$("interactive-check").disabled=false;};label.append(radio,make("span",o.text));fieldset.append(label);});
    host.append(make("p","Choose one answer, then check the reasoning. You can skip or retry; this is practice, not a graded exam.","plot-help"),fieldset);
    $("interactive-question-actions").hidden=false;$("interactive-check").disabled=selected===null;$("interactive-next").disabled=!r;
    feedback(step,r);
  }
  function render(){
    if(!active||!deck)return;clean();const step=deck.steps[stepIndex];state.cursors[path.id]={lesson:deck.id,step:step.id};persist();
    $("interactive-picker").hidden=true;$("interactive-journey").hidden=false;$("interactive-complete").hidden=true;
    $("page-title").textContent=deck.title;$("page-context").textContent=path.title+" · Interactive course";document.title=deck.title+" · Machine Learning Study Lab";
    $("interactive-position").textContent=`Lesson ${lessonIndex+1} of ${path.lessons.length} · Step ${stepIndex+1} of ${deck.steps.length}`;
    $("interactive-kind").textContent={lesson:"Explore the lesson",question:"Check your understanding",lab:"Apply in the lab"}[step.kind]||"";
    $("interactive-progress-text").textContent=summary(path);const stats=counts(path);$("interactive-progress").value=stats.seen/Math.max(1,stats.total);
    $("interactive-body").replaceChildren();$("interactive-feedback").replaceChildren();$("interactive-question-actions").hidden=true;
    $("interactive-next").disabled=false;$("interactive-back").disabled=lessonIndex===0&&stepIndex===0;
    $("interactive-step-title").textContent=step.kind==="question"?step.title:"";$("interactive-step-title").hidden=step.kind!=="question";
    if(step.kind==="question")question(step);else{
      if(step.kind==="lesson")LabLearning.renderLecture(step.document,$("interactive-body"));
      else LabLearning.markdown(step.markdown,$("interactive-body"),deck.source);
      if(step.kind==="lab"){const b=make("button","Open the relevant training lab","secondary");b.onclick=()=>LabLearning.openLab(step.lab.page,step.lab.mode);$("interactive-body").append(b);}

    }
    const nextChapter=path.lessons[lessonIndex+1],hint=step.next_hint|| (nextChapter?"Next lesson: "+nextChapter.title:"Finish this path and review the remaining questions.");
    $("interactive-next-hint").textContent=hint;$("interactive-next").textContent=stepIndex===deck.steps.length-1?(nextChapter?"Next lesson":"Finish path"):"Next";
    outline();reviewList();$("interactive-body").setAttribute("tabindex","-1");$("interactive-body").focus?.({preventScroll:true});$("page-title").scrollIntoView({block:"start"});
  }
  function move(position){if(!deck||position<0||position>=deck.steps.length)return;stepIndex=position;render();}
  async function next(){
    if(!deck)return;const step=deck.steps[stepIndex];if(step.kind==="question"&&!CourseProgress.response(state,step))return;
    CourseProgress.mark(state,step);persist();
    if(stepIndex+1<deck.steps.length)move(stepIndex+1);else if(lessonIndex+1<path.lessons.length)await open(lessonIndex+1);else complete();
  }
  async function back(){if(stepIndex>0)move(stepIndex-1);else if(lessonIndex>0){const id=path.lessons[lessonIndex-1].id,meta=index.units[id];await open(lessonIndex-1,meta.steps.at(-1).id);}}
  function reviewList(){
    if(!path)return;const stats=counts(path),host=$("interactive-review-list");$("interactive-review-count").textContent=`Review questions (${stats.review.length})`;
    host.replaceChildren(...stats.review.map(s=>{const id=s.id.split(":")[0],i=path.lessons.findIndex(l=>l.id===id),b=make("button",index.units[id].title+" · "+s.title,"journey-topic-link");b.onclick=()=>open(i,s.id);return b;}));
    if(!stats.review.length)host.append(make("p","No checked-wrong or skipped questions in this path. Unattempted checks are still unattempted, not passes.","plot-help"));
  }
  function complete(){
    clean();$("interactive-journey").hidden=true;$("interactive-picker").hidden=true;$("interactive-complete").hidden=false;$("page-title").textContent=path.title+" path";
    const s=counts(path),host=$("interactive-complete-content");host.replaceChildren(make("h2","You reached the end of this path"),make("p",summary(path)),make("p",`${s.firstCorrect} checks are currently correct on their first recorded attempt. A correct retry does not erase an earlier wrong answer. Visiting steps and answering checks are useful practice records, not proof of independent skill.`));
    const missed=path.lessons.findIndex(l=>index.units[l.id].steps.some(step=>state.visited[step.id]?.fingerprint!==step.fingerprint));
    if(missed>=0){const b=make("button","Return to an unvisited step","secondary");b.onclick=()=>{const l=path.lessons[missed],s=index.units[l.id].steps.find(step=>state.visited[step.id]?.fingerprint!==step.fingerprint);open(missed,s.id);};host.append(b);}
    if(s.review.length){const b=make("button","Review the first missed check","primary");b.onclick=()=>{const step=s.review[0],i=path.lessons.findIndex(l=>l.id===step.id.split(":")[0]);open(i,step.id);};host.append(b);}
    host.append(make("p","Explain a complete example without the notes, run a controlled experiment when you are ready, and return to the full course for deeper review. The other paths remain available."));
  }
  async function init(go){
    navigate=go;index=await get("/api/course/catalog");try{storage=window.localStorage;}catch{storage=null;}
    const loaded=CourseProgress.load(storage);state=loaded.data;storageWarning=loaded.warning;
    CourseProgress.migrate(state,index);
    $("home-interactive").disabled=false;$("nav-interactive").disabled=false;
    $("interactive-back").onclick=back;$("interactive-next").onclick=next;
    $("interactive-check").onclick=()=>{try{const step=deck.steps[stepIndex],r=CourseProgress.answer(state,step,selected);persist();feedback(step,r);$("interactive-next").disabled=false;$("interactive-progress-text").textContent=summary(path);const s=counts(path);$("interactive-progress").value=s.seen/Math.max(1,s.total);outline();reviewList();}catch(e){$("interactive-error").textContent=e.message;}};
    $("interactive-skip").onclick=()=>{const step=deck.steps[stepIndex];CourseProgress.skip(state,step);persist();render();};
    $("interactive-choose-path").onclick=selector;$("interactive-finish-paths").onclick=selector;
    $("interactive-full-lesson").onclick=()=>LabLearning.openDocument(deck.id);
  }
  return {init,show,hide};
})();
