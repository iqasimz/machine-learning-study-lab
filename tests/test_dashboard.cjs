// Dependency-free behaviour checks; this stand-in does not certify browser layout.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),cp=require('node:child_process');
const root=path.resolve(__dirname,'..'),web=path.join(root,'src/training_lab/web');
const python=process.env.LAB_PYTHON||'python3';
const readPython=code=>{const r=cp.spawnSync(python,['-c',"import sys,json; sys.path.insert(0,'src'); "+code],{cwd:root,encoding:'utf8',maxBuffer:12*1024*1024});if(r.status!==0)throw Error(r.stderr);return JSON.parse(r.stdout);};
const catalog=readPython('from training_lab.experiments import catalog; print(json.dumps(catalog()))');
const adaptation=readPython('from training_lab.adaptation import catalog; print(json.dumps(catalog()))');
const modelsCatalog=readPython('from training_lab.model_lab import catalog; print(json.dumps(catalog()))');
const modelsBundle=readPython(`from training_lab import model_lab as m
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as tmp:
 specs=[('ui-classic','ml','classification','iris','logistic'),('ui-classic-tree','ml','classification','iris','tree'),('ui-regression','ml','regression','curve','ridge'),('ui-regression-tree','ml','regression','curve','tree'),('ui-cluster','ml','clustering','clusters','kmeans'),('ui-cluster-density','ml','clustering','clusters','dbscan'),('ui-pca','ml','representation','digits','pca'),('ui-anomaly','ml','anomaly','clusters','isolation'),('ui-anomaly-b','ml','anomaly','clusters','isolation'),('ui-neural','dl','classification','digits','mlp'),('ui-neural-cnn','dl','classification','digits','cnn'),('ui-reconstruct','dl','representation','digits','autoencoder')]
 result={}
 for identifier,area,task,dataset,model in specs:
  m.fit(identifier,dict(area=area,task=task,dataset=dataset,model=model,controls=dict(train_limit=500,**(dict(epochs=1) if area=='dl' else {}))),Path(tmp))
  result[identifier]=m.detail(identifier,Path(tmp))
 pairs=[['ui-classic','ui-classic-tree'],['ui-regression','ui-regression-tree'],['ui-cluster','ui-cluster-density'],['ui-pca','ui-reconstruct'],['ui-anomaly','ui-anomaly-b'],['ui-neural','ui-neural-cnn']]
 print(json.dumps(dict(runs=result,comparisons={'|'.join(sorted(p)):m.compare(p,Path(tmp)) for p in pairs})))`);
const modelsFixtures=modelsBundle.runs;
const course=readPython('from training_lab import curriculum as c; i=c.catalog(); print(json.dumps(dict(index=i,docs={r["id"]:c.document(r["id"]) for r in i["lessons"]+i["resources"]},questions={r["id"]:c.interview(r["id"]) for r in i["questions"]})))');
const guided=readPython('from training_lab import guided_course as g; i=g.catalog(); print(json.dumps(dict(index=i,docs={id:g.lesson(id) for id in i["units"]})))');
const katex=require(path.join(web,'vendor/katex/katex.min.js'));
const html=fs.readFileSync(path.join(web,'index.html'),'utf8');
const measured={};
for(const id of ['coherent-language-first','training-lab-v01']){
 const directory=path.join(root,'runs',id);if(!fs.existsSync(directory))continue;
 const item={integrity:'PASS',can_resume:false};for(const n of ['manifest','metrics','data','tokenizer','evaluation','inspection','samples','baseline']){const p=path.join(directory,n+'.json');if(fs.existsSync(p))item[n]=JSON.parse(fs.readFileSync(p,'utf8'));}measured[id]=item;
}
// Isolated UI fixtures must not depend on a user's saved runs being present.
// This object exists only in the mock fetch responses; never write it as a run.
if(!Object.keys(measured).length)measured['ui-test-fixture']={
 integrity:'PASS',can_resume:false,
 manifest:{id:'ui-test-fixture',status:'MEASURED',engine:'lora',completed_steps:10,requested_steps:10,tokens_processed:1000,elapsed_seconds:5,
  model:{parameters:361821120,trainable_parameters:819200},hardware:{device:'cpu'},settings:{},limits:['Test fixture, not measured evidence.']},
 metrics:[{step:0,dev_nll:2,tokens_processed:0},{step:10,dev_nll:1.9,sampled_train_nll:1.8,tokens_processed:1000,learning_rate:.0001}],
 data:{documents:[{id:'test-doc',split:'dev',text:'Test fixture only.'}],counts:{train:1,dev:1,test:1}},
 evaluation:{checkpoint_step:10,pretrained:{dev:{nll:2},test:{nll:2}},transformer:{dev:{nll:1.9,perplexity:Math.exp(1.9)},test:{nll:1.9,perplexity:Math.exp(1.9)}}},
 samples:[{prompt:'Why does ice melt?',stage:'Fixture output',completion:'Wrong fixture answer: sublimation.'}]
};
const settle=async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));};
const content=n=>n.textContent+' '+n.children.map(content).join(' ');
const descendants=n=>[n,...n.children.flatMap(descendants)];

async function exercise(empty){
 const elements={},all=[],requests=[],mathErrors=[];let equations=0;
 class Element{
  constructor(tag='div',namespaceURI=null){this.tag=tag;this.namespaceURI=namespaceURI;this.children=[];this.value='';this.hidden=false;this.disabled=false;this.textContent='';this.dataset={};this.listeners={};this.attributes={};this.classList={toggle(){}};all.push(this);}
  set id(v){this._id=v;elements[v]=this;}get id(){return this._id;}
  get options(){return this.children.filter(c=>c.tag==='option');}get firstChild(){return this.children[0];}get isConnected(){return true;}
  append(...items){for(const x of items){const n=typeof x==='string'?new Element('#text'):x;if(typeof x==='string')n.textContent=x;n.parent=this;this.children.push(n);}if(this.tag==='select'&&!this.value)this.value=this.options[0]?.value||'';}
  prepend(...items){for(const n of items)n.parent=this;this.children.unshift(...items);}
  replaceChildren(...items){this.children=[];this.textContent='';if(this.tag==='select')this.value='';this.append(...items);}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
  setAttribute(k,v){this.attributes[k]=String(v);if(k==='class')this.className=v;}getAttribute(k){return this.attributes[k];}removeAttribute(k){delete this.attributes[k];}
  addEventListener(k,f){(this.listeners[k]??=[]).push(f);}removeEventListener(k,f){this.listeners[k]=(this.listeners[k]||[]).filter(x=>x!==f);}
  fire(k){for(const f of [...this.listeners[k]||[]])f({key:'Enter',preventDefault(){}});}
  scrollIntoView(){}getBoundingClientRect(){return {width:600,height:270};}
  getContext(){return {...Object.fromEntries(['scale','setTransform','clearRect','fillText','beginPath','moveTo','lineTo','stroke','save','translate','rotate','restore','fillRect','arc','fill','putImageData'].map(k=>[k,()=>{}])),createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)})};}
 }
 const document=new Element('document');Object.assign(document,{hidden:false,getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag,ns),createTextNode:t=>{const n=new Element('#text');n.textContent=t;return n;},querySelectorAll:q=>q==='.page'?all.filter(e=>e.page):q==='.run-tab'?all.filter(e=>e.runTab):q==='nav button'?all.filter(e=>e.dataset.page):q.match(/^\[data-(\w+)\]$/)?all.filter(e=>q.slice(6,-1) in e.dataset):[]});
 for(const m of html.matchAll(/<(\w+)\b[^>]*\bid="([^"]+)"[^>]*>/g)){assert.ok(!elements[m[2]],'Unique IDs');const e=new Element(m[1]);e.id=m[2];e.page=/class="page"/.test(m[0]);e.runTab=/class="run-tab"/.test(m[0]);e.hidden=/\bhidden\b/.test(m[0]);for(const [k,v] of [...m[0].matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]))e.setAttribute(k,v);if(e.attributes.value)e.value=e.attributes.value;}
 for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g))for(const o of m[2].matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g)){const n=new Element('option');n.value=o[1].match(/value="([^"]*)"/)?.[1]||o[2];elements[m[1]].append(n);}
 for(const m of html.matchAll(/<button\b[^>]*data-(page|engine|tab|subject)="([^"]+)"[^>]*>/g)){const id=m[0].match(/\bid="([^"]+)"/)?.[1];const n=id?elements[id]:new Element('button');n.dataset[m[1]]=m[2];}
 const fixtureTrace={kind:'measured',tokens:['The','cat','sat'],ids:[1,2,3],embeddings:[[.1,.2],[.3,.4],[.5,.6]],attention:[[1,0,0],[.4,.6,0],[.2,.3,.5]],next_tokens:[{text:'down',probability:.3}],note:'Fixture—not a model evaluation.'};
 const stored=new Map(),localStorage={getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v)};
 const context={console,URL,document,window:{devicePixelRatio:1,localStorage,addEventListener(){}},setInterval(){return 1;},clearInterval(){},katex:{render(tex,node,options){equations++;try{const result=katex.renderToString(tex,options);assert.match(result,/<math/);node.textContent=tex;}catch(e){mathErrors.push([tex,e.message]);throw e;}}},fetch:async(url,options)=>{
  const body=options?.body?JSON.parse(options.body):undefined;requests.push({url,body});let data;
  if(url==='/api/catalog')data=catalog;else if(url==='/api/adaptation')data=adaptation;
  else if(url==='/api/models/catalog')data=modelsCatalog;
  else if(url==='/api/models/runs')data=empty?[]:Object.entries(modelsFixtures).map(([id,r])=>({id,...r.manifest}));
  else if(url.startsWith('/api/models/run/'))data=modelsFixtures[url.split('/').at(-1)];
  else if(url==='/api/models/compare')data=modelsBundle.comparisons[[...body.runs].sort().join('|')];
  else if(url==='/api/compare'){if(context.deferComparison)await new Promise(resolve=>{context.releaseComparison=resolve;});data=context.languageComparisonFixture;}
  else if(url==='/api/hardware')data={cpu:true,cuda:false,mps:false,selected:'cpu',summary:'CPU selected',note:'Test fixture',probe_errors:{}};
  else if(url==='/api/learning')data=course.index;
  else if(url==='/api/course/catalog')data=guided.index;
  else if(url.startsWith('/api/course/lesson/'))data=guided.docs[decodeURIComponent(url.split('/').at(-1))];
  else if(url.startsWith('/api/learning/document/'))data=course.docs[decodeURIComponent(url.split('/').at(-1))];
  else if(url.startsWith('/api/learning/interview/'))data=course.questions[decodeURIComponent(url.split('/').at(-1))];
  else if(url==='/api/runs')data=empty?[]:Object.entries(measured).map(([id,r])=>({id,status:r.manifest.status,engine:r.manifest.engine}));
  else if(url==='/api/job')data={running:false};else if(url==='/api/studies')data=[];
  else if(url.startsWith('/api/run/'))data=measured[url.split('/').at(-1)];
  else if(url==='/api/preview')data={runs:[{label:'Reference',settings:catalog.defaults,estimate:{effective_windows_per_update:4}}]};
  else if(url==='/api/tokenize')data={ids:[1],pieces:[body.text],round_trip:body.text};
  else if(url==='/api/generate')data={...body,completion:'Unedited test fixture'};
  else if(url==='/api/export')data={status:'EXPORTED',directory:'local/exports/run'};
  else if(url==='/api/trace')data=fixtureTrace;
  else throw Error('Unexpected request '+url);
  assert.ok(data,'Fixture exists for '+url);return {ok:true,json:async()=>data};
 }};
 vm.createContext(context);for(const file of ['visuals.js','ml_visuals.js','architecture_math.js','architecture_lessons.js','explorer_lessons.js','explorer_ml.js','explorer_dl.js','explorer_llm.js','nlp_math.js','explorer_nlp.js','nlp_figures.js','explorer_figures.js','metrics.js','architecture_explorer.js','dsa_scenes.js','learning.js','readouts.js','lab_setup.js','models.js','course.js','app.js'])vm.runInContext(fs.readFileSync(path.join(web,file),'utf8'),context);await settle();
 const go=async name=>{all.find(e=>e.dataset.page===name).onclick();await settle();};
 assert.equal(elements.home.hidden,false);assert.equal(elements['home-visual'],undefined);const homeHTML=html.match(/<section id="home"[\s\S]*?<\/section>/)[0];assert.doesNotMatch(homeHTML,/<a\b|<svg\b|<canvas\b/);assert.equal((homeHTML.match(/<button\b/g)||[]).length,1);assert.match(homeHTML,/Explore the menu/);assert.match(homeHTML,/Interactive mode/);
 assert.match(homeHTML,/There is no required starting point/);assert.ok(homeHTML.indexOf('Site map')<homeHTML.indexOf('Choose how to explore'));
 const mapNodes=descendants(elements['home-sitemap']);assert.equal(elements['home-sitemap'].children.length,7);
 const mapLessons=mapNodes.filter(n=>n.dataset.mapLesson);assert.equal(mapLessons.length,course.index.lessons.length);
 assert.equal(new Set(mapLessons.map(n=>n.dataset.mapLesson)).size,174,'Every sidebar lesson appears once in the map');
 for(const lesson of course.index.lessons){
  const b=mapLessons.find(n=>n.dataset.mapLesson===lesson.id);assert.ok(b,lesson.id+' map entry');
  const reader=course.docs[lesson.id].reader;
  const expected=reader.type==='code'?(reader.topics.length?reader.topics:['Code and output','Explanation and sources']):[...new Set([...reader.reading.steps.map(s=>s.title),...reader.notes.map(n=>n.title)])];
  assert.deepEqual(lesson.topics,expected,lesson.id+' map follows the live student view');
  const topics=descendants(b.parent).find(n=>n.className==='map-topics');assert.deepEqual(topics.children.map(n=>n.textContent),expected);
 }
 assert.equal(mapNodes.filter(n=>n.dataset.mapQuestion).length,course.index.questions.length,'Every interview question is available in the map');
 assert.deepEqual(mapNodes.filter(n=>n.dataset.mapPage).map(n=>n.dataset.mapPage),['ml-lab','ml-results','ml-compare','dl-lab','dl-results','dl-compare','visuals','train','explore','compare']);
 for(const id of ['P8','ML07','DL05','N24','T3','R9']){await mapLessons.find(n=>n.dataset.mapLesson===id).onclick();await settle();assert.equal(elements['page-title'].textContent,course.docs[id].title);}
 await mapNodes.find(n=>n.dataset.mapPage==='ml-compare').onclick();await settle();assert.equal(elements['model-compare'].hidden,false);
 await mapNodes.find(n=>n.dataset.mapQuestion).onclick();await settle();assert.equal(elements['question-title'].textContent,course.questions[mapNodes.find(n=>n.dataset.mapQuestion).dataset.mapQuestion].title);
 await go('home');
 assert.match(document.title,/Machine Learning Study Lab/);assert.match(html,/<title>Machine Learning Study Lab<\/title>/);
 assert.equal(elements['adapt-controls'].children.length,8);assert.equal(elements['scratch-controls'].children.length,10);assert.equal(elements['scratch-advanced-controls'].children.length,12);assert.equal(elements.guide.options.length,20);
 assert.equal(requests.filter(r=>r.body&&r.url!='/api/preview').length,0,'Opening a lesson never starts training');
 for(const id of ['journal','page-next','page-guide','practice-answer','reference','section-next'])assert.equal(elements[id],undefined,'Removed '+id);
 assert.equal(elements['course-lesson'],undefined,'No course topic dropdown');
 await go('interactive');assert.equal(elements.interactive.hidden,false);assert.equal(elements['interactive-paths'].children.length,6);
 const visitedGuided=new Set();let checkedGuided=0;
 const topicButtons=()=>descendants(elements['interactive-outline']).filter(n=>n.className==='journey-topic-link');
 for(const p of guided.index.paths){
  await elements['interactive-paths'].children.find(c=>c.children.at(-1).dataset.path===p.id).children.at(-1).onclick();await settle();
  assert.equal(elements['interactive-back'].disabled,true);assert.match(elements['interactive-position'].textContent,/Lesson 1/);
  await elements['interactive-next'].onclick();await settle();assert.match(elements['interactive-position'].textContent,/Step 2/);
  await elements['interactive-back'].onclick();assert.match(elements['interactive-position'].textContent,/Step 1/);
  for(let i=0;i<p.lessons.length;i++){
   const id=p.lessons[i].id;if(visitedGuided.has(id))continue;visitedGuided.add(id);
   await elements['interactive-outline'].children[i].children[0].onclick();await settle();const deck=guided.docs[id];
   assert.equal(topicButtons().length,deck.steps.length,id+' full step map');
   for(let j=0;j<deck.steps.length;j++){
    topicButtons()[j].onclick();const step=deck.steps[j];
    assert.ok(content(elements['interactive-body']).length>10,id+' '+step.id+' rendered');
    if(step.kind==='question'){
     checkedGuided++;assert.equal(elements['interactive-next'].disabled,true,'An unanswered check cannot be counted by Next');
     const wrong=(step.question.answer+1)%4,radios=descendants(elements['interactive-body']).filter(n=>n.tag==='input');assert.equal(radios.length,4);
     radios[wrong].onchange();elements['interactive-check'].onclick();assert.match(content(elements['interactive-feedback']),/Not quite/);assert.equal(elements['interactive-next'].disabled,false);
     radios[step.question.answer].onchange();elements['interactive-check'].onclick();assert.match(content(elements['interactive-feedback']),/Correct after review/);
     assert.match(content(elements['interactive-feedback']),/Reasoning for all four choices/);
    }
   }
  }
  // A skipped check stays reviewable, and resumes at its actual source-aware step.
  await elements['interactive-outline'].children[0].children[0].onclick();await settle();const first=guided.docs[p.lessons[0].id];
  const qi=first.steps.findIndex(s=>s.kind==='question');topicButtons()[qi].onclick();elements['interactive-skip'].onclick();assert.match(content(elements['interactive-feedback']),/Skipped/);assert.ok(elements['interactive-review-list'].children.some(n=>n.tag==='button'));
  await go('home');await go('interactive');await elements['interactive-paths'].children.find(c=>c.children.at(-1).dataset.path===p.id).children.at(-1).onclick();await settle();assert.match(elements['interactive-position'].textContent,new RegExp('Step '+(qi+1)+' '));
  const beforeFull=requests.length;elements['interactive-full-lesson'].onclick();await settle();assert.equal(elements.course.hidden,false);assert.equal(elements['page-title'].textContent,course.docs[p.lessons[0].id].title);assert.ok(requests.slice(beforeFull).every(r=>!r.body));
  await go('interactive');await elements['interactive-paths'].children.find(c=>c.children.at(-1).dataset.path===p.id).children.at(-1).onclick();await settle();
  await elements['interactive-outline'].children.at(-1).children[0].onclick();await settle();topicButtons().at(-1).onclick();await elements['interactive-next'].onclick();assert.equal(elements['interactive-complete'].hidden,false);assert.match(content(elements['interactive-complete-content']),/not proof of independent skill/);assert.match(content(elements['interactive-complete-content']),/unvisited step/);
  elements['interactive-finish-paths'].onclick();
 }
 assert.equal(visitedGuided.size,174);assert.equal(checkedGuided,278);assert.ok(stored.has('model-lab.interactive.v1'));assert.equal(elements['interactive-error'].textContent,'');
 await elements['interactive-paths'].children.find(c=>c.children.at(-1).dataset.path==='llm').children.at(-1).onclick();await settle();
 const llm=guided.index.paths.find(p=>p.id==='llm'),loopIndex=llm.lessons.findIndex(l=>l.id==='T0');await elements['interactive-outline'].children[loopIndex].children[0].onclick();await settle();
 topicButtons()[guided.docs.T0.steps.findIndex(s=>s.kind==='lab')].onclick();const labButton=descendants(elements['interactive-body']).find(n=>n.tag==='button');assert.ok(labButton);labButton.onclick();await settle();assert.equal(elements.train.hidden,false,'Deliberate lab handoff opens controls without starting a job');
 assert.equal(requests.filter(r=>r.body&&r.url!='/api/preview').length,0,'Every guided screen, answer and full-lesson handoff is read-only');
 for(const [page,count] of [['basics',5],['python-basics',14],['python-data',10],['python-ml',12],['python-dl',12],['python-nlp',8],['python-llm',10],['python-libraries',9],['python-dsa',20],['ml',16],['dl',14],['language',2],['nlp',24],['training',9],['post',9]]){
  await go(page);assert.equal(elements.course.hidden,false);
  const menu=elements['course-menu-'+page],toggle=elements['course-toggle-'+page];
  assert.equal(menu.children.length,count);assert.equal(menu.hidden,false);assert.equal(toggle.getAttribute('aria-expanded'),'true');
  const last=menu.children.at(-1);await last.onclick();await settle();
  assert.equal(last.getAttribute('aria-current'),'page','Selected child highlighted');
  const lessonText=content(elements['course-content']);await go(page);assert.equal(menu.hidden,true,'Parent collapses its submenu');
  await go(page);assert.equal(menu.hidden,false);assert.equal(content(elements['course-content']),lessonText,'Expand preserves the selected lesson');
 }
 for(const lesson of course.index.lessons){
  await vm.runInContext(`LabLearning.openDocument(${JSON.stringify(lesson.id)})`,context);
  assert.equal(elements['page-title'].textContent,lesson.title,'Actual lesson title is the single page H1');
  const children=elements['course-content'].children;
  if(course.docs[lesson.id].reader.type==='code'){
   const codeReaders=descendants(elements['course-content']).filter(n=>n.className==='code-lesson');
   assert.equal(codeReaders.length,1,lesson.id+' one code reader');
   assert.ok(content(codeReaders[0]).length>500,lesson.id+' authored explanation visible');
   const examples=descendants(codeReaders[0]).filter(n=>n.dataset.example);
   const expected=course.docs[lesson.id].reader.blocks.filter(b=>b.kind==='example');
   assert.equal(examples.length,expected.length);
   for(const block of expected){const example=examples.find(n=>n.dataset.example===block.id);assert.ok(content(example).includes(block.code));assert.ok(content(example).includes(block.output));}
   const sceneBlock=course.docs[lesson.id].reader.blocks.find(block=>block.kind==='scene');
   if(sceneBlock){
    const teaching=course.docs[lesson.id].reader.teaching;
    const recipes=descendants(codeReaders[0]).filter(n=>n.className==='dsa-recipe');
    assert.equal(recipes.length,teaching.recipes.length,lesson.id+' all short recipes served');
    assert.ok(descendants(codeReaders[0]).find(n=>n.className==='dsa-definition').textContent.includes(teaching.definition));
    for(let i=0;i<recipes.length;i++){
     const nodes=descendants(recipes[i]).filter(n=>n.className==='dsa-flow-node');
     assert.equal(nodes.length,teaching.recipes[i].nodes.length);
     nodes.at(-1).onclick();
     assert.equal(recipes[i].dataset.selectedNode,teaching.recipes[i].nodes.at(-1).id);
     assert.ok(descendants(recipes[i]).find(n=>n.className==='dsa-code-line active'),'flow selection highlights code');
    }
    const scene=descendants(codeReaders[0]).find(n=>n.className==='dsa-scene');
    assert.ok(scene,lesson.id+' picture is visible');
    assert.equal(scene.dataset.scene,lesson.id);
    const alt=descendants(scene).find(n=>n.className==='dsa-alt');
    const before=alt.textContent;
    assert.match(before,/^Text alternative: /);
    descendants(scene).find(n=>n.tag==='button'&&n.textContent==='Next').onclick();
    assert.notEqual(alt.textContent,before,lesson.id+' next picture changes');
    descendants(scene).find(n=>n.tag==='button'&&n.textContent==='Reset').onclick();
    assert.equal(alt.textContent,before,lesson.id+' reset restores the picture');
    if(lesson.id==='DA05'){
     const select=descendants(scene).find(n=>n.tag==='select');
     const prompt=descendants(scene).find(n=>n.className==='dsa-prompt');
     const firstPrompt=prompt.textContent;
     select.value=String(select.children.length-1);
     select.onchange();
     assert.notEqual(prompt.textContent,firstPrompt,'case selection changes the picture');
    }
   }
   continue;
  }
  const explorers=descendants(elements['course-content']).filter(n=>n.className?.split(' ').includes('architecture-explorer'));
  assert.equal(explorers.length,1,lesson.id+' one canonical reader');
  if(course.docs[lesson.id].reader.reading){
   const edition=course.docs[lesson.id].reader.reading;
   const intro=descendants(explorers[0]).find(n=>n.className?.split(' ').includes('ax-topic-overview'));
   assert.equal(intro.hidden,false);
   if(edition.lesson_markdown){
    assert.ok(content(intro).length>700,lesson.id+' complete teaching explanation is rendered');
    assert.ok(descendants(intro).some(n=>n.tag==='a'&&/^https:\/\//.test(n.href)),lesson.id+' primary reading links are visible');
   }else assert.ok(content(intro).includes(edition.definition));
   assert.equal(descendants(explorers[0]).find(n=>n.className==='ax-diagram-column').hidden,edition.view==='reading');
  }
  assert.ok(content(explorers[0]).length>500,lesson.id+' core explanation');
  assert.ok(children.every(n=>['lesson-explorer-host','lesson-related'].includes(n.className)),lesson.id+' no second prose lesson');
  assert.equal(descendants(elements['course-content']).filter(n=>n.className==='parameter-appendix').length,0);
  const notes=course.docs[lesson.id].reader.notes;
  const guide=descendants(explorers[0]).filter(n=>n.dataset.axStep?.startsWith('note-'));
  assert.equal(guide.length,notes.length,lesson.id+' unique advanced notes live in the guide');
  for(const n of guide){n.onclick();assert.ok(content(elements['course-content']).length>100);assert.doesNotMatch(content(elements['course-content']),/3D structure/);}

 }
 const before=content(elements['course-content']);await vm.runInContext('refresh()',context);assert.equal(content(elements['course-content']),before,'Polling preserves lesson');
 for(const q of Object.values(course.questions))vm.runInContext(`LabLearning.markdown(${JSON.stringify(q.markdown)},document.getElementById('question-answer'))`,context);
 assert.equal(elements['question-select'],undefined,'No question dropdown');assert.equal(elements['question-category'],undefined,'No topic dropdown');
 await go('interview');assert.equal(elements['interview-menu'].children.length,new Set(course.index.questions.map(q=>q.category)).size);assert.equal(elements['interview-menu'].hidden,false);assert.ok(content(elements['question-answer']).length>100);
 const questionLinks=descendants(elements['interview-menu']).filter(n=>n.dataset.question);assert.equal(questionLinks.length,course.index.questions.length);
 for(const [category,count] of Object.entries(course.index.questions.reduce((a,q)=>(a[q.category]=(a[q.category]||0)+1,a),{}))){
  const list=elements['question-menu-'+category],toggle=elements['question-topic-'+category];assert.equal(list.children.length,count);
  if(list.hidden)toggle.onclick();const last=list.children.at(-1);await last.onclick();await settle();
  assert.equal(last.getAttribute('aria-current'),'page');assert.equal(list.hidden,false);assert.equal(elements['question-title'].textContent,course.questions[last.dataset.question].title);
  const answer=content(elements['question-answer']);toggle.onclick();assert.equal(list.hidden,true);toggle.onclick();assert.equal(list.hidden,false);assert.equal(content(elements['question-answer']),answer,'Topic collapse preserves answer');
 }
 const chosenQuestion=elements['question-title'].textContent;await go('interview');assert.equal(elements['interview-menu'].hidden,true);await go('interview');assert.equal(elements['interview-menu'].hidden,false);assert.equal(elements['question-title'].textContent,chosenQuestion);
 elements['interview-search'].value='question 138';vm.runInContext('LabLearning.searchInterviews()',context);assert.equal(elements['interview-search-results'].children.length,1);await elements['interview-search-results'].children[0].onclick();await settle();assert.match(elements['question-title'].textContent,/Attention/);
 elements['interview-search'].value='question 329';vm.runInContext('LabLearning.searchInterviews()',context);assert.equal(elements['interview-search-results'].children.length,1);await elements['interview-search-results'].children[0].onclick();await settle();assert.equal(elements['page-title'].textContent,course.docs.LC04.title);
 elements['interview-search'].value='';vm.runInContext('LabLearning.searchInterviews()',context);
 await vm.runInContext(`LabLearning.openQuestion(${JSON.stringify(Object.values(course.questions).find(q=>q.title===chosenQuestion).id)})`,context);
 await go('home');assert.ok(questionLinks.every(n=>n.getAttribute('aria-current')===undefined));await go('interview');assert.equal(elements['question-title'].textContent,chosenQuestion,'Returning remembers question');
 await go('visuals');assert.equal(elements['visual-select'].options.length,12);
 for(const item of elements['visual-select'].options){elements['visual-select'].value=item.value;elements['visual-select'].onchange();assert.equal(descendants(elements['visual-stage']).filter(n=>n.className==='architecture-explorer').length,1,item.value+' uses linked explorer');}
 const sceneIds=vm.runInContext('LabVisuals.catalog.map(x=>x.id)',context);
 for(const id of [...sceneIds,'targets','trace']){
  const host=new Element();context.host=host;context.options={settings:catalog.defaults,trace:fixtureTrace};
  vm.runInContext(`globalThis.scene=LabVisuals.mount(host,${JSON.stringify(id)},options)`,context);
  assert.ok(content(host).length>150,id+' explanation');
  const figure=host.children[0],body=figure.children[1];
  assert.equal(body.children[0].className,'topic-terms',id+' definitions precede the diagram');
  assert.equal(descendants(host).filter(n=>['button','input','select'].includes(n.tag)).length,0,id+' contains no teaching controls');
  assert.doesNotMatch(content(host),/NaN|Infinity/,id+' finite explanatory output');
  if(id==='batching')assert.match(content(host),/3\.2/);
  if(id==='attention')assert.match(content(host),/1\.3395/);
  vm.runInContext('scene.updateSettings({...options.settings,heads:2});scene.destroy()',context);assert.equal(host.children.length,0,id+' removes listeners/root');
 }
 await go('train');elements['read-training-loop'].onclick({preventDefault(){}});await settle();assert.equal(elements['page-title'].textContent,course.docs.T0.title);assert.equal(elements['course-menu-training'].children[0].getAttribute('aria-current'),'page');
 await go('train');all.find(e=>e.dataset.engine==='scratch').onclick();assert.equal(elements['scratch-setup'].hidden,false);assert.equal(elements['architecture-preview'],undefined);assert.equal(elements['language-structure'],undefined);assert.match(content(elements['language-experiment-review']),/make_plan/);all.find(e=>e.dataset.engine==='lora').onclick();assert.equal(elements['language-setup'].hidden,false);assert.match(content(elements['language-experiment-review']),/train\("my-adapter-experiment"/);
 await go('explore');if(empty){assert.equal(elements['empty-run'].hidden,false);assert.equal(elements['run-content'].hidden,true);}else{
  assert.equal(elements['run-content'].hidden,false);assert.match(content(elements.scores),/Development/);
  for(const t of ['tokens','inside','generate','results','curves']){all.find(e=>e.dataset.tab===t).onclick();assert.equal(elements['tab-'+t].hidden,false);}
  elements.text.value='<script>literal</script> 🧠';await elements.tokenize.onclick();assert.match(content(elements['tokens-output']),/🧠/);
  elements['trace-text'].value='The cat sat';await elements.inspect.onclick();assert.match(content(elements['actual-trace']),/Measured values/);
  elements['generation-prompt'].value='Explain learning.';await elements.generate.onclick();assert.match(content(elements['generation-output']),/Unedited test fixture/);
  if(measured[elements.runs.value].manifest.engine==='lora'){assert.equal(elements.export.disabled,true);assert.equal(elements['resume-area'].hidden,true);assert.match(content(elements.samples),/Factual error/);}
 }
 await go('compare');assert.equal(elements.compare.hidden,false);
 for(const area of ['ml','dl']){
  await go(area+'-lab');assert.equal(elements['model-lab'].hidden,false);assert.ok(elements['model-settings'].children.length>=4);assert.ok(elements['model-guide'].options.length>=5);assert.equal(elements['subject-'+area].hidden,false);
  for(const task of elements['model-task'].options){elements['model-task'].value=task.value;elements['model-task'].onchange();assert.ok(elements['model-method'].options.length>0);assert.ok(content(elements['model-method-info']).length>30);}
  await go(area+'-results');assert.equal(elements['model-results'].hidden,false);
  if(empty)assert.match(content(elements['model-results-content']),/No saved/);
  else for(const [id,r] of Object.entries(modelsFixtures).filter(([id,r])=>r.manifest.settings.area===area)){elements['model-results-run'].value=id;await elements['model-results-run'].onchange();await settle();assert.match(content(elements['model-results-content']),/PASS/);assert.ok(content(elements['model-results-content']).length>300);}
  await go(area+'-compare');assert.equal(elements['model-compare'].hidden,false);
  if(!empty){
   const pairs=area==='ml'?Object.keys(modelsBundle.comparisons).filter(s=>!s.includes('ui-neural')):['ui-neural|ui-neural-cnn'];
   for(const pair of pairs){const wanted=pair.split('|');for(const label of elements['model-compare-choices'].children){const box=label.children[0];box.checked=wanted.includes(box.value);box.onchange();}assert.equal(elements['model-compare-start'].disabled,false);await elements['model-compare-start'].onclick();await settle();assert.match(content(elements['model-compare-output']),/What changed/);assert.equal(elements['model-compare-error'].textContent,'');assert.match(content(elements['model-compare-output']),/Read quality and cost together|Compare the same reconstruction|Compare useful groupings|Separate ranking/);}
   const checked=elements['model-compare-choices'].children.map(n=>n.children[0]).find(n=>n.checked);checked.checked=false;checked.onchange();assert.equal(elements['model-compare-output'].children.length,0,'Changing the selection clears the stale comparison');assert.equal(elements['model-compare-start'].disabled,true,'One run is not a comparison');
  }
 }
 elements['metrics-language-results'].onclick();await settle();assert.equal(elements['page-title'].textContent,course.docs.P8.title);
 assert.equal(descendants(elements['course-content']).find(n=>n.dataset.axField==='family').value,'language');
 elements['metrics-language-compare'].onclick();await settle();assert.equal(elements['page-title'].textContent,course.docs.P8.title);
 if(!empty){await go('ml-results');elements['model-results-run'].value='ui-classic';await elements['model-results-run'].onchange();await settle();const b=descendants(elements['model-results-content']).find(n=>n.tag==='button'&&n.textContent==='Explore these metrics');assert.ok(b);b.onclick();await settle();assert.equal(descendants(elements['course-content']).find(n=>n.dataset.axField==='family').value,'multiclass');}
 // Isolated comparison fixtures exercise the UI, not a benchmark or stored run.
 await go('compare');await settle();
 context.languageComparisonFixture={nll_comparable:true,bits_per_byte_comparable:false,warnings:[],changes:[],runs:[0,1].map(i=>({id:'ui-language-'+i,manifest:{completed_steps:10,tokens_processed:100},evaluation:{transformer:{dev:{nll:2-i*.1},test:{nll:2.1-i*.1}}},metrics:[{step:10,dev_nll:2-i*.1,tokens_processed:100}]}))};
 vm.runInContext('runList=languageComparisonFixture.runs.map(r=>({id:r.id,status:"MEASURED",engine:"lora"}));selectedComparisons.clear();delete document.getElementById("compare-choices").dataset.signature;renderCompareChoices()',context);
 const comparisonBoxes=elements['compare-choices'].children.map(n=>n.children[0]);
 for(const b of comparisonBoxes){b.checked=true;b.onchange();}
 assert.equal(elements['compare-run'].disabled,false);await elements['compare-run'].onclick();
 assert.equal(elements['comparison-results'].hidden,false);assert.match(content(elements['comparison-reading']),/first row is the reference/);assert.match(content(elements['comparison-table']),/-0.1000/);
 comparisonBoxes[0].checked=false;comparisonBoxes[0].onchange();assert.equal(elements['comparison-results'].hidden,true);assert.equal(elements['compare-run'].disabled,true);
 comparisonBoxes[0].checked=true;comparisonBoxes[0].onchange();context.deferComparison=true;
 const pending=elements['compare-run'].onclick();await settle();assert.equal(typeof context.releaseComparison,'function');
 comparisonBoxes[0].checked=false;comparisonBoxes[0].onchange();context.releaseComparison();await pending;
 assert.equal(elements['comparison-results'].hidden,true,'An old async comparison cannot overwrite a changed selection');context.deferComparison=false;
 comparisonBoxes[0].checked=true;comparisonBoxes[0].onchange();context.languageComparisonFixture.nll_comparable=false;
 await elements['compare-run'].onclick();assert.equal(elements['comparison-chart-region'].hidden,true,'Incomparable units do not produce an empty pretend curve');assert.match(content(elements['comparison-reading']),/cannot be ranked/);
 vm.runInContext('LabLearning.markdown("<script>alert(1)</script> [bad](javascript:alert) **safe** `literal`",document.getElementById("course-content"))',context);assert.match(content(elements['course-content']),/<script>/);assert.ok(!all.some(n=>n.href?.startsWith('javascript:')));
 assert.ok(equations>=80,'All lessons/answers/static diagrams exercised actual equation parsing: '+equations);assert.deepEqual(mathErrors,[],'All course, interview and diagram equations parse');
 assert.doesNotMatch(html,/brand-icon|A working notebook/,'No logo or decorative tagline');
 console.log(`PASS ${empty?'empty':'saved'} state: 174 canonical lessons, all guided steps and 278 checks, 62 interview topics, 18 static diagrams, ${equations} equation renders; Python code/output delivery, unified interview navigation/search/code handoffs, guided navigation/review/resume/lab handoffs, ML/DL/LLM controls, real-result fixtures, tabs, traces, safety and no auto-training. DOM stand-ins only.`);
}
(async()=>{await exercise(false);await exercise(true);})().catch(e=>{console.error(e);process.exitCode=1;});
