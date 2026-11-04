// All authored variants and linked diagrams. DOM stand-ins are not browser sign-off.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const web=path.resolve(__dirname,'../src/training_lab/web'),katex=require(path.join(web,'vendor/katex/katex.min.js'));
class E{
 constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.dataset={};this.textContent='';}
 append(...nodes){for(const n of nodes){n.parent=this;this.children.push(n);}}
 replaceChildren(...nodes){this.children=[];this.textContent='';this.append(...nodes);}
 setAttribute(k,v){this.attributes[k]=String(v);if(k==='class')this.className=v;}
 getAttribute(k){return this.attributes[k];}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(n=>n!==this);}
 focus(){this.focused=true;}
}
const all=n=>[n,...n.children.flatMap(all)],content=n=>n.textContent+' '+n.children.map(content).join(' ');
let equations=0,states=0,operations=0;
const c=vm.createContext({document:{createElement:t=>new E(t),createElementNS:(_,t)=>new E(t)},
 LabLearning:{formula(n,tex,display){n.textContent=tex;assert.match(katex.renderToString(tex,{displayMode:display,output:'mathml',throwOnError:true,trust:false}),/<math/);equations++;}}});
for(const file of ['architecture_math','architecture_lessons','explorer_lessons','explorer_ml','explorer_dl','explorer_llm','nlp_math','explorer_nlp','nlp_figures','explorer_figures','metrics','architecture_explorer']){
 const source=fs.readFileSync(path.join(web,file+'.js'),'utf8');
 assert.doesNotMatch(source,/fetch\(|setInterval|setTimeout|requestAnimationFrame|innerHTML|WebGL/);vm.runInContext(source,c);
}
const ids=vm.runInContext('ConceptLessons.ids()',c);assert.equal(ids.length,77);
const materials=path.resolve(__dirname,'../learning_materials');
const profiles=JSON.parse(fs.readFileSync(path.join(materials,'reader_profiles.json'),'utf8')).lessons;
const build=(id,options={})=>vm.runInContext('ConceptLessons.build('+JSON.stringify(id)+','+JSON.stringify(options)+')',c);
const outputAt=process.argv.indexOf('--export'),output=outputAt>=0?path.resolve(process.argv[outputAt+1]):null;
if(output)fs.mkdirSync(output,{recursive:true});
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
function serialize(n){
 const a={...n.attributes};
 if(n.tag==='svg')a.xmlns='http://www.w3.org/2000/svg';
 if(n.tag==='rect'&&!a.fill){a.fill='#eef3f9';a.stroke='#9baec1';}
 if(n.tag==='text'){a.fill='#30485e';a['font-family']='Arial';}
 if(n.tag==='path'&&a.class==='ax-edge-line')a.stroke=n.parent.getAttribute('data-kind')==='training'?'#8b739b':n.parent.getAttribute('data-kind')==='bypass'?'#a57833':'#8ca3b9';
 return '<'+n.tag+Object.entries(a).map(([k,v])=>' '+k+'="'+esc(v)+'"').join('')+'>'+esc(n.textContent)+n.children.filter(n=>n.getAttribute('class')!=='ax-edge-hit').map(serialize).join('')+'</'+n.tag+'>';
}
function geometry(svg){
 const [,,w,h]=svg.getAttribute('viewBox').split(' ').map(Number);
 for(const n of all(svg)){
  for(const [k,value] of Object.entries(n.attributes))if(['x','y','x1','x2','y1','y2','cx','cy','width','height','r'].includes(k))assert.ok(Number.isFinite(+value),k);
  if(n.tag==='rect'){assert.ok(+n.attributes.x>=0);assert.ok(+n.attributes.y>=0);assert.ok(+n.attributes.x+(+n.attributes.width)<=w);assert.ok(+n.attributes.y+(+n.attributes.height)<=h);}
  if(n.tag==='text'){assert.ok(+n.attributes.x>=0&&+n.attributes.x<=w);assert.ok(+n.attributes.y>=0&&+n.attributes.y<=h);}
  if(n.tag==='circle'){assert.ok(+n.attributes.cx-Number(n.attributes.r)>=0);assert.ok(+n.attributes.cy-Number(n.attributes.r)>=0);assert.ok(+n.attributes.cx+Number(n.attributes.r)<=w);assert.ok(+n.attributes.cy+Number(n.attributes.r)<=h);}
 }
}
for(const id of ids){
 const base=build(id),variants=[{}];
 for(const ctrl of base.controls)for(const [value] of ctrl.choices)variants.push({[ctrl.key]:value});
 if(id==='P8')for(const [family,key,values] of [['multiclass','classIndex',[0,1,2]],['language','mask',['all','response']],['retrieval','k',[1,2,3]]])for(const value of values)variants.push({family,[key]:value});
 for(const options of variants){
  const lesson=build(id,options);states++;
  assert.ok(lesson.sources.length);assert.match(lesson.subtitle,/teaching examples/);
  const stepIds=new Set(lesson.steps.map(s=>s.id));assert.equal(stepIds.size,lesson.steps.length);
  for(const edge of lesson.diagram.edges){assert.ok(stepIds.has(edge.from));assert.ok(stepIds.has(edge.to));}
  c.host=new E('div');c.profile=profiles[id];const explorer=vm.runInContext('ArchitectureExplorer.mount(host,"'+id+'",{reading:profile})',c);
  for(const [key,value] of Object.entries(options)){
    const field=all(c.host).find(n=>n.dataset.axField===key);field.value=String(value);field.onchange();
  }
  for(const step of explorer.getLesson().steps){
    operations++;explorer.choose(step.id);
    for(const key of ['why','terms','input','action','calculation','output','next','change'])assert.ok(step[key]?.length>0,id+' '+step.id+' '+key);
    assert.doesNotMatch(content(c.host),/\bNaN\b/);
    const edition=explorer.getTrace().reading;
    const activeNode=edition?edition.view==='diagram'&&edition.diagram.nodes.includes(+step.id.slice(1)):true;
    assert.equal(all(c.host).filter(n=>n.getAttribute('data-selected')==='true'&&n.getAttribute('data-node')).length,activeNode?1:0);
    const baseline=build(id,options).steps.find(s=>s.id===step.id);
    assert.equal(step.calculation,baseline.calculation,'Editorial changes preserve calculated values');
    assert.equal(step.equation,baseline.equation,'Editorial changes preserve the objective');
    if(profiles[id]){assert.ok(step.prose.length>100);assert.doesNotMatch(step.next,/outgoing connection carries|endpoint of the worked/);}
    assert.equal(all(c.host).find(n=>n.dataset.axStep===step.id).getAttribute('aria-current'),'step');
    for(const svg of all(c.host).filter(n=>n.tag==='svg'))geometry(svg);
  }
  // Actual SVG edge and ordered-guide interactions agree.
  const edge=all(c.host).find(n=>n.getAttribute('data-edge'));
  if(edge){edge.onclick();assert.equal(all(c.host).find(n=>n.getAttribute('aria-current')==='step').dataset.axStep,edge.getAttribute('data-to'));}
  const button=all(c.host).find(n=>n.dataset.axStep==='s0');button.onkeydown({key:'ArrowDown',preventDefault(){}});
  assert.equal(all(c.host).find(n=>n.getAttribute('aria-current')==='step').dataset.axStep,'s1');
  if(output&&!Object.keys(options).length){
    const main=all(c.host).find(n=>n.className==='ax-svg');if(main)fs.writeFileSync(path.join(output,id+'.svg'),serialize(main));
    const detail=all(c.host).find(n=>['ax-concept-figure','ax-metric-figure','ax-nlp-figure'].includes(n.className));if(detail)fs.writeFileSync(path.join(output,id+'-detail.svg'),serialize(all(detail).find(n=>n.tag==='svg')));
    if(id==='P8')for(const [family,step] of [['classification','s3'],['classification','s4'],['classification','s5'],['classification','s6'],['multiclass','s2'],['regression','s2'],['language','s2'],['retrieval','s3']]){explorer.configure({family});explorer.choose(step);const detail=all(c.host).find(n=>n.className==='ax-metric-figure');fs.writeFileSync(path.join(output,'P8-'+family+'-'+step+'.svg'),serialize(all(detail).find(n=>n.tag==='svg')));}
  }
  if(output&&id==='DL10'&&options.family)fs.writeFileSync(path.join(output,'DL10-'+options.family+'.svg'),serialize(all(c.host).find(n=>n.className==='ax-svg')));
  if(id==='R5'&&options.approval==='deny'){
    const trace=explorer.getTrace().reading;
    assert.ok(!trace.diagram.nodes.includes(3),'Denied authorization excludes execution');
    assert.ok(trace.diagram.links.some(([from,to])=>from===2&&to===4),'Denial leads directly to reporting');
  }
  if(output&&((id==='T4'&&options.family)||(id==='R5'&&options.approval))){
    const svg=all(c.host).find(n=>n.className==='ax-svg');
    if(svg)fs.writeFileSync(path.join(output,id+'-'+Object.values(options).join('-')+'.svg'),serialize(svg));
  }
  explorer.destroy();assert.equal(c.host.children.length,0);
 }
}
// Independent arithmetic checks against rendered authored calculations.
if(output)for(const [id,cfg,step,label] of [
 ['N09',{directions:2},'s4','N09-bilstm-states'],['N10',{directions:2},'s3','N10-bigru-states'],
 ['N15',{mode:'cross'},'s3','N15-cross'],['N15',{mode:'late'},'s3','N15-late'],
 ['N12',{kind:'hmm'},'s2','N12-hmm'],['N23',{mask:'local',overlap:0},'s3','N23-local'],
 ['N11',{beam:2},'s4','N11-beam-search']
]){
 c.host=new E('div');const scene=vm.runInContext('ArchitectureExplorer.mount(host,'+JSON.stringify(id)+')',c);scene.configure(cfg);scene.choose(step);
 const detail=all(c.host).find(n=>n.className==='ax-nlp-figure'),svg=all(detail).find(n=>n.tag==='svg');geometry(svg);fs.writeFileSync(path.join(output,label+'.svg'),serialize(svg));scene.destroy();
}
// Distinct mechanisms have distinct guide bindings, not just matching labels.
for(const id of ['N11','N15']){
 c.host=new E('div');c.profile=profiles[id];
 const scene=vm.runInContext('ArchitectureExplorer.mount(host,'+JSON.stringify(id)+',{reading:profile})',c);
 if(id==='N11'){
  scene.choose('s2');assert.equal(all(c.host).filter(n=>n.className==='ax-nlp-figure').length,0);
  scene.choose('s3');const figure=all(c.host).find(n=>n.className==='ax-nlp-figure');assert.ok(figure);
  const start=all(figure).find(n=>n.getAttribute('aria-label')==='Start; ');start.onclick();
  assert.equal(all(c.host).find(n=>n.getAttribute('aria-current')==='step').dataset.axStep,'s3');
 }else{
  scene.configure({mode:'cross'});scene.choose('s1');
  const figure=all(c.host).find(n=>n.className==='ax-nlp-figure');
  assert.match(content(figure),/two input tokens/);assert.doesNotMatch(content(figure),/two valid token vectors/);
  all(figure).find(n=>n.getAttribute('aria-label')?.startsWith('One joint encoder;')).onclick();
  assert.equal(all(c.host).find(n=>n.getAttribute('aria-current')==='step').dataset.axStep,'s3');
 }
 scene.destroy();assert.equal(c.host.children.length,0);
}
const check=(id,cfg,index,text)=>assert.ok(build(id,cfg).steps[index].calculation.includes(text),id+' '+text);
check('ML02',{query:4},2,'= 2');
check('ML03',{lambda:1},1,'0.7333');
check('ML04',{threshold:.7},2,'false');
check('ML05',{lambda:.1},2,'1.3');
check('ML06',{k:3},2,'0.3333');
check('ML07',{alpha:1},3,'0.6667');
check('ML08',{cut:2.5},4,'reduction 0.5');
check('ML10',{strength:1},1,'0.185');
check('ML10',{strength:1},3,'0.75');
check('ML12',{threshold:2},3,'false');
check('ML13',{window:2},2,'4.5');
check('ML15',{mass:.2},2,'0.34');
check('DL01',{activation:'relu'},3,'0.5');
check('DL02',{w:2},3,'dL/dw=-2');
check('DL03',{optimizer:'adam'},3,'0.9');
check('DL04',{keep:.5},1,'[4, 0]');
check('DL06',{scale:.5},3,'1.5');
check('DL09',{width:1},2,'0.5');
check('DL10',{family:'diffusion'},3,'= 1');
check('DL11',{aggregation:'mean'},3,'2.25');
check('DL12',{task:'regression'},3,'=1');
check('T5',{second:6},2,'2.5');
check('T7',{second:6},3,'2.5');
check('R1',{rank:1},3,'[2.2, 1.4]');
check('R2',{method:'ppo'},2,'2.4');
check('R5',{approval:'deny'},3,'No execution result');
check('R6',{tokens:32},2,'2048 bytes');
check('R7',{scale:.2},2,'MSE=0');
check('R8',{budget:.5},2,'2 such');
check('R9',{alpha:.5},2,'0.8');
assert.equal(build('ML11').diagram.edges.find(e=>e.kind==='training').to,'s1');
assert.equal(build('DL06').diagram.edges.find(e=>e.kind==='bypass').to,'s2');
assert.equal(build('T0').diagram.edges.find(e=>e.kind==='training').from,'s4');
assert.ok(build('ML08').diagram.nodes[0].sub.includes('0.5'),'Decimal values survive diagram caption extraction');
assert.ok(build('ML08').diagram.nodes[1].sub.includes('2.5'),'The tree shows its actual split threshold');
assert.throws(()=>build('ML03',{lambda:999}));
assert.ok(!ids.includes('P1'));
assert.ok(ids.includes('DL08')&&ids.includes('P8'));
console.log('PASS: 77 course explorers; '+states+' authored/default setting states; '+operations+' operation visits; '+equations+' real KaTeX renders; bounded SVGs, linked input/edge/guide selection, arithmetic/permission checks and teardown. Not browser certification.');
for(const key of ['vision-transformer','diffusion-training','graph-attention','actor-critic']){
 c.advancedKey=key;
 const figure=vm.runInContext('ConceptFigures.advanced(advancedKey,(tag,attrs={},text)=>{const n=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;},(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;})',c);
 const svg=all(figure).find(n=>n.tag==='svg');assert.ok(svg);geometry(svg);
 assert.ok(svg.getAttribute('aria-label').length>20);
 assert.ok(content(figure).length>200);
 if(output)fs.writeFileSync(path.join(output,'advanced-'+key+'.svg'),serialize(svg));
}
console.log('PASS: four bounded advanced-study diagrams with accessible descriptions; no training or animation.');
