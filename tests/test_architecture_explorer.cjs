// DOM and equation checks, not a replacement for browser layout/accessibility QA.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const web=path.resolve(__dirname,'../src/training_lab/web'),katex=require(path.join(web,'vendor/katex/katex.min.js'));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.dataset={};this.textContent='';}
 append(...items){for(const n of items){n.parent=this;this.children.push(n);}}
 replaceChildren(...items){this.children=[];this.textContent='';this.append(...items);}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(n=>n!==this);}
 setAttribute(k,v){this.attributes[k]=String(v);if(k==='class')this.className=v;}
 getAttribute(k){return this.attributes[k];}
 focus(){this.focused=true;}
}
const all=n=>[n,...n.children.flatMap(all)],content=n=>n.textContent+' '+n.children.map(content).join(' ');
let equations=0;
const context=vm.createContext({document:{createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)},
 LabLearning:{formula(node,tex,display){node.math=katex.renderToString(tex,{displayMode:display,output:'mathml',throwOnError:true,trust:false});assert.match(node.math,/<math/);equations++;}}});
for(const file of ['architecture_math.js','architecture_lessons.js','explorer_lessons.js','architecture_explorer.js']){
 const source=fs.readFileSync(path.join(web,file),'utf8');
 assert.doesNotMatch(source,/fetch\(|setInterval|setTimeout|requestAnimationFrame|WebGL|innerHTML|\.style\b/);
 vm.runInContext(source,context);
}
const fields=host=>all(host).filter(n=>n.tag==='select');
function field(host,key,value){const node=fields(host).find(n=>n.dataset.axField===key);assert.ok(node,key);node.value=String(value);node.onchange();}
function current(host){return all(host).find(n=>n.dataset.axStep&&n.getAttribute('aria-current')==='step');}
function selected(host){return all(host).find(n=>n.getAttribute('data-node')&&n.getAttribute('data-selected')==='true');}
function key(node,value){let prevented=false;node.onkeydown({key:value,preventDefault(){prevented=true;}});assert.ok(prevented);}
function geometry(svg){
 const [,,width,height]=svg.getAttribute('viewBox').split(' ').map(Number);
 for(const n of all(svg)){
  for(const k of ['x','y','x1','x2','y1','y2','width','height'])if(n.attributes[k]!==undefined)assert.ok(Number.isFinite(+n.attributes[k]),k);
  if(n.tag==='rect'){assert.ok(+n.attributes.x>=0);assert.ok(+n.attributes.y>=0);assert.ok(+n.attributes.x+(+n.attributes.width)<=width);assert.ok(+n.attributes.y+(+n.attributes.height)<=height);}
  if(n.tag==='text'){assert.ok(+n.attributes.x<=width);assert.ok(+n.attributes.y<=height);}
 }
}
const exportsAt=process.argv.indexOf('--export'),out=exportsAt>=0?path.resolve(process.argv[exportsAt+1]):null;
if(out)fs.mkdirSync(out,{recursive:true});
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
function serialize(n){
 const a={...n.attributes};
 if(n.tag==='svg')a.xmlns='http://www.w3.org/2000/svg';
 if(n.tag==='rect'&&!a.fill){a.fill=a.class==='ax-block-boundary'?'#f8fafc':'#f1f5fa';a.stroke='#9aafc3';}
 if(n.tag==='text'){a.fill='#26374a';a['font-family']='Arial';}
 if(n.tag==='path'&&a.class==='ax-edge-line'){
  const kind=n.parent.getAttribute('data-kind');a.stroke=kind==='bypass'?'#b37b24':kind==='training'?'#7a698e':kind==='inference'?'#457e72':'#9aaabd';
  if(kind==='training')a['stroke-dasharray']='5 4';
 }
 if(n.tag==='path'&&a.fill==='context-stroke')a.fill='#6c8199';
 return '<'+n.tag+Object.entries(a).map(([k,v])=>' '+k+'="'+escape(v)+'"').join('')+'>'+escape(n.textContent)+n.children.filter(c=>c.getAttribute('class')!=='ax-edge-hit').map(serialize).join('')+'</'+n.tag+'>';
}
const markerIds=[];
const materials=path.resolve(__dirname,'../learning_materials');
const profiles=JSON.parse(fs.readFileSync(path.join(materials,'reader_profiles.json'),'utf8')).lessons;
for(const type of ['transformer','cnn']){
 const host=new Element('div');context.host=host;
 context.reading=profiles[type==='cnn'?'DL05':'T3'];
 const explorer=vm.runInContext('ArchitectureExplorer.mount(host,"'+type+'",{reading})',context);
 assert.ok(content(host).includes(context.reading.definition),type+' topic defined before controls');
 const lessons=explorer.getLesson();
 assert.equal(lessons.steps.length,type==='cnn'?12:25);
 assert.equal(fields(host).length,type==='cnn'?6:5);
 assert.match(content(host),/not a trained|not SmolLM2/);
 assert.equal(all(host).filter(n=>n.tag==='a'&&n.href.startsWith('https://')).length,3);
 const diagram=all(host).find(n=>n.className==='ax-svg');geometry(diagram);
 for(const step of lessons.steps){
  explorer.choose(step.id);
  assert.equal(current(host).dataset.axStep,step.id);
  assert.ok(selected(host),step.id+' main structure highlighted');
  assert.match(content(host),/What enters/);assert.match(content(host),/What leaves/);
  for(const k of ['why','terms','action','input','output','equation','calculation','next','change'])assert.ok(step[k].length>0,step.id+' '+k);
  for(const svg of all(host).filter(n=>n.tag==='svg'))geometry(svg);
 }
 // Clicking the main diagram and connection selects the receiving explanation.
 const firstNode=all(host).find(n=>n.getAttribute('data-node'));
 key(firstNode,'Enter');assert.equal(current(host).dataset.axStep,type==='cnn'?'image':'tokens');
 const edge=all(host).find(n=>n.getAttribute('data-edge')==='flow-'+(type==='cnn'?'image':'tokens'));
 key(edge,' ');assert.equal(current(host).dataset.axStep,type==='cnn'?'conv1':'embedding');assert.match(content(host),/Selected connection/);
 const stepButtons=all(host).filter(n=>n.dataset.axStep);
 key(stepButtons[1],'ArrowDown');assert.equal(current(host).dataset.axStep,lessons.steps[2].id);
 if(type==='transformer'){
  const original=JSON.stringify(explorer.getTrace().logits);
  field(host,'block',1);assert.equal(JSON.stringify(explorer.getTrace().logits),original);
  explorer.choose('block-input');assert.equal(explorer.getTrace().cfg.block,1);assert.match(content(host),/earlier block/);
  const bypass=all(host).find(n=>n.getAttribute('data-edge')==='residual-one');
  assert.equal(bypass.getAttribute('data-from'),'block-input');bypass.onclick();assert.equal(current(host).dataset.axStep,'residual1');
  field(host,'heads',4);field(host,'head',3);field(host,'heads',1);assert.equal(explorer.getTrace().cfg.head,0);
  field(host,'layers',1);assert.equal(explorer.getTrace().cfg.block,0);
  explorer.choose('scores');const inset=all(host).find(n=>n.className==='ax-inset-svg');assert.ok(inset);
  all(inset).find(n=>n.getAttribute('aria-label')==='Block future positions').onclick();assert.equal(current(host).dataset.axStep,'mask');
  explorer.choose('generation');
  let i=0;while(i++<6){const b=all(host).find(n=>n.textContent==='Append one example token');if(b.disabled)break;b.onclick();}
  assert.ok(explorer.getTrace().ids.length<=6);assert.ok(all(host).find(n=>n.textContent==='Append one example token').disabled);
 }else{
  field(host,'row',7);field(host,'col',7);field(host,'stride',2);
  assert.equal(explorer.getTrace().cfg.row,3);assert.equal(explorer.getTrace().cfg.col,3);
  field(host,'kernel',5);field(host,'pattern','cross');field(host,'filter',1);
  explorer.choose('conv1');
  const cells=all(host).filter(n=>n.className==='ax-cell');assert.equal(cells.length,16);
  cells[0].onclick();assert.equal(explorer.getTrace().cfg.row,0);assert.equal(explorer.getTrace().cfg.col,0);
  assert.match(content(host),/Entrywise products/);assert.equal(current(host).dataset.axStep,'conv1');
 }
 all(host).find(n=>n.textContent==='Reset example').onclick();
 assert.equal(explorer.getTrace().cfg.stride??explorer.getTrace().cfg.layers,type==='cnn'?1:2);
 for(const n of all(host).filter(n=>n.tag==='marker'))markerIds.push(n.id||n.getAttribute('id'));
 if(out)fs.writeFileSync(path.join(out,type+'.svg'),serialize(all(host).find(n=>n.className==='ax-svg')));
 if(out&&type==='transformer'){explorer.choose('scores');fs.writeFileSync(path.join(out,'attention-inset.svg'),serialize(all(host).find(n=>n.className==='ax-inset-svg')));}
 if(out&&type==='cnn'){explorer.choose('conv1');fs.writeFileSync(path.join(out,'convolution-inset.svg'),serialize(all(host).find(n=>n.className==='ax-inset-svg')));}
 explorer.destroy();assert.equal(host.children.length,0);explorer.choose('image');assert.equal(host.children.length,0);
}
assert.equal(new Set(markerIds).size,markerIds.length,'No SVG marker ID collisions');
const css=fs.readFileSync(path.join(web,'style.css'),'utf8');
assert.match(css,/\.ax-diagram-frame\{height:650px;max-height:76vh;overflow:auto/);
assert.match(css,/@container \(max-width:780px\)/);
assert.equal((css.match(/{/g)||[]).length,(css.match(/}/g)||[]).length);
console.log('PASS: 37 linked operations, actual equation rendering ('+equations+'), box/edge/guide keyboard selection, input/settings recomputation, bounded generation, cell inspection, unique IDs and teardown. DOM stand-ins, not browser sign-off.');
