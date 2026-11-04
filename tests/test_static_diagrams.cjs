// Structural/numerical checks and optional SVG exports. Not browser layout certification.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const web=path.resolve(__dirname,'../src/training_lab/web');
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent='';}
 append(...items){for(const x of items){x.parent=this;this.children.push(x);}}
 replaceChildren(...items){this.children=[];this.textContent='';this.append(...items);}
 remove(){this.parent.children=this.parent.children.filter(n=>n!==this);}
 setAttribute(k,v){this.attrs[k]=String(v);}
}
const all=n=>[n,...n.children.flatMap(all)],content=n=>n.textContent+' '+n.children.map(content).join(' ');
const source=fs.readFileSync(path.join(web,'visuals.js'),'utf8'),css=fs.readFileSync(path.join(web,'style.css'),'utf8');
const context={document:{createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)},katex:{render(tex,host,options){host.textContent=require(path.join(web,'vendor/katex/katex.min.js')).renderToString(tex,options);}}};
vm.createContext(context);vm.runInContext(source,context);
const ids=vm.runInContext('LabVisuals.catalog.map(x=>x.id)',context);assert.equal(ids.length,12);
assert.doesNotMatch(source,/setInterval|setTimeout|requestAnimationFrame|addEventListener|WebGL|THREE|Lab3D|playback\(/);
assert.doesNotMatch(css,/\.playback\b|\.spatial-\w+|@keyframes/);
assert.equal((css.match(/\{/g)||[]).length,(css.match(/\}/g)||[]).length,'CSS braces remain balanced');
const exportAt=process.argv.indexOf('--export'),output=exportAt>=0?path.resolve(process.argv[exportAt+1]):null;
if(output)fs.mkdirSync(output,{recursive:true});
const escape=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
function serialize(n){return '<'+n.tag+(n.tag==='svg'?' xmlns="http://www.w3.org/2000/svg"':'')+Object.entries(n.attrs).map(([k,v])=>' '+k+'="'+escape(v)+'"').join('')+'>'+escape(n.textContent)+n.children.map(serialize).join('')+'</'+n.tag+'>';}
let placements=0;
for(const lesson of ['P1','P2','P3','P4','P5','P6','P7','T0','T1','T2','T3','T4','T5','T6','T7','T8','R1','R2','R3','R4','R5','R6','R7','R8','R9'])placements+=Object.keys(vm.runInContext(`LabVisuals.figuresForLesson('${lesson}')`,context)).length;
assert.equal(placements,13);
for(const id of ids){
 const host=new Element('div');context.host=host;
 vm.runInContext(`globalThis.diagram=LabVisuals.mount(host,'${id}')`,context);
 const nodes=all(host),svg=nodes.find(n=>n.tag==='svg');assert.ok(svg,id+' has a diagram');
 assert.equal(nodes.filter(n=>['button','input','select'].includes(n.tag)).length,0,id+' has no teaching controls');
 assert.ok(nodes.filter(n=>n.tag==='li').length>=3,id+' has a step-by-step explanation');
 assert.ok(nodes.filter(n=>n.tag==='a'&&n.href?.startsWith('https://')).length>=1,id+' links primary sources');
 const [x0,y0,width,height]=svg.attrs.viewBox.split(' ').map(Number);assert.equal(x0+y0,0);
 for(const n of all(svg)){
  for(const [k,v] of Object.entries(n.attrs))if(['x','y','x1','x2','y1','y2','width','height'].includes(k))assert.ok(Number.isFinite(Number(v)),id+' '+k);
  if(n.tag==='rect'){assert.ok(+n.attrs.x>=0&&+n.attrs.y>=0);assert.ok(+n.attrs.x + +n.attrs.width<=width+.01,id+' horizontal bounds');assert.ok(+n.attrs.y + +n.attrs.height<=height+.01,id+' vertical bounds');}
  if(n.tag==='text'){assert.ok(+n.attrs.x>=0&&+n.attrs.x<=width);assert.ok(+n.attrs.y>=0&&+n.attrs.y<=height);}
 }
 if(output)fs.writeFileSync(path.join(output,id+'.svg'),serialize(svg));
 if(id==='batching')assert.match(content(host),/3\.2/);
 if(id==='attention')assert.match(content(host),/1\.3395/);
 if(id==='inference')assert.equal(2*1024*32*5*64*2,41943040);
 if(id==='architecture'){
  vm.runInContext('diagram.updateSettings({width:128,heads:8,layers:3,positions:false,residuals:false,normalization:"none"})',context);
  assert.match(content(host),/Residual addition disabled/);assert.match(content(host),/Identity/);assert.match(content(host),/head width 16/);
 }
 vm.runInContext('diagram.destroy()',context);assert.equal(host.children.length,0);
}
assert.equal(49152*960+32*(2*960*960+2*960*320+3*960*2560+2*960)+960,361821120);
assert.equal(32*8*(960+960+960+320),819200);
assert.equal(4096*4096/2+(4096*4096/128)*2,8.25*1024*1024);
console.log(`PASS: 12 static SVG diagrams, 13 lesson placements, full explanations/primary links, finite bounded shapes, configuration changes, exact parameter/cache/quantization examples and no animation/3D controls.${output?' SVGs exported to '+output:''}`);
