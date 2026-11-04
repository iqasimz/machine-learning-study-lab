// Static SVG contract: definitions, finite geometry, reading order, sources, cleanup.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const web=path.resolve(__dirname,'../src/training_lab/web');
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent='';}
 append(...children){children.forEach(c=>{c.parent=this;this.children.push(c);});}
 replaceChildren(...children){this.children=[];this.textContent='';this.append(...children);}
 setAttribute(k,v){this.attrs[k]=String(v);}
 remove(){this.parent.children=this.parent.children.filter(x=>x!==this);}
}
const all=n=>[n,...n.children.flatMap(all)],text=n=>n.textContent+' '+n.children.map(text).join(' ');
const source=fs.readFileSync(path.join(web,'ml_visuals.js'),'utf8');
assert.doesNotMatch(source,/setInterval|setTimeout|requestAnimationFrame|innerHTML|WebGL|THREE|addEventListener/);
const context={document:{createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)}};
vm.createContext(context);vm.runInContext(source,context);
for(const id of ['selection','ensembles','pca','backprop','convolution','recurrence']){
 const host=new Element('div');context.host=host;vm.runInContext(`globalThis.figure=ModelFigures.mount(host,'${id}')`,context);
 const nodes=all(host),svg=nodes.find(n=>n.tag==='svg');assert.ok(svg);
 assert.equal(nodes.filter(n=>['button','input','select'].includes(n.tag)).length,0);
 assert.equal(nodes.filter(n=>n.tag==='li').length,3);
 assert.ok(nodes.filter(n=>n.tag==='a'&&n.href?.startsWith('https://')).length>=1);
 assert.ok(text(host).length>1000);assert.doesNotMatch(text(host),/NaN|Infinity/);
 const terms=nodes.findIndex(n=>n.className==='topic-terms');assert.ok(terms>=0&&terms<nodes.indexOf(svg));
 const [x,y,width,height]=svg.attrs.viewBox.split(' ').map(Number);assert.equal(x+y,0);assert.equal(width,720);
 for(const n of all(svg)){
  for(const [k,v] of Object.entries(n.attrs))if(['x','y','cx','cy','x1','x2','y1','y2','width','height','r','font-size'].includes(k))assert.ok(Number.isFinite(+v),`${id}: ${k}`);
  if(n.tag==='rect'){assert.ok(+n.attrs.x>=0&&+n.attrs.y>=0);assert.ok(+n.attrs.x + +n.attrs.width<=width+.01);assert.ok(+n.attrs.y + +n.attrs.height<=height+.01);}
  if(n.tag==='text'){assert.ok(+n.attrs.x>=0&&+n.attrs.x<=width);assert.ok(+n.attrs.y>=0&&+n.attrs.y<=height);if(n.attrs['font-size'])assert.ok(+n.attrs['font-size']>=12);}
 }
 if(id==='backprop')assert.match(text(host),/2\.6/);
 if(id==='convolution'){assert.match(text(host),/10 parameters/);assert.match(text(host),/Output = −1/);assert.equal(3*3+1,10);}
 vm.runInContext('figure.destroy()',context);assert.equal(host.children.length,0);
}
console.log('PASS: six static ML/DL SVGs; definitions before figures, bounded finite geometry, legible declared text sizes, three-step explanations, primary sources and cleanup. Not browser layout certification.');
