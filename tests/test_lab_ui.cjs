// Executes the shipped lab controllers against DOM stand-ins, not a visual browser.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const site=path.resolve(__dirname,'../site'),html=fs.readFileSync(path.join(site,'lab.html'),'utf8');
const snapshots=JSON.parse(fs.readFileSync(path.join(site,'api-data.json'),'utf8'));
const settle=async()=>{for(let i=0;i<12;i++)await new Promise(r=>setImmediate(r));};
const elements={},all=[],requests=[];
class Element{
 constructor(tag='div',namespaceURI=null){Object.assign(this,{tag,namespaceURI,children:[],value:'',hidden:false,disabled:false,textContent:'',dataset:{},listeners:{},attributes:{},classList:{toggle(){}}});all.push(this);}
 set id(value){this._id=value;elements[value]=this;}get id(){return this._id;}
 get options(){return this.children.filter(c=>c.tag==='option');}get firstChild(){return this.children[0];}get isConnected(){return true;}
 append(...items){for(const value of items){const n=typeof value==='string'?new Element('#text'):value;if(typeof value==='string')n.textContent=value;n.parent=this;this.children.push(n);}if(this.tag==='select'&&!this.value)this.value=this.options[0]?.value||'';}
 prepend(...items){for(const n of items)n.parent=this;this.children.unshift(...items);}
 replaceChildren(...items){this.children=[];this.textContent='';if(this.tag==='select')this.value='';this.append(...items);}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
 setAttribute(k,v){this.attributes[k]=String(v);if(k==='class')this.className=v;}getAttribute(k){return this.attributes[k];}removeAttribute(k){delete this.attributes[k];}
 addEventListener(k,f){(this.listeners[k]??=[]).push(f);}removeEventListener(k,f){this.listeners[k]=(this.listeners[k]||[]).filter(x=>x!==f);}
 scrollIntoView(){}getBoundingClientRect(){return {width:600,height:270};}
 getContext(){return {...Object.fromEntries(['scale','setTransform','clearRect','fillText','beginPath','moveTo','lineTo','stroke','save','translate','rotate','restore','fillRect','arc','fill','putImageData'].map(k=>[k,()=>{}])),createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)})};}
}
const document=new Element('document');Object.assign(document,{hidden:false,getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag,ns),createTextNode:t=>{const n=new Element('#text');n.textContent=t;return n;},querySelector:()=>null,querySelectorAll:q=>q==='.page'?all.filter(e=>e.page):q==='.run-tab'?all.filter(e=>e.runTab):q==='nav button'?[]:q.match(/^\[data-(\w+)\]$/)?all.filter(e=>q.slice(6,-1) in e.dataset):[]});
for(const m of html.matchAll(/<(\w+)\b[^>]*\bid="([^"]+)"[^>]*>/g)){
 assert.ok(!elements[m[2]],'Unique IDs');const e=new Element(m[1]);e.id=m[2];e.page=/class="page"/.test(m[0]);e.runTab=/class="run-tab"/.test(m[0]);e.hidden=/\bhidden\b/.test(m[0]);
 for(const [k,v] of [...m[0].matchAll(/([\w-]+)="([^"]*)"/g)].map(x=>[x[1],x[2]]))e.setAttribute(k,v);if(e.attributes.value)e.value=e.attributes.value;
}
for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g))for(const o of m[2].matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g)){const n=new Element('option');n.value=o[1].match(/value="([^"]*)"/)?.[1]||o[2];elements[m[1]].append(n);}
for(const m of html.matchAll(/<button\b[^>]*data-(page|engine|tab|subject)="([^"]+)"[^>]*>/g)){const id=m[0].match(/\bid="([^"]+)"/)?.[1],n=id?elements[id]:new Element('button');n.dataset[m[1]]=m[2];}
const context={console,URL,URLSearchParams,document,location:{search:'?page=ml-lab&course=ml',href:'https://example.github.io/study-lab/lab.html?page=ml-lab&course=ml'},window:{LAB_STATIC_PREVIEW:true,devicePixelRatio:1,addEventListener(){},localStorage:{getItem(){return null},setItem(){}}},setInterval(){return 1},clearInterval(){},katex:{render(tex,n){n.textContent=tex}},fetch:async(url,options)=>{
 requests.push({url,options});let data=snapshots[url];
 if(url==='/api/preview')data={runs:[{label:'Reference',settings:snapshots['/api/catalog'].defaults,estimate:{effective_windows_per_update:4}}]};
 assert.notEqual(data,undefined,'Unexpected endpoint '+url);return {ok:true,json:async()=>data};
}};
context.window.history={replaceState(state,title,value){const url=new URL(value);Object.assign(context.location,{href:url.href,search:url.search,hash:url.hash});}};
vm.createContext(context);
for(const m of html.matchAll(/<script\b[^>]*src="([^"]+)"[^>]*>/g)){
 const filename=m[1].replace(/^\.\//,'');if(filename==='preview.js'||filename==='reader.js'||filename.startsWith('vendor/'))continue;
 vm.runInContext(fs.readFileSync(path.join(site,filename),'utf8'),context,{filename});
}
(async()=>{
 await vm.runInContext('labReady',context);await settle();
 assert.equal(elements['model-lab'].hidden,false);assert.ok(elements['model-method'].options.length>0);
 const pages={'ml-lab':'model-lab','ml-results':'model-results','ml-compare':'model-compare','dl-lab':'model-lab','dl-results':'model-results','dl-compare':'model-compare',train:'train',explore:'explore',compare:'compare'};
 for(const [page,target] of Object.entries(pages)){
  vm.runInContext('show('+JSON.stringify(page)+')',context);await settle();assert.equal(elements[target].hidden,false,page);
  assert.ok(elements['model-start'].disabled,'No online model training');assert.ok(elements['run-training'].disabled,'No online LLM training');
 }
 context.location.search='?page=dl-lab&course=nlp';context.location.href='https://example.github.io/study-lab/lab.html'+context.location.search;
 vm.runInContext('show("dl-lab")',context);await settle();assert.equal(elements['page-title'].textContent,'NLP training lab');
 assert.ok(elements['model-experiment-review'].children.length>0,'Summary and code are rendered');
 vm.runInContext('setEngine("scratch")',context);await settle();assert.ok(elements['training-experiment-review']||elements['scratch-review']||elements['run-training']);
 for(const n of all)assert.notEqual(n.className,'error-visible');
 assert.equal(elements.home.children.length,0,'No bootstrap error');
 assert.ok(!requests.some(r=>r.url.includes('learning')||r.url.includes('private')),'No excluded readers');
 assert.ok(requests.every(r=>!r.options?.body||r.url==='/api/preview'),'No training launched');
 console.log('PASS: all nine lab routes, NLP context, populated controls/code and disabled training (DOM stand-ins).');
})().catch(error=>{console.error(error);process.exitCode=1;});
