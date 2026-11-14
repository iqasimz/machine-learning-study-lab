const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const site=path.resolve(__dirname,'../site'),snapshots=JSON.parse(fs.readFileSync(path.join(site,'api-data.json'),'utf8'));
const calls=[];
const context={URL,Response,location:{href:'https://example.github.io/study-lab/lab.html?page=train',origin:'https://example.github.io'},
 document:{currentScript:{src:'https://example.github.io/study-lab/preview.js'},addEventListener(){}},
 window:{fetch:async url=>{calls.push(String(url));return new Response(JSON.stringify(snapshots));}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(site,'preview.js'),'utf8'),context);
(async()=>{
 for(const endpoint of Object.keys(snapshots)){
  const response=await context.window.fetch(endpoint);assert.equal(response.status,200,endpoint);
  assert.deepEqual(await response.json(),snapshots[endpoint]);
 }
 assert.equal(calls.length,1);assert.equal(calls[0],'https://example.github.io/study-lab/api-data.json');
 for(const endpoint of ['/api/train','/api/adapt','/api/models/train','/api/models/prepare','/api/compare','/api/cancel','/api/export','/api/resume','/api/tokenize']){
  assert.equal((await context.window.fetch(endpoint,{method:'POST',body:'{}'})).status,405,endpoint);
 }
 assert.equal((await context.window.fetch('/api/learning')).status,404);
 const registry=snapshots['/api/catalog'];
 for(const guide of registry.guides){
  const response=await context.window.fetch('/api/preview',{method:'POST',body:JSON.stringify({guide:guide.id,variation:'all',overrides:{}})});
  assert.equal(response.status,200,guide.id);assert.equal((await response.json()).runs.length,guide.variations.length);
 }
 console.log('PASS: repository subpath, cached catalogs, guided planning and blocked write operations.');
})().catch(error=>{console.error(error);process.exitCode=1;});
