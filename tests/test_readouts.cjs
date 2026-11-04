// Explanations must follow saved evidence, including absent/incomparable cases.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict'),path=require('path');
const c=vm.createContext({});vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/training_lab/web/readouts.js'),'utf8'),c);
const r=vm.runInContext('LabReadouts',c);
for(const key of ['accuracy','macro_f1','balanced_accuracy','r2'])assert.equal(r.metric(key).higher,true);
for(const key of ['mae','rmse','log_loss','brier_multiclass','ece_10_bins','dev_nll','test_bits_per_byte'])assert.equal(r.metric(key).higher,false);
assert.equal(r.metric('unknown').higher,null);
assert.equal(r.difference(.7,.6),'+0.1000');assert.equal(r.difference(.5,.6),'-0.1000');assert.equal(r.difference(null,0),'not recorded');
const model={manifest:{status:'MEASURED',settings:{task:'classification'}},result:{predictions:{dev:{scores:{macro_f1:.7}}},baseline:{dev:{macro_f1:.6}}}};
assert.match(r.modelResult(model).text,/0.7000.*0.6000.*\+0.1000/);
assert.match(r.modelResult(model).text,/Higher is better/);
model.manifest.settings.task='regression';model.result.predictions.dev.scores={mae:0};model.result.baseline.dev={mae:1};
assert.match(r.modelResult(model).text,/0.0000.*1.0000.*-1.0000/);assert.match(r.modelResult(model).text,/Lower is better/);
model.manifest.status='CANCELLED';assert.doesNotMatch(r.modelResult(model).text,/0\.0000/);
assert.match(r.modelResult(null).title,/No saved/);
assert.match(r.modelResult({manifest:{status:'MEASURED',settings:{}},result:{}}).title,/No recognized score/);
for(const task of ['classification','regression','representation','clustering','anomaly']){
 const text=r.modelComparison({task,rows:[{id:'reference'}]},task==='regression'?'mae':'macro_f1').text;
 assert.ok(text.length>150);assert.doesNotMatch(text,/NaN|undefined/);
}
const run={manifest:{status:'MEASURED',engine:'lora'},evaluation:{checkpoint_step:0,pretrained:{dev:{nll:2}},transformer:{dev:{nll:2}}}};
assert.match(r.languageResult(run).text,/2.0000.*2.0000/);assert.match(r.languageResult(run).text,/checkpoint is update 0/);
assert.match(r.languageResult(run).text,/not necessarily more factual/);
assert.match(r.languageComparison({},'').text,/not zero loss/);
assert.match(r.languageComparison({},'dev_bits_per_byte').text,/raw text byte/);
assert.doesNotMatch(r.languageComparison({},'dev_nll').text,/NaN|undefined/);
console.log('PASS: saved-data interpretations, metric directions, zero/missing values, incomplete runs, all task families and incompatible language comparisons. No inferred winner or invented measurement.');
