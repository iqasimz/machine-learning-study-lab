// Regression for Retina redraw growth. A canvas height property also changes its
// HTML height attribute; the old chart() read that bitmap size as CSS height.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const web=path.resolve(__dirname,"../src/training_lab/web");
const app=fs.readFileSync(path.join(web,"app.js"),"utf8");
const html=fs.readFileSync(path.join(web,"index.html"),"utf8");
const css=fs.readFileSync(path.join(web,"style.css"),"utf8");
const start=app.indexOf("function chart("),end=app.indexOf("\nfunction renderLive()",start);
assert.ok(start>=0&&end>start);
const chartSource=app.slice(start,end);
const series=[{color:"#315d9c",values:[{x:0,y:2},{x:10,y:1.5}]}];
for(const [id,height] of [["live-loss",270],["live-rate",270],["curve",290],["comparison-curve",300],["model-live-curve",270],["model-result-curve",290],["model-comparison-curve",300]]){
  assert.match(html,new RegExp('<div class="chart-frame[^"]*"><canvas id="'+id+'"'));
  assert.doesNotMatch(html,new RegExp('<canvas id="'+id+'"[^>]*height='));
  for(const ratio of [1,1.25,2,3]){
    let width=640,displayHeight=height,clears=0,transform;
    const ctx=Object.fromEntries(["fillText","beginPath","moveTo","lineTo","stroke","fillRect"].map(k=>[k,()=>{}]));
    ctx.setTransform=(...values)=>{transform=values;};
    ctx.clearRect=(x,y,w,h)=>{clears++;assert.equal(w,width);assert.equal(h,displayHeight);};
    const canvas={width:300,height:150,getBoundingClientRect:()=>({width,height:displayHeight}),
      getAttribute(name){return String(this[name]);},getContext:()=>ctx};
    const sandbox={window:{devicePixelRatio:ratio},canvas,series};vm.createContext(sandbox);vm.runInContext(chartSource,sandbox);
    for(let redraw=0;redraw<500;redraw++){
      vm.runInContext("chart(canvas,series)",sandbox);
      assert.equal(canvas.width,Math.round(width*ratio),id+" stable bitmap width");
      assert.equal(canvas.height,Math.round(height*ratio),id+" stable bitmap height");
      assert.equal(transform[0],canvas.width/width);assert.equal(transform[3],canvas.height/height);
    }
    // Resize, zoom-density change, hidden tab and empty data must not inflate it.
    width=347.5;vm.runInContext("chart(canvas,series)",sandbox);assert.equal(canvas.width,Math.round(width*ratio));assert.equal(canvas.height,Math.round(height*ratio));
    sandbox.window.devicePixelRatio=2;vm.runInContext("chart(canvas,[])",sandbox);assert.equal(canvas.height,height*2);
    const priorClears=clears;width=0;displayHeight=0;vm.runInContext("chart(canvas,series)",sandbox);assert.equal(clears,priorClears);
    width=640;displayHeight=height;vm.runInContext("chart(canvas,series)",sandbox);assert.equal(canvas.height,height*2);
  }
}
assert.match(css,/\.chart-frame\{[^}]*height:270px;[^}]*max-height:300px;[^}]*overflow:hidden/);
assert.match(css,/\.chart-frame>canvas\{[^}]*position:absolute;[^}]*height:100%/);
console.log("PASS: all seven ML/DL/LLM charts remain bounded across 14,000 redraws at four pixel densities, resizing, hidden/empty states and density changes. Programmatic sizing checks, not browser layout QA.");
