"use strict";

const ArchitectureExplorer=(()=>{
  const NS="http://www.w3.org/2000/svg",M=ArchitectureMath;
  let instance=0;
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const sv=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;};
  const placement={T3:{"1":"transformer"},DL08:{"1":"DL08"},DL05:{"1":"cnn"}};
  function mount(host,type,options={}){
    const custom=typeof ConceptLessons!=="undefined"&&ConceptLessons.has(type);
    if(!custom&&!["transformer","cnn"].includes(type))throw Error("Unknown architecture explorer.");
    const prefix="ax-"+(++instance)+"-",root=el("section",undefined,"architecture-explorer");
    root.dataset.architecture=type;root.setAttribute("aria-label",custom?"Interactive lesson walkthrough":type==="cnn"?"Interactive CNN walkthrough":"Interactive Transformer walkthrough");
    const heading=el("h3"),intro=el("p",undefined,"ax-intro"),overview=el("div",undefined,"ax-topic-overview"),instructions=el("p","Select a component, connection or guide entry to inspect it.","ax-instructions");
    heading.hidden=options.showHeading===false;
    const settings=el("div",undefined,"ax-settings"),layout=el("div",undefined,"ax-layout"),left=el("div",undefined,"ax-diagram-column"),frame=el("div",undefined,"ax-diagram-frame");
    frame.setAttribute("tabindex","0");frame.setAttribute("aria-label","Architecture diagram; scroll within this frame to follow lower operations");
    const legend=el("p",type==="transformer"?"Blue = selected operation and connected paths. Amber bypasses preserve earlier features. Dashed paths belong to training; the outer right return belongs to generation. Select any path for its meaning.":"Blue = selected operation and connected paths. Solid arrows carry features forward; dashed arrows compare with a target and illustrate a classifier update.","ax-legend");
    const right=el("div",undefined,"ax-guide-column"),flow=el("ol",undefined,"ax-flow"),detail=el("div",undefined,"ax-detail"),status=el("p",undefined,"ax-status");
    status.setAttribute("role","status");status.setAttribute("aria-live","polite");
    const nav=el("div",undefined,"ax-nav"),back=el("button","Previous entry","secondary"),next=el("button","Next entry","primary"),reset=el("button","Reset example","text-button");
    back.type=next.type=reset.type="button";nav.append(back,next,reset);
    const flowLabel=el("h4","Guide"),diagramLabel=el("h4","Diagram");flow.id=prefix+"flow";flow.setAttribute("aria-label","Operations and further detail");
    const sectionList=el("div",undefined,"ax-section-list");sectionList.append(flowLabel,flow,nav,status);
    right.append(sectionList,detail);left.append(diagramLabel,frame,legend);layout.append(left,right);
    const links=el("p",undefined,"ax-sources");root.append(heading,overview,intro,instructions,settings,layout,links);host.replaceChildren(root);
    let cfg={},trace,lesson,selected=type==="cnn"?"image":"tokens",selectedEdge=null,dead=false;
    let nodes=[],edges=[],guide=[],fields=new Map(),insetNodes=new Map();
    function button(label,fn,cls="text-button"){const b=el("button",label,cls);b.type="button";b.onclick=fn;return b;}
    function field(key,label,choices,value,help){
      const wrap=el("label",undefined,"ax-setting"),name=el("span",label),select=el("select");
      select.dataset.axField=key;select.setAttribute("aria-label",label);
      for(const [val,text] of choices){const o=el("option",text);o.value=String(val);select.append(o);}select.value=String(value);
      select.onchange=()=>{
        cfg[key]=typeof value==="string"?select.value:Number(select.value);
        if(type==="transformer"){cfg.head=Math.min(cfg.head??0,(cfg.heads??2)-1);cfg.block=Math.min(cfg.block??0,(cfg.layers??2)-1);}
        else if(type==="cnn"){const size=Math.ceil(8/(cfg.stride??1));cfg.row=Math.min(cfg.row??2,size-1);cfg.col=Math.min(cfg.col??2,size-1);}
        selectedEdge=null;render();fields.get(key)?.focus?.({preventScroll:true});
      };
      wrap.append(name,el("small",help),select);settings.append(wrap);fields.set(key,select);
    }
    function controls(){
      fields=new Map();settings.replaceChildren();
      if(custom){for(const c of trace.controls)field(c.key,c.label,c.choices,trace.cfg[c.key],c.help);settings.hidden=!trace.controls.length;return;}
      if(type==="transformer"){
        const c=trace.cfg;field("heads","Attention heads",[[1,"1 head · 4 features"],[2,"2 heads · 2 features each"],[4,"4 heads · 1 feature each"]],c.heads,"Hidden width stays four; the head split and score scaling change.");
        field("layers","Blocks in the stack",[[1,"1 block"],[2,"2 blocks"],[3,"3 blocks"]],c.layers,"Different blocks use different fixed example matrices.");
        field("block","Inspect block",Array.from({length:c.layers},(_,i)=>[i,"Block "+(i+1)]),c.block,"The diagram shows this block; output scores still use the whole stack.");
        field("position","Inspect input position",trace.ids.map((id,i)=>[i,i+": "+trace.vocabulary[id]]),c.position,"Rows use this query position. Future keys remain masked.");
        field("head","Inspect attention head",Array.from({length:c.heads},(_,i)=>[i,"Head "+(i+1)]),c.head,"Changes the detailed score/weight tables, not the model.");
      }else{
        const c=trace.cfg,size=trace.z1[0].length;
        field("pattern","Input pattern",[["vertical","Vertical stroke"],["horizontal","Horizontal stroke"],["cross","Cross"]],c.pattern,"Authored grayscale patterns, not an accuracy benchmark.");
        field("kernel","First filter size",[[3,"3 × 3"],[5,"5 × 5"]],c.kernel,"Zero-padding is adjusted to (kernel−1)/2 in this example.");
        field("stride","First convolution stride",[[1,"1 pixel"],[2,"2 pixels"]],c.stride,"Larger movement creates a smaller output grid and classifier head.");
        field("filter","Inspect first-layer filter",[[0,"Filter 0"],[1,"Filter 1"]],c.filter,"Each shared filter produces its own feature map.");
        field("row","Output row",Array.from({length:size},(_,i)=>[i,i]),c.row,"Zero-based row in the first convolution's output.");
        field("col","Output column",Array.from({length:size},(_,i)=>[i,i]),c.col,"Change this or select a cell in the output map below.");
      }
    }
    function shape(channel){return channel.length+" × "+channel[0].length+" × "+channel[0][0].length;}
    function draw(){
      if(custom){drawConcept();return;}
      nodes=[];edges=[];frame.replaceChildren();
      const height=type==="cnn"?750:978,svg=sv("svg",{viewBox:"0 0 420 "+height,class:"ax-svg",role:"group","aria-label":type==="cnn"?"CNN forward path and training branch":"Pre-norm decoder, residual bypasses, generation return and training branch"});
      svg.append(sv("title",{},type==="cnn"?"CNN computation graph":"Transformer computation graph"));
      svg.append(sv("desc",{},"Boxes and paths are selectable. The ordered HTML guide provides the same operations without needing the diagram."));
      const defs=sv("defs"),marker=sv("marker",{id:prefix+"arrow",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:6,markerHeight:6,orient:"auto-start-reverse"});
      marker.append(sv("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"context-stroke"}));defs.append(marker);svg.append(defs);
      const paths=sv("g"),boxes=sv("g");svg.append(paths,boxes);frame.append(svg);
      function node(id,label,sub,y,ids=[id]){
        const group=sv("g",{class:"ax-node",role:"button",tabindex:0,"aria-label":label+". "+sub,"data-node":id});
        group.append(sv("rect",{x:104,y,width:222,height:40,rx:7}),sv("text",{x:215,y:y+16,"text-anchor":"middle","font-size":14},label),sv("text",{x:215,y:y+32,"text-anchor":"middle","font-size":11,class:"ax-shape"},sub));
        group.onclick=()=>choose(ids[0]);group.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();choose(ids[0]);}};
        boxes.append(group);nodes.push({id,ids,group,y});return {id,y};
      }
      function edge(id,from,to,d,select=to,kind="forward",explanation=""){
        const g=sv("g",{class:"ax-edge",role:"button",tabindex:0,"aria-label":explanation||"Follow "+from+" to "+to,"data-edge":id,"data-from":from,"data-to":to,"data-kind":kind});
        g.append(sv("path",{d,class:"ax-edge-hit",fill:"none",stroke:"transparent","stroke-width":15}));
        g.append(sv("path",{d,class:"ax-edge-line",fill:"none","stroke-width":2,"marker-end":"url(#"+prefix+"arrow)"}));
        g.onclick=()=>choose(select,id);g.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();choose(select,id);}};
        paths.append(g);edges.push({id,from,to,select,kind,group:g,explanation});
      }
      const chain=[];
      if(type==="transformer"){
        const n=trace.ids.length,c=trace.cfg;
        const boundary=sv("rect",{x:84,y:236,width:260,height:332,rx:10,class:"ax-block-boundary"});paths.append(boundary);
        boxes.append(sv("text",{x:94,y:249,"font-size":11,class:"ax-diagram-label"},"Block "+(c.block+1)+" / "+c.layers));
        chain.push(node("tokens","Token IDs","1 example × "+n+" positions",20),node("embedding","Token embedding",n+" × 4 features",74),node("position","Add position vectors",n+" × 4",128));
        chain.push(node("block-input","Selected block input",c.block?c.block+" earlier block(s) calculated":"Embedding + position result",182));
        chain.push(node("norm1","Normalize",n+" × 4",254));
        chain.push(node("attention","Causal self-attention",c.heads+" heads · select for calculations",308,["qkv","heads","scores","mask","attention-softmax","mix","project"]));
        chain.push(node("residual1","Add original block input",n+" × 4",362),node("norm2","Normalize",n+" × 4",416));
        chain.push(node("mlp","Feed-forward MLP","4 → 8 → 4 features per token",470,["expand","activation","contract"]));
        chain.push(node("residual2","Add pre-MLP representation",n+" × 4",524),node("stack","Continue through later blocks",(c.layers-c.block-1)+" later block(s) remaining",586));
        chain.push(node("final-norm","Final normalization",n+" × 4",640),node("logits","Vocabulary projection","4 → 6 scores at each position",694),node("vocab-softmax","Vocabulary softmax","6 probabilities per position",748));
        node("generation","Choose and append one token","Inference · weights stay fixed",802);
        node("loss","Compare with known target","Training · one scalar loss",856);
        node("update","Gradient → optimizer update","Head-only teaching calculation",910);
        edge("residual-one","block-input","residual1","M 104 202 H 48 V 382 H 104","residual1","bypass","Bypass 1 carries the selected block input unchanged to addition. Not a backwards loop.");
        edge("residual-two","residual1","residual2","M 326 382 H 367 V 544 H 326","residual2","bypass","Bypass 2 carries the first residual result unchanged around the MLP.");
        edge("infer","vocab-softmax","generation","M 215 788 V 802","generation","inference","Generation selects a token from the LAST input position; no parameter changes.");
        edge("autoregression","generation","tokens","M 326 822 H 396 V 40 H 326","generation","inference","Append the selected token and return to the input path. This is generation, not backpropagation.");
        edge("target-loss","vocab-softmax","loss","M 104 768 H 78 V 876 H 104","loss","training","For training, compare with a known target instead of sampling and appending.");
        edge("backward","loss","update","M 215 896 V 910","update","training","Backpropagation computes gradients; the optimizer then changes parameters.");
        edge("head-update","update","logits","M 104 930 H 22 V 714 H 104","update","training","This limited calculation updates only the output matrix. Full training also differentiates through the upstream blocks.");
      }else{
        const p=trace;
        chain.push(node("image","Grayscale image","1 × 1 × 8 × 8",20),node("conv1","Convolution 1",shape(p.z1)+" · "+p.parameters.conv1+" parameters",78));
        chain.push(node("relu1","ReLU",shape(p.a1)+" · no learned weights",136),node("pool1","2 × 2 max pool",shape(p.p1),194));
        chain.push(node("conv2","Convolution 2",shape(p.z2)+" · "+p.parameters.conv2+" parameters",252),node("relu2","ReLU",shape(p.a2),310),node("pool2","2 × 2 max pool",shape(p.p2),368));
        chain.push(node("flatten","Flatten",p.flat.length+" retained features",426),node("classifier","Dense class head","3 logits · "+p.parameters.head+" parameters",484),node("class-softmax","Class softmax","3 probabilities",542));
        chain.push(node("class-loss","Compare with target B","Training · one scalar loss",600),node("class-update","Gradient → optimizer update","Head-only teaching calculation",658));
        edge("cnn-update","class-update","classifier","M 104 678 H 42 V 504 H 104","class-update","training","A head-only update changes the classifier. Full CNN training also computes upstream gradients.");
      }
      for(let i=1;i<chain.length;i++){
        const a=chain[i-1],b=chain[i];
        edge("flow-"+a.id,a.id,b.id,"M 215 "+(a.y+40)+" V "+b.y,b.id,(b.id.includes("loss")||b.id.includes("update"))?"training":"forward","Output of "+a.id+" becomes input to "+b.id+". Select to inspect that receiving operation.");
      }
    }
    function drawConcept(){
      nodes=[];edges=[];frame.replaceChildren();
      const d=trace.diagram,svg=sv("svg",{viewBox:"0 0 "+d.width+" "+d.height,class:"ax-svg",role:"group","aria-label":trace.title});
      svg.append(sv("title",{},trace.title),sv("desc",{},"Every component and connection is selectable. The HTML guide provides the same explanation."));
      const defs=sv("defs"),marker=sv("marker",{id:prefix+"arrow",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:6,markerHeight:6,orient:"auto"});
      marker.append(sv("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#7188a0"}));defs.append(marker);svg.append(defs);
      const paths=sv("g"),boxes=sv("g");svg.append(paths,boxes);
      for(const e of d.edges){
        const a=d.nodes.find(n=>n.id===e.from),b=d.nodes.find(n=>n.id===e.to);
        let path;
        if(e.kind==="training"||e.kind==="bypass"){const rail=e.kind==="training"?26:48;path="M "+a.x+" "+(a.y+a.h/2)+" H "+rail+" V "+(b.y+b.h/2)+" H "+b.x;}
        else if(a.y===b.y)path="M "+(a.x+a.w)+" "+(a.y+a.h/2)+" H "+b.x;
        else{const ax=a.x+a.w/2,bx=b.x+b.w/2,mid=(a.y+a.h+b.y)/2;path="M "+ax+" "+(a.y+a.h)+" V "+mid+" H "+bx+" V "+b.y;}
        const note=e.note||"Output of "+a.label+" feeds "+b.label+".";
        const group=sv("g",{class:"ax-edge",role:"button",tabindex:0,"data-edge":e.id,"data-from":e.from,"data-to":e.to,"data-kind":e.kind,"aria-label":note});
        group.append(sv("path",{d:path,class:"ax-edge-hit",fill:"none",stroke:"transparent","stroke-width":15}),sv("path",{d:path,class:"ax-edge-line",fill:"none","stroke-width":2,"marker-end":"url(#"+prefix+"arrow)"}));
        group.onclick=()=>choose(e.to,e.id);group.onkeydown=event=>{if(["Enter"," "].includes(event.key)){event.preventDefault();choose(e.to,e.id);}};
        paths.append(group);edges.push({...e,group,explanation:note});
      }
      function textLines(label,max){const words=label.split(" "),lines=[];let row="";for(const word of words){if(row&&(row+" "+word).length>max){lines.push(row);row=word;}else row+=(row?" ":"")+word;}if(row)lines.push(row);return lines;}
      for(const n of d.nodes){
        const g=sv("g",{class:"ax-node",role:"button",tabindex:0,"data-node":n.id,"aria-label":n.label+". "+n.sub});
        g.append(sv("rect",{x:n.x,y:n.y,width:n.w,height:n.h,rx:7}));
        const lines=textLines(n.label,n.w<200?24:29);
        lines.forEach((line,i)=>g.append(sv("text",{x:n.x+n.w/2,y:n.y+(lines.length>1?17:22)+i*15,"text-anchor":"middle","font-size":12.5},line)));
        if(lines.length===1){const limit=n.w<200?32:40,sub=n.sub.length>limit?n.sub.slice(0,limit-1)+"…":n.sub;g.append(sv("text",{x:n.x+n.w/2,y:n.y+42,"text-anchor":"middle","font-size":9.5,class:"ax-shape"},sub));}
        g.onclick=()=>choose(n.ids[0]);g.onkeydown=event=>{if(["Enter"," "].includes(event.key)){event.preventDefault();choose(n.ids[0]);}};
        boxes.append(g);nodes.push({...n,group:g});
      }
      frame.append(svg);
    }
    function attentionInset(){
      insetNodes=new Map();
      const fig=el("figure",undefined,"ax-attention-inset"),svg=sv("svg",{viewBox:"0 0 420 302",class:"ax-inset-svg",role:"group","aria-label":"Inside attention: Q and K form scores; masked softmax weights mix V"});
      const items=[["qkv",12,14,396,34,"Project normalized tokens into Q, K, V"],["scores",12,85,245,34,"Q × Kᵀ / √head width"],["mask",12,143,245,34,"Block future positions"],["attention-softmax",12,201,245,34,"Softmax over allowed keys"],["mix",12,259,396,34,"Attention weights × V → head output"]];
      const line=(x1,y1,x2,y2)=>svg.append(sv("path",{d:"M "+x1+" "+y1+" L "+x2+" "+y2,fill:"none",stroke:"#8d9eb5","stroke-width":2,"marker-end":"url(#"+prefix+"arrow)"}));
      line(130,48,130,85);line(130,119,130,143);line(130,177,130,201);line(130,235,130,259);line(350,48,350,259);
      svg.append(sv("text",{x:140,y:70,"font-size":12},"Q and K"),sv("text",{x:358,y:154,"font-size":12},"V"));
      for(const [id,x,y,width,height,text] of items){
        const g=sv("g",{class:"ax-node",role:"button",tabindex:0,"data-selected":String(selected===id),"aria-label":text});
        g.append(sv("rect",{x,y,width,height,rx:5}),sv("text",{x:x+width/2,y:y+22,"text-anchor":"middle","font-size":13},text));
        g.onclick=()=>choose(id);g.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();choose(id);insetNodes.get(id)?.focus?.({preventScroll:true});}};insetNodes.set(id,g);svg.append(g);
      }
      fig.append(svg,el("figcaption","Q and K determine matching weights; the separate V path supplies what is mixed. Select a box here or use the ordered guide.","ax-caption"));return fig;
    }
    function convolutionInset(){
      const c=trace.cfg,k=c.kernel,p=trace.padding,size=trace.z1[0].length;
      const fig=el("figure",undefined,"ax-convolution-inset"),svg=sv("svg",{viewBox:"0 0 420 310",class:"ax-inset-svg",role:"img","aria-label":"Highlighted image patch, shared filter, entrywise products and selected output value"});
      function grid(values,x,y,cell,mode){
        const min=Math.min(...values.flat()),max=Math.max(...values.flat());
        values.forEach((row,r)=>row.forEach((value,col)=>{
          const inputPatch=mode==="input"&&r>=c.row*c.stride-p&&r<c.row*c.stride-p+k&&col>=c.col*c.stride-p&&col<c.col*c.stride-p+k;
          const selectedCell=mode==="output"&&r===c.row&&col===c.col;
          const level=max===min ? .5 : (value-min)/(max-min);
          const fill=inputPatch?"#f8d58f":selectedCell?"#a9cff3":"rgb("+Math.round(245-90*level)+","+Math.round(248-65*level)+","+Math.round(252-25*level)+")";
          svg.append(sv("rect",{x:x+col*cell,y:y+r*cell,width:cell-1,height:cell-1,fill,stroke:selectedCell?"#18588e":"#d4dee7","stroke-width":selectedCell?2:.4}));
          if(mode==="kernel"||mode==="products")svg.append(sv("text",{x:x+col*cell+cell/2,y:y+r*cell+cell*.64,"text-anchor":"middle","font-size":7.5,fill:"#24374a"},String(Number(value.toFixed(2)))));
        }));
      }
      svg.append(sv("text",{x:14,y:21,"font-size":12},"Input · 8 × 8"),sv("text",{x:164,y:21,"font-size":12},"Shared filter "+c.filter),sv("text",{x:300,y:21,"font-size":12},"Output · "+size+" × "+size));
      grid(trace.input[0],14,38,12,"input");grid(trace.f1[c.filter][0],164,48,19,"kernel");grid(trace.z1[c.filter],300,38,12,"output");
      for(const [x1,x2] of [[116,155],[264,291]])svg.append(sv("path",{d:"M "+x1+" 89 H "+x2,fill:"none",stroke:"#728ba4","stroke-width":1.5,"marker-end":"url(#"+prefix+"arrow)"}));
      svg.append(sv("text",{x:14,y:157,"font-size":11},"Amber: the input patch for output ("+c.row+", "+c.col+")."));
      svg.append(sv("text",{x:14,y:175,"font-size":11},"Same filter at every position; outside-image inputs are zero."));
      grid(trace.products,14,199,19,"products");
      const productSum=trace.products.flat().reduce((a,b)=>a+b,0);
      svg.append(sv("text",{x:122,y:218,"font-size":12},"Multiply matching cells, then sum:"),sv("text",{x:122,y:240,"font-size":12},M.fmt(productSum)+" + bias "+M.fmt(trace.b1[c.filter])+" = "+M.fmt(trace.z1[c.filter][c.row][c.col])));
      fig.append(svg,el("figcaption","Left: spatial patch → shared filter → one feature-map value. The small grid below shows the products before summation. Grid labels use two decimals; calculations use unrounded values. Use the output-cell buttons below to move the highlighted patch.","ax-caption"));return fig;
    }
    function table(spec){
      const wrap=el("div",undefined,"ax-table"),label=el("p",spec.label,"ax-table-label"),table=el("table");
      const rows=spec.rows||spec.data,headers=spec.headers||rows[0].map((_,i)=>"Column "+i);
      const head=el("thead"),tr=el("tr");for(const h of headers)tr.append(el("th",String(h)));head.append(tr);table.append(head);
      const body=el("tbody");for(const row of rows){const tr=el("tr");for(const value of row)tr.append(el("td",typeof value==="number"?M.fmt(value):String(value)));body.append(tr);}table.append(body);wrap.append(label,table);return wrap;
    }
    function map(){
      const wrap=el("div",undefined,"ax-output-map"),c=trace.cfg,grid=el("div",undefined,"ax-cell-grid ax-cell-grid-"+trace.z1[0].length);
      grid.setAttribute("role","group");grid.setAttribute("aria-label","Select a first-convolution output cell");
      trace.z1[c.filter].forEach((row,r)=>row.forEach((value,col)=>{
        const b=button(M.fmt(value),()=>{cfg.row=r;cfg.col=col;selected="conv1";render();},"ax-cell");b.setAttribute("aria-label","Row "+r+", column "+col+", output "+M.fmt(value));b.setAttribute("aria-pressed",String(r===c.row&&col===c.col));grid.append(b);
      }));
      wrap.append(el("p","Select an output cell to inspect its exact input patch and filter products.","ax-caption"),grid);return wrap;
    }
    function explain(){
      const s=lesson.steps.find(s=>s.id===selected);detail.replaceChildren();const h=el("h4",s.title);
      if(s.kind==="note"){
        const body=el("div",undefined,"markdown ax-depth");
        detail.append(h,body);LabLearning.markdown(s.markdown,body,options.source||"");
        if(s.advanced_id&&typeof ConceptFigures!=="undefined"){
          const figure=ConceptFigures.advanced?.(s.advanced_id,sv,el);
          if(figure)detail.append(figure);
        }
        return;
      }
      detail.append(h,el("p",s.why,"ax-purpose"));
      if(options.reading){const terms=el("p",undefined,"ax-definitions");terms.append(el("strong","Terms used here. "),el("span",s.terms));detail.append(terms);}
      else detail.append(el("p",s.terms,"ax-definitions"));
      if(selectedEdge){const e=edges.find(e=>e.id===selectedEdge);if(e)detail.append(el("p","Selected connection: "+e.explanation,"ax-connection-note"));}
      if(trace.reading?.view!=="reading"){
        const io=el("div",undefined,"ax-io");for(const [label,text] of [["What enters",s.input],["What leaves",s.output]]){const p=el("p");p.append(el("strong",label),el("span",text));io.append(p);}detail.append(io);
      }
      if(!options.reading)detail.append(el("h5","What happens"),el("p",s.action));
      if(type==="P8"&&typeof MetricsLab!=="undefined")detail.append(MetricsLab.figure(trace.cfg,selected,sv,el,choose));
      else if(/^N\d{2}$/.test(type)&&typeof NLPFigures!=="undefined"){const figure=NLPFigures.make(type,selected,trace,sv,el,choose);if(figure)detail.append(figure);}
      else if(custom&&typeof ConceptFigures!=="undefined"){
        const relevant=!options.reading||!({P3:["s3"],ML06:["s3"],ML10:["s3"]}[type]||[]).includes(selected);
        const figure=relevant?ConceptFigures.make(type,selected,trace,sv,el,choose):null;if(figure)detail.append(figure);
      }
      if(type==="transformer"&&["qkv","heads","scores","mask","attention-softmax","mix","project"].includes(selected))detail.append(attentionInset());
      if(type==="cnn"&&selected==="conv1")detail.append(convolutionInset());
      if(s.equation){const math=el("div",undefined,"ax-equation");LabLearning.formula(math,s.equation,true);detail.append(math);}
      detail.append(el("h5","Worked example"),el("p",s.calculation));
      if(type==="cnn"&&selected==="conv1")detail.append(map());
      for(const spec of s.values)detail.append(table(spec));
      const end=el("div",undefined,"ax-next-explanation");end.append(el("h5",options.reading?"Connection to the next idea":"Where it goes next"),el("p",s.next),el("h5",options.reading?"Important limitation":"What changes—and what does not"),el("p",s.change));detail.append(end);
      if(type==="transformer"&&selected==="generation"){
        const append=button("Append one example token",()=>{
          const ids=[...trace.ids,trace.nextId];cfg.ids=ids;cfg.position=ids.length-1;selectedEdge="autoregression";render();
        },"secondary");append.disabled=trace.ids.length>=6||trace.ids.at(-1)===5;
        detail.append(append,el("p",trace.ids.length>=6?"Six-token teaching limit reached. Reset the example to start again.":trace.ids.at(-1)===5?"The end token was appended. Reset to start again.":"Only this in-memory example changes. No API, training or saved model is used.","ax-caption"));
      }
    }
    function highlight(){
      const active=nodes.find(n=>n.ids.includes(selected))?.id;
      for(const n of nodes){const yes=n.id===active;n.group.setAttribute("data-selected",String(yes));n.group.setAttribute("aria-pressed",String(yes));}
      for(const e of edges){e.group.setAttribute("data-active",String(e.id===selectedEdge||e.to===active||e.from===active));e.group.setAttribute("aria-pressed",String(e.id===selectedEdge));}
      guide.forEach((b,i)=>{const yes=lesson.steps[i].id===selected;b.setAttribute("aria-current",yes?"step":"false");});
      const index=lesson.steps.findIndex(s=>s.id===selected);back.disabled=index===0;next.disabled=index===lesson.steps.length-1;
      status.textContent="Entry "+(index+1)+" of "+lesson.steps.length;
    }
    function choose(id,edgeId=null){
      if(dead||!lesson.steps.some(s=>s.id===id))return;selected=id;selectedEdge=edgeId;highlight();explain();
      const active=nodes.find(n=>n.ids.includes(selected));if(active&&frame.clientHeight&&frame.clientWidth){const width=frame.children[0]?.getBoundingClientRect?.().width||frame.clientWidth,scale=width/(custom?trace.diagram.width:420),y=active.y*scale;if(y<frame.scrollTop||y>frame.scrollTop+frame.clientHeight-60)frame.scrollTop=Math.max(0,y-frame.clientHeight/2);}
      const index=lesson.steps.findIndex(s=>s.id===selected);if(flow.clientHeight)flow.scrollTop=Math.max(0,(guide[index]?.offsetTop||0)-flow.offsetTop-flow.clientHeight/2);
    }
    function render(){
      if(dead)return;trace=custom?ConceptLessons.build(type,cfg):type==="cnn"?M.cnn(cfg):M.transformer(cfg);
      if(options.reading&&custom)trace=ConceptLessons.read(trace,options.reading);
      cfg={...trace.cfg};lesson=custom?trace:ArchitectureLessons[type](trace);
      if(options.reading&&!custom){lesson=ConceptLessons.read({...lesson,cfg:trace.cfg},options.reading);trace.reading=lesson.reading;}
      const core=[...lesson.steps],words=t=>new Set(t.toLowerCase().match(/[a-z]{4,}/g)||[]);
      for(const [i,n] of (options.notes||[]).entries()){
        const query=words(n.title),scores=core.map(s=>{const tokens=words(s.title+" "+s.terms+" "+s.action);return [...query].filter(w=>tokens.has(w)).length;});
        const best=Math.max(...scores),anchor=options.reading?null:core[best? scores.indexOf(best):core.length-1].id;
        lesson.steps.push({...n,id:"note-"+i,kind:"note",anchor,title:(core.length+i+1)+" · "+n.title});
      }
      if(!lesson.steps.some(s=>s.id===selected))selected=lesson.steps[0].id;
      if(custom)legend.textContent=lesson.legend;
      heading.textContent=lesson.title;intro.textContent=lesson.subtitle;
      overview.replaceChildren();overview.hidden=!options.reading;intro.hidden=!!options.reading;
      const reading=trace.reading;
      if(reading){
        overview.append(el("p",reading.definition,"ax-topic-definition"),el("p",reading.example,"ax-topic-example"),el("p",reading.connection,"ax-topic-connection"));
        root.className="architecture-explorer ax-reviewed"+(reading.view==="reading"?" ax-reading":"");
        flowLabel.textContent="In this lesson";diagramLabel.textContent=reading.diagramLabel||"Worked calculation";
        instructions.textContent=reading.view==="reading"?"Choose a section below. The examples and comparisons are explained separately; they are not stages of one model.":"Select a component, connection or section to inspect the calculation and its explanation.";
        legend.textContent=reading.legend||"Arrows show the stated dependency, not just reading order. Select an arrow to read what connects the calculations. Examples use fixed teaching values, not measured training results.";
      }
      left.hidden=reading?.view==="reading";
      controls();if(!left.hidden)draw();else{nodes=[];edges=[];frame.replaceChildren();}flow.replaceChildren();guide=[];
      for(const s of lesson.steps.filter(s=>s.kind==="note"))nodes.find(n=>n.ids.includes(s.anchor))?.ids.push(s.id);
      for(const [i,s] of lesson.steps.entries()){
        const item=el("li"),b=button(s.title,()=>choose(s.id),"ax-step");b.dataset.axStep=s.id;
        if(s.level==="advanced"&&lesson.steps[i-1]?.level!=="advanced")item.append(el("span","Advanced study","ax-further-label"));
        else if(options.reading&&s.kind==="note"&&lesson.steps[i-1]?.kind!=="note")item.append(el("span","Further explanation","ax-further-label"));
        b.onkeydown=e=>{if(e.key==="ArrowDown"||e.key==="ArrowUp"){e.preventDefault();const j=Math.max(0,Math.min(lesson.steps.length-1,i+(e.key==="ArrowDown"?1:-1)));choose(lesson.steps[j].id);guide[j].focus?.({preventScroll:true});}};item.append(b);flow.append(item);guide.push(b);
      }
      const sourceMap=new Map([...lesson.sources,...options.sources||[]].map(([title,url])=>[url,title]));
      links.replaceChildren(el("span","Sources: "));[...sourceMap].forEach(([url,title],i)=>{if(i)links.append(el("span"," · "));const a=el("a",title);a.href=url;a.target="_blank";a.rel="noopener noreferrer";links.append(a);});
      choose(selected,selectedEdge);
    }
    back.onclick=()=>{const i=lesson.steps.findIndex(s=>s.id===selected);if(i>0)choose(lesson.steps[i-1].id);};
    next.onclick=()=>{const i=lesson.steps.findIndex(s=>s.id===selected);if(i+1<lesson.steps.length)choose(lesson.steps[i+1].id);};
    reset.onclick=()=>{cfg={};selected=type==="cnn"?"image":"tokens";selectedEdge=null;render();};
    render();
    return {choose,root,getTrace:()=>trace,getLesson:()=>lesson,configure(values){cfg={...cfg,...values};selectedEdge=null;render();},destroy(){dead=true;root.remove();}};
  }
  return {mount,placement};
})();
