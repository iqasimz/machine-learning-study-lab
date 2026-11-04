"use strict";

const NLPFigures=(()=>{
  let serial=0;
  function make(id,selected,r,sv,el,choose){
    const supported=["N03","N07","N08","N09","N10","N11","N12","N13","N15","N18","N19","N21","N22","N23"];
    if(!supported.includes(id))return null;
    // Source conditioning is already drawn in the main N11 diagram. Show the
    // independent search fixture only while its own explanation is selected.
    if(id==="N11"&&!["s3","s4"].includes(selected))return null;
    const c=r.cfg,m=r.facts,f=ArchitectureMath.fmt,fig=el("figure",undefined,"ax-nlp-figure"),svg=sv("svg",{viewBox:"0 0 430 390",role:"group","aria-label":r.title+" worked structure"});
    const marker="nlp-arrow-"+(++serial),defs=sv("defs"),arrow=sv("marker",{id:marker,viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:5,markerHeight:5,orient:"auto"});arrow.append(sv("path",{d:"M0 0L10 5L0 10Z",fill:"#758fa7"}));defs.append(arrow);svg.append(defs);
    const txt=(x,y,text,size=11)=>svg.append(sv("text",{x,y,"text-anchor":"middle","font-size":size,fill:"#2d475d"},String(text)));
    function edge(x1,y1,x2,y2,label="",color="#8a9eb1"){svg.append(sv("line",{x1,y1,x2,y2,stroke:color,"stroke-width":1.5,"marker-end":"url(#"+marker+")"}));if(label)txt((x1+x2)/2+15,(y1+y2)/2-6,label,10);}
    function box(x,y,w,h,title,detail,step,color="#edf3f9"){
      const g=sv("g",{role:"button",tabindex:0,"aria-label":title+"; "+detail});
      g.append(sv("rect",{x,y,width:w,height:h,rx:6,fill:selected===step?"#d6e7f7":color,stroke:selected===step?"#24669d":"#acbdcc","stroke-width":selected===step?2.5:1}),sv("text",{x:x+w/2,y:y+h/2-(detail?3:-4),"text-anchor":"middle","font-size":11,fill:"#253f55"},title));
      if(detail)g.append(sv("text",{x:x+w/2,y:y+h/2+13,"text-anchor":"middle","font-size":10,fill:"#456076"},detail));
      g.onclick=()=>choose(step);g.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();choose(step);}};svg.append(g);
    }
    let caption;
    if(id==="N08"||id==="N10"&&selected==="s3"||id==="N09"&&selected==="s4"){
      txt(215,22,"States aligned with the ORIGINAL token order");
      const forward=id==="N08"?"s1":id==="N10"?"s2":"s4",back=id==="N08"?"s2":id==="N10"?"s3":"s4",merge=id==="N08"?"s3":id==="N10"?"s3":"s4";
      const xs=[72,215,358];
      for(let i=0;i<3;i++){if(i<2)edge(xs[i]+39,118,xs[i+1]-39,118);if(m.backward&&i>0)edge(xs[i]-39,227,xs[i-1]+39,227);}
      for(let i=0;i<3;i++){
        box(xs[i]-35,40,70,39,"x="+m.inputs[i],"t="+i,"s0");edge(xs[i],79,xs[i],96);
        box(xs[i]-39,96,78,44,"forward h",f(m.forward[i].h),forward);
        if(m.backward){box(xs[i]-39,205,78,44,"backward h",f(m.backward[i].h),back);edge(xs[i],140,xs[i]-51,156);edge(xs[i]-51,156,xs[i]-51,270);edge(xs[i]-51,270,xs[i],287);edge(xs[i],249,xs[i],287);edge(xs[i]+35,60,xs[i]+51,180);edge(xs[i]+51,180,xs[i]+25,205);}
        else edge(xs[i],140,xs[i],287);
        box(xs[i]-54,287,108,52,"token output",m.output[i].map(f).join(" | "),merge);
      }
      txt(215,368,"Terminal summary: "+m.terminal.map(f).join(" | "),12);
      caption="Arrows along each row carry that direction's earlier state. Vertical connections align states to the same input token. The terminal backward summary is at original position 0, not at the last token. No weights are trained here.";
    }else if(id==="N10"){
      const a=m.forward[c.position];txt(215,20,"GRU at t="+c.position+": reset AFTER recurrent projection");
      edge(90,84,90,120);edge(340,84,340,120);edge(90,167,90,202);edge(90,249,215,278);edge(340,167,340,235);edge(340,235,250,278);edge(215,316,215,342);
      edge(20,60,6,60);edge(6,60,6,362);edge(6,362,100,362,"z × old h");
      box(20,40,140,44,"Old hidden h",f(a.hp),"s0");box(275,40,130,44,"Input x",a.x,"s0");
      box(20,120,140,47,"Recurrent projection","0.8 × old h","s1");box(275,120,130,47,"Input projection","0.5 × x","s1");
      box(20,202,140,47,"Reset contribution","r="+c.reset,"s1");box(145,273,140,43,"Add, then tanh",f(a.candidate),"s1");
      box(100,342,245,43,"Mix: .75 × n + .25 × old h",f(a.h),"s2");
      caption="The reset gate scales the recurrent projection before candidate formation. The old state also has its own update-gated route into the output. There is no separate LSTM cell c. Select the final guide entry to inspect aligned directional states.";
    }else if(id==="N09"){
      const a=m.forward[c.position];txt(215,20,"LSTM at t="+c.position+": x="+a.x+", previous h="+f(a.hp));
      edge(85,89,85,125);edge(315,89,315,125);edge(85,180,185,223);edge(315,180,245,223);edge(215,268,215,307);
      box(15,40,140,49,"Previous cell",f(a.cp),"s0");box(245,40,170,49,"Candidate g",f(a.candidate),"s1");
      box(15,125,140,55,"Retain: f × old c",f(a.f)+" × "+f(a.cp),"s2","#f4e8d3");
      box(245,125,170,55,"Write: i × candidate","0.5 × "+f(a.candidate),"s2","#dcebe5");
      box(145,218,140,50,"Add → new c",f(a.c),"s2");
      box(115,307,200,57,"Expose: o × tanh(c)","0.75 × tanh(c) = "+f(a.h),"s3");
      caption="The two branches are retained old memory and new candidate information. They add into c; the output gate exposes h without replacing c. Both states continue to the next position. Select an operation for the exact calculation.";
    }else if(id==="N03"){
      txt(215,26,"Smoothed, row-normalized TF-IDF");m.vocabulary.forEach((t,i)=>txt(125+i*75,65,t));
      m.vectors.forEach((row,i)=>{txt(38,113+i*69,"doc "+(i+1));row.forEach((value,j)=>box(90+j*75,86+i*69,68,51,f(value),"",j===m.index?"s2":"s0",value?"#d8e7f4":"#f4f6f8"));});
      txt(215,330,"Feature columns retain the fitted vocabulary order",11);
      caption="Rows are documents, columns are terms. Weighting and normalization change values; they do not create contextual token interactions. Select a cell to inspect the weighting stage.";
    }else if(id==="N07"){
      txt(215,25,"One shared temporal filter, two windows");[1,-1,2].forEach((x,i)=>box(50+i*130,52,70,40,String(x),"token "+i,"s0"));
      edge(85,92,140,153);edge(215,92,140,153);edge(215,92,300,153);edge(345,92,300,153);
      box(65,153,150,55,"× [.5,−.5]",f(m.convolution[0]),"s1");box(245,153,150,55,"× [.5,−.5]",f(m.convolution[1]),"s1");
      edge(140,208,140,252);edge(320,208,320,252);box(80,252,110,44,"ReLU",f(m.activated[0]),"s2");box(265,252,110,44,"ReLU",f(m.activated[1]),"s2");
      edge(135,296,215,332);edge(320,296,215,332);box(135,332,160,43,c.pool+" pool",f(m.pooled),"s2");
      caption="Every arrow into a filter identifies the local input window. Both filters drawn here use the SAME parameters. The pooled scalar loses the location information present in the two-position feature map.";
    }else if(id==="N12"){
      txt(215,25,"Viterbi trellis: two possible tags at each position");const xs=[65,215,365],ys=[105,225];
      for(let t=0;t<2;t++)for(let i=0;i<2;i++)for(let j=0;j<2;j++)edge(xs[t]+22,ys[i],xs[t+1]-25,ys[j],"",m.best[t]===i&&m.best[t+1]===j?"#3b846d":"#cad5de");
      for(let t=0;t<3;t++)for(let j=0;j<2;j++)box(xs[t]-43,ys[j]-29,86,58,["A","B"][j]+" · t="+t,"δ="+f(m.scores[t][j]),m.best[t]===j?"s2":"s1",m.best[t]===j?"#d9eee4":"#f1f4f7");
      txt(215,312,"Green path: "+m.best.map(x=>["A","B"][x]).join(" → "),13);txt(215,346,"Best score "+f(m.bestScore)+"; logZ "+f(m.logZ),12);
      caption="Green edges/fills identify the winning path; blue outlines identify the selected explanation. Each node stores the best prefix ending at this tag. Training/normalization sums paths instead of replacing their sum by the Viterbi maximum.";
    }else if(id==="N15"){
      txt(215,22,c.mode==="bi"?"Independent encoding, then vector comparison":c.mode==="cross"?"Joint input, joint token interactions":"Separate token encodings, then late interaction");
      box(15,55,170,50,"Query",c.mode==="cross"?"two input tokens":"two valid token vectors","s1");box(245,55,170,50,"Document","candidate input","s2");
      if(c.mode==="cross"){
        edge(100,105,215,153);edge(330,105,215,153);box(85,153,260,64,"One joint encoder","[query | separator | document]","s3");edge(215,217,215,283);box(105,283,220,58,"Pair relevance head","no reusable final document vector","s3");
      }else{
        edge(100,105,100,163);edge(330,105,330,163);
        box(15,163,170,59,c.mode==="bi"?"Pool + normalize":"Keep token vectors",c.mode==="bi"?m.q.map(f).join(" | "):"q₁=[1,0], q₂=[0,1]","s1");
        box(245,163,170,59,c.mode==="bi"?"Cache document vector":"Cache document tokens",c.mode==="bi"?"d₁=[.8,.6], d₂=[0,1]":"multiple vectors / document","s2");
        edge(100,222,215,283);edge(330,222,215,283);box(80,283,270,60,c.mode==="bi"?"Compare vectors":"Max token match, then sum",c.mode==="bi"?"cosine scores: "+m.scores.map(f).join(", "):"scores: 2 versus 1.4","s3");
      }
      caption="The point where the two texts meet determines interaction capacity, caching and cost. This architecture illustration does not provide a measured ranking of trained bi-, cross- or late-interaction models.";
    }else if(id==="N13"){
      txt(215,23,"A dependency tree is not a POS-tag sequence");box(165,48,100,42,"ROOT","","s1");edge(215,90,215,160,"root");box(160,160,110,50,"chase","token 1","s1");edge(170,210,85,283,"subject");edge(260,210,345,283,"object");box(25,283,120,50,"cats","token 0","s0");box(285,283,120,50,"mice","token 2","s0");
      caption="A single root and directed head links form the authored tree. Head/relation choices are structural predictions; they do not themselves establish semantic or argumentative support.";
    }else if(id==="N19"){
      txt(215,30,"Context coordinates: inclusive token indices here");m.words.forEach((w,i)=>box(10+i*84,85,78,50,w,String(i),"s0"));
      txt(63,188,"Gold");txt(63,266,"Predicted");
      for(let i=0;i<5;i++){if(i>=m.gold[0]&&i<=m.gold[1])box(10+i*84,200,78,34,"match","","s2","#d9ebe1");if(i>=m.prediction[0]&&i<=m.prediction[1])box(10+i*84,278,78,34,"selected","","s1","#dce9f6");}
      txt(215,354,"Answer-token F1 "+f(m.f1)+"; exact match "+m.em,13);
      caption="Span overlap gives partial answer credit under this tokenizer. Real extractive readers must reject reversed, overlong, padded and question-token spans before scoring. This is not exact entity evaluation.";
    }else if(id==="N23"){
      txt(215,22,"Allowed attention pairs: query rows × key columns");const mask=c.mask==="global"?m.global:m.local;
      mask.forEach((row,i)=>row.forEach((yes,j)=>svg.append(sv("rect",{x:107+j*27,y:48+i*27,width:23,height:23,fill:yes?"#7ea8cb":"#eef2f6",stroke:"#d3dde5"}))));
      [0,1,2,3,4,5,6,7].forEach(i=>{txt(118+i*27,40,i,9);txt(94,64+i*27,i,9);});
      txt(215,293,"Separate alternative: windows "+m.windows.map(x=>x.join("–")).join(", "),11);
      txt(215,324,m.exposures+" processed positions; 8 unique tokens",12);txt(215,354,"Span 3–4 jointly covered: "+(m.coversBoundary?"yes":"no"),12);
      caption="The mask describes one sparse-attention alternative. The window counts describe a separate chunking alternative. They are not silently combined into one implemented encoder. Overlap improves this boundary's coverage but repeats work.";
    }else if(id==="N21"){
      txt(215,25,"Same authored units, different declared target policy");const roles=["claim","premise","other"],texts=["Fund transit","It reduces traffic","Hello"];
      texts.forEach((text,i)=>{box(10,65+i*88,180,55,text,roles[i],"s0");edge(190,92+i*88,285,92+i*88);box(290,65+i*88,125,55,"target label",String(m.labels[i]),"s1");});
      caption="These role annotations and policies are authored illustrations. The mapping is not a universal definition of argumentative language or a claim that every dataset follows this scheme.";
    }else if(id==="N22"){
      txt(215,25,"Fine-tuning source rows; evaluation target columns");txt(185,73,"Target A");txt(325,73,"Target B");m.matrix.forEach((row,i)=>{txt(62,154+i*100,"Source "+["A","B"][i]);row.forEach((x,j)=>box(130+j*140,115+i*100,110,70,f(x),i===j?"within source":"transfer",i===c.source?"s1":"s0",i===c.source?"#d8e8f6":"#f0f4f7"));});
      caption="All scores are authored. Per-target results preserve information that a grand average hides. A transfer gap does not identify language, topic or label-policy shift as its cause.";
    }else if(id==="N18"){
      txt(215,28,"Clip matches BEFORE the brevity penalty");m.reference.forEach((w,i)=>box(13+i*105,85,95,46,w,"reference","s0"));m.words.forEach((w,i)=>box(13+i*105,195,95,46,w,"candidate","s1"));txt(215,300,"Clipped matches "+m.hits+"; precision "+f(m.precision),12);txt(215,339,"BP "+f(m.bp)+" → BLEU-1 "+f(m.bleu1),13);
      caption="This is a one-reference, unigram teaching score. Repeating 'sat' cannot create a second reference occurrence. Standard corpus BLEU needs its declared n-gram, aggregation, smoothing and tokenization conventions.";
    }else if(id==="N11"){
      txt(215,25,"Fixed length-two search tree (no end token)");box(165,45,100,40,"Start","","s3");edge(190,85,100,140);edge(240,85,330,140);box(35,140,130,48,"a · P=.6","kept by both beams","s3");box(265,140,130,48,"b · P=.4",c.beam===2?"kept by beam 2":"pruned by beam 1","s3");
      edge(90,188,60,275);edge(120,188,165,275);edge(310,188,270,275);edge(345,188,375,275);
      m.paths.forEach((p,i)=>box(10+i*105,275,95,55,p[0],"P="+f(p[1]),"s4",i===3&&c.beam===2?"#d8eadf":"#edf3f9"));
      caption="Greedy chooses a at the first step and x under the declared tie rule, scoring 0.3. Beam 2 keeps b, allowing b y at 0.38. A wider real search still does not guarantee the most useful or truthful output.";
    }
    fig.append(svg,el("figcaption",caption,"ax-caption"));return fig;
  }
  return {make};
})();
