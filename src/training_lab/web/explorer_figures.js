"use strict";
const ConceptFigures=(()=>{
  function make(id,selected,r,sv,el,choose){
    const c=r.cfg,fig=el("figure",undefined,"ax-concept-figure"),svg=sv("svg",{viewBox:"0 0 430 270",class:"ax-inset-svg",role:"group","aria-label":"Worked structure for "+r.title});
    let caption="",used=true;
    const text=(x,y,s,size=12)=>svg.append(sv("text",{x,y,"font-size":size,fill:"#30485e","text-anchor":"middle"},s));
    const line=(x1,y1,x2,y2,color="#a7b7c8",dash=false)=>svg.append(sv("line",{x1,y1,x2,y2,stroke:color,"stroke-width":1.5,...(dash?{"stroke-dasharray":"4 3"}:{})}));
    const rect=(x,y,w,h,color="#e7eff8")=>svg.append(sv("rect",{x,y,width:w,height:h,rx:4,fill:color,stroke:"#a7b7c8"}));
    const circle=(x,y,label,step,color="#e7eff8")=>{
      const g=sv("g",{class:"ax-figure-node",...(step?{role:"button",tabindex:0,"aria-label":label+"; select related explanation"}:{})});
      g.append(sv("circle",{cx:x,cy:y,r:22,fill:selected===step?"#c8dff5":color,stroke:"#6388ab","stroke-width":selected===step?2.5:1.2}),sv("text",{x,y:y+4,"font-size":11,"text-anchor":"middle",fill:"#243b53"},label));
      if(step){g.onclick=()=>choose(step);g.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();choose(step);}};}svg.append(g);
    };
    const bars=(labels,values,zero=0)=>{
      const max=Math.max(1,...values.map(Math.abs)),base=zero?145:220;
      line(35,base,405,base);
      values.forEach((value,i)=>{const x=48+i*330/values.length,w=Math.min(65,260/values.length),h=Math.abs(value)/max*(zero?70:150);
        rect(x,zero&&value<0?base:base-h,w,Math.max(1,h),value<0?"#eeddd4":"#c8dff5");
        text(x+w/2,zero&&value<0?base+h+16:base-h-10,ArchitectureMath.fmt(value),11);text(x+w/2,253,labels[i],11);
      });
    };
    if(id==="DL01"){
      const h=[1,.5].map(x=>c.activation==="relu"?x:Math.tanh(x)),out=h[0]-h[1];
      for(const a of [[65,85],[65,185]])for(const b of [[215,85],[215,185]])line(...a,...b);
      line(237,85,343,135);line(237,185,343,135);
      text(65,34,"Input");text(215,34,"Hidden features");text(365,34,"Output");
      circle(65,85,"1","s0");circle(65,185,"2","s0");
      circle(215,85,ArchitectureMath.fmt(h[0]),"s2");circle(215,185,ArchitectureMath.fmt(h[1]),"s2");
      circle(365,135,ArchitectureMath.fmt(out),"s3");
      text(140,68,"0.5",10);text(105,126,"−0.5",10);text(178,102,"0.25",10);text(140,205,"0.5",10);
      text(295,88,"×1",11);text(295,190,"×(−1)",11);
      caption="Each hidden feature reads both inputs; output weights combine the activated features. Select a node to inspect its stage.";
    }else if(id==="DL07"){
      text(215,28,"Same cell parameters at every time step");
      const inputs=[1,0,1],hs=[];let h=0,cell=0;
      for(const x of inputs){if(c.variant==="rnn")h=Math.tanh(.5*x+.8*h);else if(c.variant==="gru")h=.75*Math.tanh(.5*x+.4*h)+.25*h;else{cell=.8*cell+.5*Math.tanh(.5*x);h=.7*Math.tanh(cell);}hs.push(h);}
      [75,215,355].forEach((x,i)=>{if(i)line(x-118,145,x-22,145);line(x,98,x,122);circle(x,76,"x="+inputs[i],"s"+(i+1));circle(x,145,ArchitectureMath.fmt(hs[i]),"s"+(i+1));text(x,205,"h"+(i+1),12);});
      text(215,245,c.variant==="lstm"?"LSTM also carries a separate cell state c.":"The carried hidden state links the steps.",12);
      caption="Unrolling draws repeated uses of one cell. It does not create three independently learned parameter sets.";
    }else if(id==="DL11"){
      line(70,80,215,80);line(215,80,360,80);
      circle(70,80,"A: 1","s0");circle(215,80,"B: 2","s0");circle(360,80,"C: 4","s0");
      line(70,102,180,165,"#6388ab");line(360,102,250,165,"#6388ab");
      rect(150,150,130,45);text(215,178,c.aggregation+"(1,4)");line(215,195,215,224);
      text(215,245,"B' = 0.5 × (2 + "+(c.aggregation==="mean"?"2.5":"5")+")");
      caption="B receives A/C messages from the previous layer. The same synchronous rule updates other nodes; the graph edges are not a text sequence.";
    }else if(id==="ML12"){
      const px=x=>100+x*65,py=y=>190-y*55;
      line(55,190,360,190);line(100,235,100,35);line(px(-.5),py(-.5),px(2.8),py(2.8),"#477b70");
      line(px(2),py(0),px(1),py(1),"#a57833",true);
      circle(px(2),py(0),"x","s1");circle(px(1),py(1),"x̂","s2");
      text(280,45,"Retained direction [1,1]/√2",11);text(245,243,"Original [2,0] → reconstructed [1,1]",11);
      caption="The amber dashed residual is orthogonal to this PCA direction. Its squared length is 2; the threshold is a separate choice.";
    }else if(["ML06","ML11"].includes(id)){
      const xs=id==="ML06"?[0,1,2,4,5]:[0,2,8,10],max=id==="ML06"?5:10,x=value=>40+350*value/max;
      line(35,135,400,135);xs.forEach(value=>{circle(x(value),135,String(value),id==="ML06"?"s1":"s0");});
      if(id==="ML06"){line(x(3),70,x(3),111,"#a57833");text(x(3),52,"Query 3");caption="Distance is horizontal separation on this number line. The chosen k controls how many training labels are aggregated.";}
      else{[1,9].forEach(value=>{line(x(value),190,x(value),158,"#a57833");text(x(value),213,"new μ="+value,11);});text(215,38,"Initial centroids 0 and 10");text(x(c.query),85,"query "+c.query,11);line(x(c.query),93,x(c.query),111,"#a57833",true);caption="The training assignments update centers to 1 and 9. The inspected query does not enter this fitting step.";}
    }else if(id==="P2"){
      const w=c.weight,score=id==="P2"?2*w-1:2*w;
      line(40,220,400,220);line(40,220,40,25);
      const x=t=>40+t*110,y=t=>220-t*35;
      line(x(0),y(id==="P2"?-1:0),x(3),y(3*w-(id==="P2"?1:0)),"#6388ab");
      line(x(2),y(score),x(2),y(4),"#a57833",true);
      circle(x(2),y(score),score===4?"ŷ=y":"ŷ","s1");if(score!==4)circle(x(2),y(4),"y=4","s2");
      text(215,250,"Input x₁=2; amber line is the prediction error",11);
      caption=id==="P2"?"Feature 2 stays 1 and contributes −1. Only the first coefficient changes.":"This line shows a one-weight predictor, not an already fitted regression model.";
    }else if(["P3","ML04","R3","P7","R4"].includes(id)){
      let labels,values;
      if(id==="P3"){labels=["class 1","class 0"];values=[c.p,1-c.p];}
      else if(id==="ML04"){const p=1/(1+Math.exp(-.4));labels=["positive","negative"];values=[p,1-p];}
      else if(id==="R3"){labels=["token A","token B","token C"];values=ArchitectureMath.softmax([2,1,0].map(x=>x/c.temperature));}
      else{labels=["doc A","doc B"];values=id==="P7"?[.8,0]:[.8,.3];}
      bars(labels,values);
      if(id==="ML04"){const y=220-c.threshold*150;line(35,y,405,y,"#a57833",true);text(350,y-8,"cutoff "+c.threshold,11);}
      caption=id==="P7"||id==="R4"?"These bars are similarity scores, not probabilities or truth estimates. The main workflow also applies permissions.":"Probability bars share one normalization axis. Changing a decision threshold does not change these probabilities.";
    }else if(id==="ML03"){
      const w=11/(14+c.lambda),x=t=>45+t*110,y=t=>220-t*50;
      line(45,220,400,220);line(45,220,45,30);
      line(x(0),y(0),x(3),y(3*w),"#6388ab");
      [1,2,2].forEach((target,i)=>{line(x(i+1),y(target),x(i+1),y(w*(i+1)),"#a57833",true);svg.append(sv("circle",{cx:x(i+1),cy:y(target),r:5,fill:"#b08035"}),sv("circle",{cx:x(i+1),cy:y(w*(i+1)),r:4,fill:"#477ca6"}));text(x(i+1)-35,y(target)-22,"y="+target,11);});
      text(215,252,"Fitted slope w = "+ArchitectureMath.fmt(w)+" at λ = "+c.lambda,12);
      caption="Amber points are targets; blue points lie on the fitted line. Dashed residuals show errors. Increasing the penalty shrinks this slope toward zero.";
    }else if(id==="ML10"){
      const p=[.9,.6,.4,.1].map(x=>1/(1+Math.exp(-c.strength*Math.log(x/(1-x))))),ys=[1,0,1,0];
      const x=v=>60+310*v,y=v=>220-150*v;
      line(60,220,370,220);line(60,220,60,70);line(60,220,370,70,"#a7b7c8",true);
      [[0,.8],[.8,1.0001]].forEach(([lo,hi])=>{
        const rows=p.map((q,i)=>({confidence:Math.max(q,1-q),correct:Number(Number(q>=.5)===ys[i])})).filter(r=>r.confidence>=lo&&r.confidence<hi);
        if(!rows.length)return;const conf=rows.reduce((a,r)=>a+r.confidence,0)/rows.length,acc=rows.reduce((a,r)=>a+r.correct,0)/rows.length;
        line(x(conf),y(conf),x(conf),y(acc),"#a57833",true);
        circle(x(conf),y(acc),String(rows.length),"s2");
      });
      text(215,263,"Mean confidence →",12);text(215,25,"Vertical position: observed fraction correct",12);
      caption="The diagonal means confidence matches accuracy. Each circle is one occupied bin; its number is the case count. Four cases illustrate the calculation, not reliable calibration evidence.";
    }else if(["P4","T6","T5","T7"].includes(id)){
      let labels,values;
      if(id==="P4"){labels=["train","development","test"];values=[c.train,90-c.train,10];}
      else if(id==="T6"){labels=["weights MB","gradients MB","Adam MB"];values=[4*c.million,4*c.million,8*c.million];}
      else{labels=["A valid tokens","B valid tokens"];values=[2,c.second];}
      bars(labels,values);
      caption=id==="P4"?"The held-out partitions do not flow back into fitting. These counts are a teaching split, not a prescribed universal ratio.":id==="T6"?"Static float32 state estimate only; activations, workspaces and allocator overhead are not included.":"Unequal valid counts require weighted means. Tensor padding does not contribute valid targets.";
    }else if(id==="DL09"){
      text(70,28,"Input");text(215,28,"Latent");text(360,28,"Reconstruction");
      const latent=c.width===1?[.5]:[1,0],rec=c.width===1?[.5,.5]:[1,0];
      [1,0].forEach((value,i)=>circle(70,90+i*90,String(value),"s0"));
      latent.forEach((value,j)=>{const y=latent.length===1?135:90+j*90;[90,180].forEach(iy=>line(92,iy,193,y));circle(215,y,String(value),"s0");rec.forEach((_,k)=>line(237,y,338,90+k*90));});
      rec.forEach((value,i)=>circle(360,90+i*90,String(value),"s1"));
      caption="The bottleneck and decoder are explicit fixed maps. A wider identity example reconstructs perfectly without proving useful representation learning.";
    }else if(id==="R7"){
      bars(["−1.2 weight","0.4 weight","0.8 weight"],[-1.2,.4,.8],1);
      const q=[-1.2,.4,.8].map(x=>Math.max(-8,Math.min(7,Math.round(x/c.scale))));
      text(215,24,"Integer codes ["+q.join(", ")+"] at scale "+c.scale,12);
      caption="Signs matter. The exact code/reconstruction/error is explained alongside the diagram; this browser figure does not pack values into four-bit storage.";
    }else used=false;
    if(!used)return null;
    svg.append(sv("title",{},r.title+" — worked structure"));
    fig.append(svg,el("figcaption",caption,"ax-caption"));return fig;
  }
  function advanced(key,sv,el){
    const plans={
      "vision-transformer":{
        title:"Image patches through a Vision Transformer",height:385,
        nodes:[["image",12,25,152,["224 × 224 RGB image","one image, three channels"]],["patch",190,25,152,["196 flattened patches","16 × 16 × 3 = 768 values"]],["project",368,25,152,["Learned projection","768 features → 192"]],["position",368,145,152,["Positions + class token","197 × 192 features"]],["blocks",190,145,152,["Transformer blocks","mix permitted patch features"]],["head",12,145,152,["Task head","class scores or other targets"]]],
        links:[["image","patch"],["patch","project"],["project","position"],["position","blocks"],["blocks","head"]],
        labels:[[260,265,"Patch count and hidden width are different axes."],[260,300,"Global attention: 197 × 197 scores per head."],[260,335,"The class token is learned input, not a supplied label."]],
        caption:"Read left to right across the top, then back across the second row. Patch embedding changes feature width; attention mixes positions. This is a shape diagram, not a trained vision model."
      },
      "diffusion-training":{
        title:"Diffusion training and generation use different paths",height:400,
        nodes:[["data",12,40,152,["Training data x₀","sample timestep t + noise ε"]],["noisy",190,40,152,["Construct noisy xₜ","closed-form corruption"]],["loss",368,40,152,["Predict ε; score error","backpropagate; update weights"]],["start",12,220,152,["Generation starts at noise","load trained, fixed weights"]],["repeat",190,220,152,["Reverse sampler","repeated denoiser calls"]],["output",368,220,152,["Generated sample","stop at chosen endpoint"]]],
        links:[["data","noisy"],["noisy","loss"],["start","repeat"],["repeat","output"]],
        labels:[[260,20,"TRAINING: learn the denoiser"],[260,195,"SAMPLING: use the learned denoiser"],[260,340,"A training update need not run the full noising trajectory."]],
        caption:"The upper path changes weights; the lower path generates with fixed weights. Original training noise is not available to subtract during generation. Sampler steps are not optimizer updates."
      },
      "graph-attention":{
        title:"Graph attention weights permitted neighbors",height:365,
        nodes:[["a",12,35,152,["Neighbor A value = 2","supplied score = log 3"]],["b",12,150,152,["Neighbor B value = 6","supplied score = 0"]],["weights",205,90,150,["Neighbor softmax","weights 0.75 and 0.25"]],["mix",368,90,152,["Weighted mixture","0.75 × 2 + 0.25 × 6 = 3"]]],
        links:[["a","weights"],["b","weights"],["weights","mix"],["a","mix"],["b","mix"]],
        labels:[[260,275,"Only the declared neighborhood can contribute."],[260,310,"Learned attention weight ≠ proof of causal importance."]],
        caption:"Scores determine weights; the outer paths separately carry projected neighbor values to the mixture. The scores are supplied for this calculation, not learned by this picture. Self-loops and multiple heads require explicit conventions."
      },
      "actor-critic":{
        title:"Actor and critic play different roles",height:395,
        nodes:[["state",12,35,152,["State s","permitted observation"]],["actor",190,35,152,["Actor policy π(a | s)","sample an action"]],["env",368,35,152,["Environment transition","observe reward r + state s′"]],["critic",190,190,152,["Critic value V(s)","estimate return baseline"]],["signal",368,190,152,["TD error / advantage","r + γV(s′) − V(s)"]],["updates",12,190,152,["Separate updates","actor: policy; critic: value"]]],
        links:[["state","actor"],["state","critic"],["actor","env"],["env","signal"],["critic","signal"],["signal","updates"]],
        labels:[[260,310,"The critic is a learned estimate, not a ground-truth reward."],[260,345,"Updates use the specified actor and value objectives."]],
        caption:"The top path is interaction; the lower path calculates a learning signal. The state also supplies the critic input. The simplified picture omits rollout storage and optimizer internals; terminal handling is explained in the text."
      }
    };
    const plan=plans[key];if(!plan)return null;
    const fig=el("figure",undefined,"ax-concept-figure"),svg=sv("svg",{viewBox:"0 0 532 "+plan.height,role:"img","aria-label":plan.title});
    svg.append(sv("title",{},plan.title));
    const nodes=new Map(plan.nodes.map(n=>[n[0],n]));
    for(const [from,to]of plan.links){
      if(key==="graph-attention"&&to==="mix"&&from!=="weights"){
        const top=from==="a",start=top?35:208,rail=top?12:230,end=top?90:148;
        svg.append(sv("path",{d:"M88 "+start+" L88 "+rail+" L444 "+rail+" L444 "+end,fill:"none",stroke:"#a57833","stroke-width":1.5}));
        svg.append(sv("path",{d:top?"M444 90 L441 83 L447 83 Z":"M444 148 L441 155 L447 155 Z",fill:"#a57833"}));
        continue;
      }
      if(key==="actor-critic"&&from==="state"&&to==="critic"){
        svg.append(sv("path",{d:"M88 93 L88 140 L265 140 L265 190",fill:"none",stroke:"#8194a7","stroke-width":1.5}));
        svg.append(sv("path",{d:"M265 190 L262 183 L268 183 Z",fill:"#8194a7"}));
        continue;
      }
      if(key==="actor-critic"&&from==="signal"&&to==="updates"){
        svg.append(sv("path",{d:"M444 248 L444 278 L88 278 L88 248",fill:"none",stroke:"#8194a7","stroke-width":1.5}));
        svg.append(sv("path",{d:"M88 248 L85 255 L91 255 Z",fill:"#8194a7"}));
        continue;
      }
      const a=nodes.get(from),b=nodes.get(to),ac=[a[1]+a[3]/2,a[2]+29],bc=[b[1]+b[3]/2,b[2]+29];
      const horizontal=Math.abs(ac[0]-bc[0])>Math.abs(ac[1]-bc[1]);
      const direction=horizontal?Math.sign(bc[0]-ac[0]):Math.sign(bc[1]-ac[1]);
      const x1=horizontal?ac[0]+direction*a[3]/2:ac[0],y1=horizontal?ac[1]:ac[1]+direction*29;
      const x2=horizontal?bc[0]-direction*b[3]/2:bc[0],y2=horizontal?bc[1]:bc[1]-direction*29;
      svg.append(sv("path",{d:"M"+x1+" "+y1+" L"+x2+" "+y2,fill:"none",stroke:"#8194a7","stroke-width":1.5}));
      const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1,ux=dx/len,uy=dy/len;
      svg.append(sv("path",{d:"M"+x2+" "+y2+" L"+(x2-7*ux+3*uy)+" "+(y2-7*uy-3*ux)+" L"+(x2-7*ux-3*uy)+" "+(y2-7*uy+3*ux)+" Z",fill:"#8194a7"}));
    }
    for(const [,x,y,w,lines]of plan.nodes){
      svg.append(sv("rect",{x,y,width:w,height:58,rx:7,fill:"#f0f5fa",stroke:"#a8bbce"}));
      lines.forEach((line,i)=>svg.append(sv("text",{x:x+w/2,y:y+23+i*18,"text-anchor":"middle","font-size":i?9:10,fill:"#30485e"},line)));
    }
    for(const [x,y,text]of plan.labels)svg.append(sv("text",{x,y,"text-anchor":"middle","font-size":11,fill:"#30485e"},text));
    fig.append(svg,el("figcaption",plan.caption,"ax-caption"));return fig;
  }
  return {make,advanced};
})();
