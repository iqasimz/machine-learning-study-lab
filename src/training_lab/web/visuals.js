/* Static, fully drawn teaching diagrams. No animation, illustrative controls or hidden stages. */
const LabVisuals = (() => {
  "use strict";
  const NS="http://www.w3.org/2000/svg";
  const C={ink:"#26364c",blue:"#3664a3",pale:"#e8eef5",teal:"#28796e",green:"#e1eee7",warm:"#9c6b32",sand:"#f4eadd",line:"#b8c5cf",muted:"#627383"};
  const catalog=[
    ["architecture","A decoder block and its tensor shapes"],
    ["architectures","How the main model families differ"],
    ["attention","One complete attention calculation"],
    ["masking","Attention masks and loss masks are different"],
    ["data","Inputs, next-token targets and response masking"],
    ["training-loop","The complete training loop"],
    ["batching","Token-weighted gradient accumulation"],
    ["normalization","Residual paths and normalization"],
    ["lora","Frozen weights and low-rank adaptation"],
    ["preferences","DPO, PPO and GRPO training signals"],
    ["inference","Prefill, decoding and the KV cache"],
    ["distributed","Data parallelism and sharded state"],
  ].map(([id,title])=>({id,title}));
  const lessonFigures={P5:{4:"normalization"},T0:{1:"training-loop"},T1:{1:"data"},T2:{5:"masking"},
    T3:{1:"architecture",3:"attention",5:"normalization"},T4:{1:"architectures"},T5:{2:"batching"},
    T7:{2:"distributed"},R1:{3:"lora"},R2:{5:"preferences"},R6:{3:"inference"}};
  const diagramTerms={
  "architecture": "B counts examples, T token positions and d features per token. A residual adds matching-shape vectors; normalization rescales features. Q/K/V are attention projections; MLP or FFN is the position-wise feed-forward network.",
  "architectures": "A decoder generates tokens; an encoder represents an input. Causal attention sees no future output; bidirectional attention uses both input directions. Cross-attention connects two sequences; an expert is a selected feed-forward sub-network.",
  "attention": "Q (query) and K (key) produce matching scores; V (value) supplies vectors to mix. A causal mask blocks future positions. Softmax converts each allowed score row into weights summing to 1; d_head is head width.",
  "masking": "A query is the position reading information; a key is a position it can read. The attention mask controls visibility. The loss mask m selects targets to score; p is their predicted probability.",
  "data": "A token is a tokenizer-defined text unit. The target is the next token to predict. A causal attention mask controls visibility; a loss mask m selects scored targets. The denominator counts those selected targets.",
  "training-loop": "A batch supplies inputs and known targets. Loss measures prediction error, gradients measure its sensitivity to weights θ, and the optimizer changes weights. Validation checks separate text without an update.",
  "batching": "A microbatch fits one forward/backward calculation. n_b counts its valid targets, and mean loss L_b averages their penalties. Accumulation combines gradients from several microbatches before one optimizer update.",
  "normalization": "x is one token's feature vector; F is a learned sublayer; d counts features. A residual adds x back. LayerNorm uses mean and variance; RMSNorm uses root-mean-square magnitude. ε prevents division by zero.",
  "lora": "W is the original frozen projection. A and B are trainable matrices with small inner width r, called rank. α/r scales their contribution. d_in and d_out are the original input/output widths.",
  "preferences": "A policy is a model's action distribution. A reference is a fixed comparison model; a rollout is sampled behaviour. Reward scores a response; advantage compares reward/return with a baseline. DPO uses preference pairs; PPO/GRPO use sampled reward signals.",
  "inference": "Prefill processes prompt tokens; decode produces later tokens. The KV cache stores keys and values for earlier positions. B is batch size, T cached length, L layers, h_KV KV heads, d_h head width and b bytes per value.",
  "distributed": "A worker is a training process. Data parallelism gives workers different examples; sharding partitions model state. Gradient reduction combines local derivatives into the intended global update.",
  "trace": "An embedding is a token vector; attention weights mix permitted token positions. Logits are raw next-token scores and probabilities normalize them. These bounded tables show measured checkpoint values, not an explanation of factual correctness."
};
  const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const svgNode=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;};
  const fixed=(v,d=3)=>Number.isFinite(Number(v))?Number(v).toFixed(d):"—";
  function math(host,tex,display=true){host.replaceChildren();if(globalThis.katex){try{katex.render(tex,host,{displayMode:display,output:"mathml",trust:false,throwOnError:true,strict:"warn"});return;}catch{}}host.append(node("code",tex));}
  function mount(host,id,options={}){
    if(id==="targets")id="data";
    if(id!=="trace"&&!catalog.some(c=>c.id===id))throw Error("Unknown static diagram: "+id);
    let dead=false,settings=options.settings||{};
    const root=node("figure",undefined,"visual-lab"),title=node("div",undefined,"visual-title"),body=node("div",undefined,"visual-display");
    root.setAttribute("aria-label",id==="trace"?"Actual checkpoint calculation":catalog.find(c=>c.id===id).title);
    title.append(node("h"+(options.headingLevel||3),id==="trace"?"Actual checkpoint values":catalog.find(c=>c.id===id).title),node("span",id==="trace"?"Measured values":"Static diagram","visual-tag"));root.append(title,body);host.replaceChildren(root);
    let picture;
    function canvas(height,label){picture=svgNode("svg",{viewBox:`0 0 720 ${height}`,role:"img",class:"visual-svg","aria-label":label});picture.append(svgNode("title",{},label));body.append(picture);}
    function text(x,y,value,attrs={}){picture.append(svgNode("text",{x,y,fill:C.ink,"font-size":13,...attrs},value));}
    function box(x,y,w,h,labels,fill=C.pale){
      picture.append(svgNode("rect",{x,y,width:w,height:h,rx:5,fill,stroke:C.line}));
      const limit=Math.max(10,Math.floor((w-18)/6.2)),lines=[];
      for(const original of Array.isArray(labels)?labels:[labels]){let line="";for(const word of String(original).split(" ")){if(line&&(line+" "+word).length>limit){lines.push(line);line=word;}else line+=(line?" ":"")+word;}if(line)lines.push(line);}
      const spacing=Math.min(19,(h-9)/Math.max(1,lines.length));lines.forEach((s,i)=>text(x+w/2,y+h/2+(i-(lines.length-1)/2)*spacing+4,s,{"text-anchor":"middle","font-size":12}));
    }
    function arrow(x1,y1,x2,y2,color=C.muted,dashed=false){
      picture.append(svgNode("line",{x1,y1,x2,y2,stroke:color,"stroke-width":1.7,...(dashed?{"stroke-dasharray":"5 4"}:{})}));
      const angle=Math.atan2(y2-y1,x2-x1),len=7,p1=[x2-len*Math.cos(angle-.4),y2-len*Math.sin(angle-.4)],p2=[x2-len*Math.cos(angle+.4),y2-len*Math.sin(angle+.4)];picture.append(svgNode("polygon",{points:`${x2},${y2} ${p1} ${p2}`,fill:color}));
    }
    function finish(caption,steps,tex,sources){
      body.append(node("figcaption",caption,"diagram-caption"));const list=node("ol",undefined,"diagram-steps");steps.forEach(s=>list.append(node("li",s)));body.append(list);
      if(tex){const f=node("div",undefined,"formula");math(f,tex);body.append(f);}
      const links=node("p",undefined,"diagram-sources");links.append(node("span","Sources: "));sources.forEach(([label,url])=>{const a=node("a",label);a.href=url;a.target="_blank";a.rel="noopener noreferrer";links.append(a);});body.append(links);
    }
    function table(headings,rows,parent=body){const wrap=node("div",undefined,"table-scroll"),t=node("table"),head=node("thead"),tr=node("tr"),tbody=node("tbody");headings.forEach(h=>tr.append(node("th",h)));head.append(tr);rows.forEach(row=>{const r=node("tr");row.forEach(v=>r.append(node("td",String(v))));tbody.append(r);});t.append(head,tbody);wrap.append(t);parent.append(wrap);}
    function render(){
      body.replaceChildren();
      body.append(node("p",diagramTerms[id],"topic-terms"));
      if(id==="architecture"){
        const c=settings.model_config||settings,d=c.hidden_size||c.width||64,h=c.num_attention_heads||c.heads||4,kv=c.num_key_value_heads||h,l=c.num_hidden_layers||c.layers||2,ff=c.intermediate_size||4*d,rope=!!c.rope_theta,norm=c.normalization==="none"?"Identity":c.rms_norm_eps||c.normalization==="rms"?"RMSNorm":"LayerNorm",residual=c.residuals!==false;
        canvas(480,`A complete pre-normalized decoder block: width ${d}, ${h} query heads, ${kv} KV heads, repeated ${l} times`);
        box(150,15,420,48,[`Token vectors X · [B, T, ${d}]`,rope?"Rotary positions enter Q/K inside attention":c.positions===false?"No explicit position embeddings":"Add learned position embeddings"],C.green);
        box(150,98,420,43,`${norm} → attention input`);arrow(360,63,360,98);
        box(150,176,420,54,[`${h} query heads · ${kv} key/value heads · head width ${d/h}`,"Causal attention → output projection"],C.pale);arrow(360,141,360,176);
        box(150,265,420,35,residual?"Add attention output to X":"Residual addition disabled",C.sand);arrow(360,230,360,265);
        box(150,335,420,52,[`${norm} → ${rope?"gated SiLU":"GELU"} feed-forward`,`${d} → ${ff} → ${d} features per token`]);arrow(360,300,360,335);
        box(150,422,420,35,residual?"Add feed-forward input back → next block":"Feed-forward output → next block",C.green);arrow(360,387,360,422);
        if(residual){arrow(150,40,74,40,C.teal);arrow(74,40,74,282,C.teal);arrow(74,282,150,282,C.teal);text(14,166,"skip",{"font-size":11});arrow(570,282,643,282,C.teal);arrow(643,282,643,440,C.teal);arrow(643,440,570,440,C.teal);}
        finish(`One block is shown completely and repeats ${l} times with different learned weights. All sublayers return width ${d}. A preview describes configuration, not model quality.`,[
          "The batch axis B groups examples; T counts token positions; d is the feature width. Attention mixes positions; the feed-forward network transforms each position separately.",
          `Queries split into ${h} heads of ${d/h} components. ${kv<h?"Grouped queries share fewer key/value heads; this saves KV storage.":"Here every query head has its own key/value head."}`,
          "Each residual addition requires matching shapes. The skip path carries the input around a sublayer rather than concatenating a new feature axis.",
          "After the final block, the model applies its configured final normalization and projects to vocabulary scores. Softmax gives next-token probabilities."
        ],residual?String.raw`H=X+\operatorname{Attention}(\operatorname{Norm}(X)),\quad Y=H+\operatorname{FFN}(\operatorname{Norm}(H))`:String.raw`H=\operatorname{Attention}(\operatorname{Norm}(X)),\quad Y=\operatorname{FFN}(\operatorname{Norm}(H))`,[["Transformer paper","https://arxiv.org/abs/1706.03762"],["SmolLM2 configuration","https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/config.json"]]);
      }else if(id==="architectures"){
        canvas(400,"Decoder, encoder, encoder–decoder and mixture-of-experts model structures");
        const rows=[["Decoder","Earlier output","Causal self-attention","Next token"],["Encoder","Whole input","Bidirectional attention","Token / task outputs"],["Encoder–decoder","Source → encoder","Decoder cross-attends","Output sequence"],["Sparse experts","Attention output","Router → selected MLPs","Weighted merge"]];
        rows.forEach((r,i)=>{const y=20+i*95;text(15,y+17,r[0],{"font-weight":600,"font-size":11});box(150,y,155,55,r[1]);box(337,y,190,55,r[2],C.green);box(559,y,150,55,r[3]);arrow(305,y+28,337,y+28);arrow(527,y+28,559,y+28);});
        finish("Compare the information available at each prediction, not only the model's name. Expert routing changes the feed-forward path; it is not a separate causal language-model objective.",[
          "A causal decoder cannot read future output tokens. A bidirectional encoder can read both sides of an unmasked input position.",
          "An encoder–decoder separates source representation from output generation. Cross-attention uses decoder queries with encoder keys and values.",
          "A sparse expert block chooses some feed-forward networks for each token; total stored parameters and active compute are different quantities.",
          "Recurrent/state-space models summarize history in a state. Hybrids combine state updates and attention; multimodal systems add modality encoders/fusion. Their exact variants are described in the lesson."
        ],null,[["BERT","https://arxiv.org/abs/1810.04805"],["T5","https://www.jmlr.org/papers/v21/20-074.html"],["Switch Transformer","https://www.jmlr.org/papers/v23/21-0998.html"],["Mamba","https://arxiv.org/abs/2312.00752"]]);
      }else if(id==="attention"){
        canvas(360,"Attention: Q/K/V projections, scaled masked scores, softmax weights and a weighted value sum");
        box(15,18,160,60,["Input X","[B, T, d]"]);["Q = XWq","K = XWk","V = XWv"].forEach((v,i)=>{box(245,18+i*90,180,52,v,i===2?C.green:C.pale);arrow(175,48,245,44+i*90);});
        box(495,18,208,62,["QKᵀ / √d_head","Add causal mask"],C.sand);arrow(425,44,495,44);arrow(425,134,495,62);
        box(495,128,208,52,["Row-wise softmax","Each full row sums to 1"],C.green);arrow(599,80,599,128);
        box(495,230,208,52,["Weights × V","Mix allowed positions"]);arrow(599,180,599,230);arrow(425,224,495,255);
        text(24,323,"Worked final row: scaled scores [1/√2, 0, blocked] → weights [0.670, 0.330, 0].",{"font-size":13});
        finish("The whole calculation is visible. The worked row has two permitted keys and one masked key; the zeros come from the mask, not an unimportant future token.",[
          "Project token vectors into queries Q, keys K and values V. For one head, each has head width d_head.",
          "Compare each query with each key. Scale by √d_head; use −∞ for forbidden score entries before softmax.",
          "For query [1, 0], keys [1, 0] and [0, 1], and head width 2, scaled scores are [1/√2, 0]. Softmax is [0.6698, 0.3302]; values [2, 0] and [0, 4] mix into [1.3395, 1.3210].",
          "Join head outputs and project back to model width. An attention weight is a mixing coefficient, not a factual-confidence score or causal explanation."
        ],String.raw`\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{\rm head}}}+M\right)V`,[["Attention Is All You Need","https://arxiv.org/abs/1706.03762"],["PyTorch attention","https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"]]);
      }else if(id==="masking"){
        canvas(310,"Causal visibility matrix versus a separate shifted-target scoring mask");
        text(20,30,"Attention: permitted query–key pairs",{"font-weight":600,"font-size":12});
        text(371,30,"Loss: shifted targets that count",{"font-weight":600,"font-size":12});
        for(let i=0;i<4;i++){
          text(109+i*45,66,"K"+(i+1),{"text-anchor":"middle","font-size":11});text(25,100+i*45,"Q"+(i+1),{"font-size":11});
          for(let j=0;j<4;j++)box(90+j*45,79+i*45,38,32,j<=i?"1":"0",j<=i?C.green:C.pale);
          box(370+i*81,90,72,40,"Target "+(i+1));text(406+i*81,152,i<2?"masked":"scored",{"text-anchor":"middle","font-size":11});
          box(370+i*81,172,72,35,i<2?"m = 0":"m = 1",i<2?C.pale:C.green);
        }
        text(20,290,"1 = allowed; 0 = blocked before softmax",{"font-size":11});text(371,248,"Visible context can remain unscored.",{"font-size":12});text(371,273,"A loss mask never blocks attention.",{"font-size":12});
        finish("These masks answer different questions. The square matrix changes information flow; the vector changes the loss denominator and which shifted targets train the model.",[
          "For query position Q3, keys K1–K3 are allowed and K4 is blocked. A causal mask blocks future information even when the future tokens are stored in the same batch.",
          "A zero loss-mask entry excludes that target's prediction from the objective. It does not prevent later response positions from reading the corresponding prompt input.",
          "When packing independent documents, combine the causal rule with same-document visibility if isolation is intended. Preserve the separately computed response/padding target mask."
        ],String.raw`M_{ij}=\begin{cases}0&j\le i\\-\infty&j>i\end{cases},\qquad L=-\frac{\sum_t m_t\log p_t}{\sum_t m_t}`,[["Causal attention","https://arxiv.org/abs/1706.03762"],["SFT loss masks","https://huggingface.co/docs/trl/sft_trainer"]]);
      }else if(id==="data"){
        canvas(300,"Shifted next-token inputs and separate visibility versus scoring masks");
        const input=["<start>","User",": question","Assistant",": answer"],target=["User",": question","Assistant",": answer","<end>"];
        text(15,74,"Input",{"font-weight":600,"font-size":12});text(15,151,"Next-token",{"font-weight":600,"font-size":12});text(15,169,"targets",{"font-weight":600,"font-size":12});
        input.forEach((v,i)=>{box(140+i*114,49,104,38,v);arrow(192+i*114,87,192+i*114,136);box(140+i*114,136,104,38,target[i],i>=3?C.green:C.pale);text(192+i*114,205,i>=3?"score response":"loss masked",{"text-anchor":"middle","font-size":10});});
        text(20,258,"Causal attention: each input position can read itself and earlier inputs, never a later input.",{"font-size":12});
        finish("Each box is an illustrative span, not necessarily one real tokenizer token. In actual code, response boundaries and the one-position shift must be computed at token resolution.",[
          "A next-token target comes from the original sequence, shifted left relative to the input. Training can score many positions in parallel while still blocking future input.",
          "An attention mask controls information flow. A loss mask controls which target positions contribute to the objective. A user prompt can be visible but excluded from loss.",
          "Padding must not create scored targets. Packing separate documents may also require a block-diagonal visibility mask and boundary-target rules; an end marker alone is not an attention barrier."
        ],String.raw`L=-\frac{\sum_t m_t\log p_\theta(x_{t+1}\mid x_{\le t})}{\sum_t m_t}`,[["Causal language modelling","https://huggingface.co/docs/transformers/tasks/language_modeling"],["Response-only loss","https://huggingface.co/docs/trl/sft_trainer"]]);
      }else if(id==="training-loop"){
        canvas(420,"Data preparation, repeated gradient updates, held-out validation, checkpointing and inference");
        const boxes=[[20,20,"Prepare and split","Data / tokenizer / masks"],[260,20,"Build a training batch","Inputs + known targets"],[500,20,"Forward and loss","Predictions, then penalty"],[500,160,"Backward","Compute gradients only"],[260,160,"Accumulate / clip","Combine intended targets"],[20,160,"Optimizer step","Change trainable weights"],[20,310,"Validation","No gradient update"],[260,310,"Save selected checkpoint","Plus resume state"],[500,310,"Inference","Predict new output"]];
        boxes.forEach(([x,y,a,b],i)=>box(x,y,200,64,[a,b],i===5?C.sand:i>=6?C.green:C.pale));
        [[220,52,260,52],[460,52,500,52],[600,84,600,160],[500,192,460,192],[260,192,220,192],[120,224,120,310],[220,342,260,342],[460,342,500,342]].forEach(a=>arrow(...a));
        arrow(20,192,6,192,C.teal);arrow(6,192,6,115,C.teal);arrow(6,115,360,115,C.teal);arrow(360,115,360,84,C.teal);text(72,107,"Repeat on new training batches",{"font-size":11,fill:C.teal});
        finish("The repeat path is the training loop. Validation and inference reuse weights but do not update them. Data preparation and parameter initialization are not repeated from scratch at every batch.",[
          "Forward builds predictions; loss compares them with the permitted known targets. Backward computes gradients without applying an update.",
          "Accumulate the intended token-weighted gradients, unscale if using loss scaling, then clip once at the update boundary. The optimizer changes parameters and its own state.",
          "Validate periodically using separate data, evaluation mode and no backward pass. Select checkpoints on validation, not on the final test set.",
          "A resumable checkpoint needs optimizer/schedule/random/sampler state as well as weights. Ordinary generation has no optimizer step."
        ],String.raw`\theta_{t+1}=\operatorname{Optimizer}(\theta_t,\nabla_\theta L_t,\text{optimizer state})`,[["Autograd mechanics","https://docs.pytorch.org/docs/stable/notes/autograd.html"],["Saving checkpoints","https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html"]]);
      }else if(id==="batching"){
        canvas(270,"Microbatch loss sums are weighted by valid target counts before one optimizer update");
        [[10,2],[30,4],[60,3]].forEach(([n,l],i)=>{box(20+i*240,20,200,75,[`Microbatch ${i+1}: ${n} targets`,`Mean loss ${l} → loss sum ${n*l}`]);arrow(120+i*240,95,360,155);});box(160,155,400,75,["Total loss sum 320 / total targets 100","Matched mean loss = 3.2 → one update"],C.green);
        finish("Unequal numbers of valid targets make an unweighted mean of batch means the wrong token-average objective. Accumulation is about the denominator as well as memory.",[
          "Count scored target tokens after shifting, padding and response masking—not all input tokens.",
          "Sum each microbatch's token losses, then divide by the common target count for the update. If counts are known in advance, backward each loss sum divided by that total.",
          "The example gives (10×2 + 30×4 + 60×3)/100 = 3.2. Averaging 2, 4 and 3 instead gives 3.0.",
          "With ordinary DDP gradient averaging, account for the world-size factor when normalizing by the global token count. Clip and step only after the complete accumulation window."
        ],String.raw`L=\frac{\sum_b n_b\bar L_b}{\sum_b n_b},\qquad \nabla L=\sum_b\frac{n_b}{\sum_j n_j}\nabla\bar L_b`,[["Gradient accumulation / AMP","https://docs.pytorch.org/docs/stable/notes/amp_examples.html"]]);
      }else if(id==="normalization"){
        canvas(260,"Pre-normalized sublayer and residual addition, with LayerNorm and RMSNorm alternatives");
        box(20,65,120,50,"Input x");box(230,65,155,50,"Normalize");box(465,65,140,50,"Sublayer F");box(645,65,55,50,"+");arrow(140,90,230,90);arrow(385,90,465,90);arrow(605,90,645,90);arrow(80,115,80,185,C.teal);arrow(80,185,670,185,C.teal);arrow(670,185,670,115,C.teal);text(22,228,"The bypass carries x unchanged; the learned branch adds F(Norm(x)).",{"font-size":13});
        finish("Normalization acts across one token's feature components in this decoder. Residual addition preserves the feature width. This is pre-normalization; post-normalization uses a different ordering.",[
          "LayerNorm subtracts the feature mean and divides by √(variance + ε), then applies learned scale and optional bias.",
          "RMSNorm divides by √(mean squared feature value + ε), without mean subtraction. It is not identical to LayerNorm.",
          "For [1, 3], ignoring ε and learned affine terms, LayerNorm gives [−1, 1], while RMSNorm gives [1/√5, 3/√5].",
          "The residual derivative includes an identity path. This can help gradient flow but does not guarantee stable training at arbitrary depth or learning rate."
        ],String.raw`y=x+F(\operatorname{Norm}(x)),\quad \operatorname{RMS}(x)=\sqrt{\frac1d\sum_i x_i^2+\epsilon}`,[["LayerNorm","https://arxiv.org/abs/1607.06450"],["RMSNorm","https://arxiv.org/abs/1910.07467"]]);
      }else if(id==="lora"){
        const d=settings.model_config?.hidden_size||4096,r=settings.rank||8;
        canvas(290,"Frozen weight path and a parallel trainable low-rank path merge at the output");box(20,100,110,50,"Input X");box(265,20,210,58,["Frozen W",`${d} × ${d} query projection`]);box(195,180,170,55,["Trainable A",`${d} × ${r} · row convention`],C.sand);box(405,180,170,55,["Trainable B",`${r} × ${d} · row convention`],C.sand);box(630,100,70,50,"+",C.green);arrow(130,125,265,49);arrow(130,125,195,207);arrow(365,207,405,207);arrow(475,49,665,100);arrow(575,207,665,150);
        finish("The frozen path remains in the computation. Only the small adapter matrices are trainable. The diagram uses row-vector notation; PyTorch stores linear-layer weights transposed relative to that convention.",[
          `For a ${d}×${d} projection at rank ${r}, the adapter has ${2*d*r} parameters rather than ${d*d} in a full matrix update. This count is for one projection, not the whole model.`,
          "The adapter adds a scaled low-rank change to the original output. Freezing W avoids its parameter-gradient/optimizer storage, but activations and forward computation still cost memory.",
          "Grouped-query attention can make the value projection's output width smaller than the query projection. Count each adapted module's actual shape.",
          "Training an adapter is not pretraining a new language model. Compare with the unchanged base under the same template, data and evaluation settings."
        ],String.raw`Y=XW+\frac{\alpha}{r}XAB,\quad A\in\mathbb R^{d_{in}\times r},\quad B\in\mathbb R^{r\times d_{out}}`,[["LoRA","https://arxiv.org/abs/2106.09685"],["PEFT implementation","https://huggingface.co/docs/peft/developer_guides/lora"]]);
      }else if(id==="preferences"){
        canvas(330,"DPO uses preference pairs; PPO and GRPO use sampled responses and reward-based updates");
        const rows=[["DPO","Fixed chosen / rejected pair","Policy + reference log-probs","Pairwise preference loss"],["PPO","Sample from old policy","Reward + value baseline","Clipped policy update"],["GRPO","Sample a response group","Rewards → group advantages","Clipped policy update"]];
        rows.forEach(([a,b,c,d],i)=>{const y=25+i*100;text(15,y+30,a,{"font-weight":600});box(85,y,180,65,b);box(302,y,190,65,c,C.green);box(529,y,178,65,d);arrow(265,y+32,302,y+32);arrow(492,y+32,529,y+32);});
        finish("All three can change language-model weights, but the data and optimization loops differ. This is a comparison diagram, not a claim that full LLM RL runs in this lab.",[
          "DPO trains on labelled pairs and compares policy log-probability margins against a reference; it does not require an explicit reward model or fresh rollout for each ordinary offline update.",
          "PPO generates on-policy data, estimates advantages, and limits policy movement with a clipped surrogate. Standard RLHF often adds a reference-policy KL penalty and a learned value model.",
          "GRPO uses multiple sampled responses per prompt and derives advantages from their relative rewards, avoiding a separate value critic in its original form. Normalization, length handling and loss variants matter.",
          "Reward or preference improvement is not automatically factuality, safety or generalization. Evaluate on independent outcomes, not only the training reward."
        ],null,[["DPO","https://arxiv.org/abs/2305.18290"],["PPO","https://arxiv.org/abs/1707.06347"],["DeepSeekMath / GRPO","https://arxiv.org/abs/2402.03300"]]);
      }else if(id==="inference"){
        canvas(290,"Prefill populates each layer's KV cache; decoding appends one token's keys and values");box(20,30,200,70,["Prefill prompt","Compute its K/V at each layer"]);box(270,30,210,70,["Stored K/V cache","T positions · L layers"],C.green);arrow(220,65,270,65);box(20,180,200,65,["One new token","Compute its new Q/K/V"]);box(270,180,210,65,["New Q reads old + new K/V","Produce next-token scores"]);box(530,180,170,65,["Append new K/V","Repeat decode"],C.sand);arrow(220,212,270,212);arrow(375,100,375,180);arrow(480,212,530,212);arrow(615,180,615,65);arrow(615,65,480,65);
        finish("Cache entries are request state, not permanent learned memory. The diagram shows ordinary causal cached decoding; a simple implementation can instead recompute the prefix.",[
          "Prefill computes the prompt in parallel over its positions. Decode uses the previous cache so old K/V projections are not recomputed at every output step.",
          "Raw cache bytes are 2×B×T×L×H_KV×d_head×bytes_per_number. The factor 2 is for keys and values; use KV heads, not query heads.",
          "For one SmolLM2 request, 2×1024×32×5×64×2 = 41,943,040 bytes, or 40 MiB in two-byte cache elements, before allocator/serving overhead.",
          "GQA reduces stored KV-head count. Paged allocation reduces fragmentation; it does not remove the information stored for the retained positions."
        ],String.raw`M_{KV}=2BTLH_{KV}d_{head}b`,[["GQA","https://arxiv.org/abs/2305.13245"],["PagedAttention","https://arxiv.org/abs/2309.06180"]]);
      }else if(id==="distributed"){
        canvas(320,"Two data-parallel workers combine gradients; sharding changes where model state resides");
        box(20,25,260,75,["Worker 1 · same parameters","Different batch → local gradients"]);box(440,25,260,75,["Worker 2 · same parameters","Different batch → local gradients"]);box(235,150,250,60,["All-reduce gradients","Normalize the global objective"],C.green);arrow(150,100,300,150);arrow(570,100,420,150);box(20,250,260,45,"Same optimizer update");box(440,250,260,45,"Same optimizer update");arrow(300,210,150,250);arrow(420,210,570,250);
        finish("Ordinary data parallelism replicates the model. ZeRO/FSDP can shard state; tensor and pipeline parallelism split different parts of the computation. More workers do not guarantee faster training.",[
          "Each replica computes a gradient on different data. Synchronization combines these into the gradient of the intended global objective.",
          "If workers have unequal numbers of valid targets, normalize by the global target count, accounting for the framework's reduction convention.",
          "ZeRO stages progressively partition optimizer state, gradients and parameters. Full sharding temporarily gathers parameters for computation and then reshards them.",
          "Communication, small matrix sizes, pipeline bubbles and a slow worker can limit speed. Check numerical equivalence on a small case before scaling a job."
        ],String.raw`B_{effective}=B_{micro}\,A\,N_{workers}`,[["PyTorch DDP","https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"],["ZeRO","https://arxiv.org/abs/1910.02054"]]);
      }else if(id==="trace"){
        const t=options.trace||{},tokens=t.tokens||[],a=t.attention||[],n=Math.min(12,tokens.length);
        if(!n){body.append(node("p","Inspect a completed checkpoint to load actual values."));return;}
        canvas(100+n*32,"Actual first-layer, first-head attention weights: rows are queries and columns are keys");
        const cell=Math.min(34,400/n),x=220,y=55;for(let i=0;i<n;i++){text(x+i*cell+cell/2,35,String(i+1),{"text-anchor":"middle","font-size":11});text(14,y+i*32+20,`${i+1}. ${String(tokens[i]).slice(0,22)}`,{"font-size":11});for(let j=0;j<n;j++){const v=Number(a[i]?.[j])||0;picture.append(svgNode("rect",{x:x+j*cell,y:y+i*32,width:cell-2,height:28,fill:`rgba(54,100,163,${.06+.94*v})`,stroke:C.line}));if(n<=8)text(x+j*cell+cell/2,y+i*32+18,fixed(v,2),{"text-anchor":"middle","font-size":10,fill:v>.55?"#fff":C.ink});}}
        body.append(node("figcaption",`Measured, not a teaching simulation. Showing ${n} of ${tokens.length} captured positions; the full attention calculation used the complete captured input. A truncated displayed row need not sum to one. ${t.note||""}`,"diagram-caption"));
        table(["Position","Token ID","Token","Embedding components (first ≤8)"],tokens.slice(0,n).map((token,i)=>[i+1,t.ids?.[i]??"—",token,(t.embeddings?.[i]||[]).slice(0,8).map(v=>fixed(v,4)).join(", ")]));
        const exact=node("details",undefined,"diagram-details");exact.append(node("summary","Exact attention weights in the displayed slice"));table(["Query / key",...Array.from({length:n},(_,i)=>i+1)],a.slice(0,n).map((row,i)=>[i+1,...row.slice(0,n).map(v=>fixed(v,5))]),exact);body.append(exact);
        if(t.query&&t.key&&t.value){const projections=node("details",undefined,"diagram-details");projections.append(node("summary","Measured query, key and value components"),node("p",t.projection_note||"First layer/head, up to eight projected components. The full calculation used complete vectors."));table(["Position","Query components","Key components","Value components"],tokens.slice(0,n).map((_,i)=>[i+1,...[t.query,t.key,t.value].map(values=>(values[i]||[]).slice(0,8).map(v=>fixed(v,4)).join(", "))]),projections);body.append(projections);}
        if(t.next_tokens?.length)table(["Next token","Probability"],t.next_tokens.map(v=>[v.token??v.text??v.id,fixed((v.probability??v.prob??0)*100,3)+"%"]));
        if(t.training_update){const u=t.training_update,details=node("details",undefined,"diagram-details");details.append(node("summary","Measured diagnostic weight changes"),node("p",u.note),node("p",`One-batch loss ${fixed(u.loss_before,5)} → ${fixed(u.loss_after,5)}; gradient norm ${fixed(u.gradient_norm,5)}. This is not a held-out score.`));const rows=[];u.before.forEach((row,i)=>row.forEach((v,j)=>rows.push([`${i+1}, ${j+1}`,fixed(v,7),fixed(u.gradient[i][j],7),fixed(u.after[i][j],7)])));table(["Weight row, column","Before","Gradient","After"],rows,details);body.append(details);}
      }
    }
    render();return {updateSettings(next){if(!dead){settings=next;render();}},destroy(){dead=true;root.remove();}};
  }
  return {catalog,mount,math,figuresForLesson:id=>lessonFigures[id]||{},sceneForLesson:()=>null};
})();
