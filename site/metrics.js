"use strict";

// Pure arithmetic over authored cases; no training, checkpoint loading or API calls.
const MetricsLab=(()=>{
  const sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length,divide=(a,b)=>b?a/b:null;
  const fmt=x=>x===null?"undefined (empty denominator)":ArchitectureMath.fmt(x);
  function binary(threshold=.5){
    const scores=[.9,.8,.6,.4,.3,.1],labels=[1,0,1,1,0,0],predicted=scores.map(p=>Number(p>=threshold));
    let tp=0,fp=0,tn=0,fn=0;
    const cases=scores.map((p,i)=>{const kind=labels[i]?(predicted[i]?"TP":"FN"):(predicted[i]?"FP":"TN");if(kind==="TP")tp++;else if(kind==="FP")fp++;else if(kind==="TN")tn++;else fn++;return {score:p,true:labels[i],predicted:predicted[i],kind};});
    const positives=labels.filter(Boolean).length,negatives=labels.length-positives;
    const precision=divide(tp,tp+fp),recall=divide(tp,tp+fn),f1=divide(2*tp,2*tp+fp+fn);
    let wins=0;for(let i=0;i<scores.length;i++)if(labels[i])for(let j=0;j<scores.length;j++)if(!labels[j])wins+=scores[i]>scores[j]?1:scores[i]===scores[j]?.5:0;
    let seen=0,hits=0,ap=0;[...cases].sort((a,b)=>b.score-a.score).forEach(r=>{seen++;if(r.true){hits++;ap+=(hits/seen)/positives;}});
    const curve=[Infinity,...scores,-Infinity].map(t=>{const a=cases.filter(r=>r.score>=t);return {threshold:t,tpr:a.filter(r=>r.true).length/positives,fpr:a.filter(r=>!r.true).length/negatives};});
    return {cases,tp,fp,tn,fn,precision,recall,f1,accuracy:(tp+tn)/6,fpr:fp/negatives,rocAuc:wins/(positives*negatives),averagePrecision:ap,curve};
  }
  function multiclass(index=0){
    const matrix=[[4,1,0],[1,2,1],[0,2,3]],n=sum(matrix.flat());
    const classes=matrix.map((row,i)=>{const tp=row[i],support=sum(row),predicted=sum(matrix.map(r=>r[i])),fp=predicted-tp,fn=support-tp;return {tp,fp,fn,support,precision:divide(tp,predicted),recall:divide(tp,support),f1:divide(2*tp,2*tp+fp+fn)};});
    const correct=sum(classes.map(x=>x.tp));
    return {matrix,classes,selected:classes[index],n,accuracy:correct/n,macroF1:mean(classes.map(x=>x.f1)),weightedF1:sum(classes.map(x=>x.f1*x.support))/n,microF1:correct/n,balancedAccuracy:mean(classes.map(x=>x.recall))};
  }
  function regression(){
    const truth=[1,2,3,4],prediction=[1,3,2,5],residual=prediction.map((x,i)=>x-truth[i]),mse=mean(residual.map(x=>x*x)),sst=sum(truth.map(x=>(x-mean(truth))**2));
    return {truth,prediction,residual,mae:mean(residual.map(Math.abs)),mse,rmse:Math.sqrt(mse),r2:1-sum(residual.map(x=>x*x))/sst};
  }
  function language(mask="all"){
    const probabilities=[.5,.25,.1],selected=mask==="all"?[true,true,true]:[false,true,true],losses=probabilities.map(p=>-Math.log(p));
    const total=sum(losses.filter((_,i)=>selected[i])),count=selected.filter(Boolean).length,nll=total/count;
    return {probabilities,selected,losses,total,count,nll,perplexity:Math.exp(nll),bitsPerByte:mask==="all"?total/Math.log(2)/12:null};
  }
  function retrieval(k=3){
    const ranked=["B","A","C"],relevant=["A","C"],hits=ranked.slice(0,k).filter(x=>relevant.includes(x)).length;
    let found=0,ap=0;ranked.forEach((x,i)=>{if(relevant.includes(x)){found++;ap+=found/(i+1)/relevant.length;}});
    const dcg=sum(ranked.map((x,i)=>relevant.includes(x)?1/Math.log2(i+2):0)),idcg=1+1/Math.log2(3);
    return {ranked,relevant,hits,k,precision:hits/k,recall:hits/relevant.length,reciprocalRank:.5,averagePrecision:ap,ndcg:dcg/idcg};
  }
  function measure(c){
    if(c.family==="classification")return binary(c.threshold);
    if(c.family==="multiclass")return multiclass(c.classIndex);
    if(c.family==="regression")return regression();
    if(c.family==="language")return language(c.mask);
    if(c.family==="retrieval")return retrieval(c.k);
    throw Error("Unsupported metric family.");
  }
  const C=ConceptLessons,S=C.step,Q=C.choice,T=C.table;
  const controls=[
    Q("family","Prediction task",[["classification","Binary classification"],["multiclass","Multiclass averaging"],["regression","Numeric regression"],["language","Language-model prediction"],["retrieval","Ranked retrieval"]],"The task determines what is counted and which denominator is meaningful."),
    Q("threshold","Predict positive at probability ≥",[[.3,"0.3"],[.5,"0.5"],[.7,"0.7"],[.95,"0.95 · no predicted positives"]],"Scores/labels stay fixed; only decisions change."),
    Q("classIndex","Class to inspect",[[0,"Class A"],[1,"Class B"],[2,"Class C"]],"A one-versus-rest view counts this class as positive."),
    Q("mask","Scored language-model targets",[["all","All three targets"],["response","Last two targets only"]],"Loss averages only the selected targets; do not invent byte counts for a different scored region."),
    Q("k","Retrieved prefix length",[[1,"Top 1"],[2,"Top 2"],[3,"Top 3"]],"Precision@k and recall@k inspect this prefix; full-ranking AP/nDCG remain fixed.")
  ];
  C.add("P8",{title:"See exactly what a metric counts",subtitle:"Choose a task, then inspect its cases, denominator and result.",defaults:{family:"classification",threshold:.5,classIndex:0,mask:"all",k:3},controls,sources:[["Metric definitions and conventions","https://scikit-learn.org/stable/modules/model_evaluation.html"],["F1 and averaging","https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"],["Perplexity","https://huggingface.co/docs/transformers/perplexity"]],build:c=>{
    const m=measure(c),steps=[],add=(...a)=>steps.push(S(...a));
    if(c.family==="classification"){
      add("Turn scores into decisions","Separate probability estimation from the action rule.","The positive class is the class whose detection we are measuring.","Six authored scores and known labels.","Predict positive at the chosen threshold.","\\hat y_i=\\mathbf1[p_i\\ge\\tau]","At cutoff "+c.threshold+", predictions are ["+m.cases.map(x=>x.predicted)+"].","Six binary decisions.","Changing the cutoff changes counts, not the fixed model scores.");
      add("Count the four outcomes","Name each error before combining it into a metric.","True/false describes correctness; positive/negative describes the predicted label.","Predictions paired with known labels.","Count True Positive, False Positive, True Negative and False Negative.","N=TP+FP+TN+FN","TP="+m.tp+", FP="+m.fp+", TN="+m.tn+", FN="+m.fn+".","A confusion matrix.","Rows in this view are true labels; columns are predicted labels.");
      add("Accuracy: all correct cases","Ask what fraction of the whole sample was correct.","Accuracy gives each example one vote, so common classes can dominate.","All six examples.","Divide correct positive/negative decisions by all examples.","Accuracy=\\frac{TP+TN}{N}","("+m.tp+"+"+m.tn+")/6 = "+fmt(m.accuracy)+".","Accuracy "+fmt(m.accuracy)+".","A majority-only classifier can score highly on an imbalanced dataset.");
      add("Precision: inspect predicted positives","Ask how reliable a positive prediction was.","Precision's denominator is the predicted-positive column, not every actual positive.","TP="+m.tp+" and FP="+m.fp+".","Count correct positives among all predicted positives.","Precision=\\frac{TP}{TP+FP}",m.tp+"/("+m.tp+"+"+m.fp+") = "+fmt(m.precision)+".","Precision "+fmt(m.precision)+".","When no positive is predicted, this ratio is undefined. Libraries may choose a zero-division convention; state it.");
      add("Recall: inspect actual positives","Ask how many positives were found.","Recall's denominator is the actual-positive row, including missed positives.","TP="+m.tp+" and FN="+m.fn+".","Count found positives among all actual positives.","Recall=\\frac{TP}{TP+FN}",m.tp+"/("+m.tp+"+"+m.fn+") = "+fmt(m.recall)+".","Recall "+fmt(m.recall)+".","Lowering the cutoff often finds more positives but may add false alarms.");
      add("F1: combine precision and recall","Reward finding positives without ignoring false alarms.","F1 is the harmonic mean of precision/recall when those ratios are defined; the count formula also handles some zero cases.","True positives, false positives and false negatives.","Use the count form directly.","F1=\\frac{2TP}{2TP+FP+FN}","2×"+m.tp+"/(2×"+m.tp+"+"+m.fp+"+"+m.fn+") = "+fmt(m.f1)+".","F1 "+fmt(m.f1)+".","F1 ignores true negatives. It is not the arithmetic average of precision and recall, or a universal utility score.");
      add("Compare thresholds without mixing metrics","A ranking score and one threshold's decision score answer different questions.","ROC-AUC compares positive/negative score order; AP weights precision at positive recall increments.","The same scores across possible thresholds.","Use ROC/PR curves for discrimination; choose a threshold on development data.","TPR=Recall,\\quad FPR=\\frac{FP}{FP+TN}","At this cutoff FPR="+fmt(m.fpr)+". Full score ranking: ROC-AUC="+fmt(m.rocAuc)+", AP="+fmt(m.averagePrecision)+".","Decision rates and full-ranking scores.","AUC does not assess probability calibration; AP is not automatically trapezoidal PR area.");
    }else if(c.family==="multiclass"){
      const s=m.selected;
      add("Read the multiclass confusion matrix","Keep actual and predicted axes consistent.","The diagonal contains correct predictions; support is the true count in a class.","A 3×3 authored matrix with "+m.n+" examples.","Inspect row/column counts for class "+["A","B","C"][c.classIndex]+".","N=\\sum_{i,j}C_{ij}","Selected class: TP="+s.tp+", FP="+s.fp+", FN="+s.fn+", support="+s.support+".","One-versus-rest class counts.","A class's false positives come from its column outside the diagonal.");
      add("Calculate each class's F1","Do the class calculation before averaging.","Each class takes its turn as the positive class.","Per-class true positives and errors.","Compute 2TP/(2TP+FP+FN) separately for A/B/C.","F1_c=\\frac{2TP_c}{2TP_c+FP_c+FN_c}","Class F1 values: "+m.classes.map(x=>fmt(x.f1)).join(", ")+".","Three class-level F1 values.","Averaging class precision and recall first, then taking F1, is not macro-F1.");
      add("Macro-F1: give each class equal weight","Make class balance in the metric explicit.","Macro averaging gives every included class the same weight.","The three F1 values.","Take their ordinary mean.","F1_{macro}=\\frac1C\\sum_cF1_c","Macro-F1 = "+fmt(m.macroF1)+".","Macro-F1 "+fmt(m.macroF1)+".","Absent/excluded labels and zero-division conventions can change the result.");
      add("Weighted and micro averaging","These aggregate different objects.","Weighted F1 averages class F1 by support; micro pools decision counts before scoring.","Per-class F1/support and all one-versus-rest counts.","Compute support weighting and pooled-count F1.","F1_{weighted}=\\sum_c\\frac{n_c}{N}F1_c","Weighted F1="+fmt(m.weightedF1)+"; micro-F1="+fmt(m.microF1)+"; accuracy="+fmt(m.accuracy)+".","Two distinct aggregates.","Micro-F1 equals accuracy here only because this is single-label multiclass with every label included.");
      add("Balanced accuracy: average class recall","Do not call every balanced score macro-F1.","Balanced accuracy gives each class recall equal weight.","Recall per class, not class F1.","Average class recalls.","BalancedAccuracy=\\frac1C\\sum_c Recall_c","Balanced accuracy = "+fmt(m.balancedAccuracy)+".","Balanced accuracy "+fmt(m.balancedAccuracy)+".","For multilabel tasks and other averaging rules, identify the precise definition used by the evaluator.");
    }else if(c.family==="regression"){
      add("Inspect numeric residuals","Keep the original target units.","A residual is prediction minus observed target.","Targets [1,2,3,4], predictions [1,3,2,5].","Subtract target from prediction for each case.","e_i=\\hat y_i-y_i","Residuals ["+m.residual+"].","Four signed errors.","Positive/negative errors can cancel in a mean signed error.");
      add("MAE: typical absolute error","Treat each error magnitude linearly.","MAE averages absolute residuals.","The four residuals.","Remove signs, then average.","MAE=\\frac1N\\sum_i|e_i|","(0+1+1+1)/4 = "+fmt(m.mae)+".","MAE "+fmt(m.mae)+" target units.","A large miss contributes in direct proportion to its magnitude.");
      add("MSE and RMSE: emphasize large misses","Squaring gives large errors greater influence.","MSE uses squared target units; RMSE restores target units.","The same residuals.","Average squares, then optionally take the square root.","MSE=\\frac1N\\sum_i e_i^2,\\quad RMSE=\\sqrt{MSE}","MSE="+fmt(m.mse)+"; RMSE="+fmt(m.rmse)+".","MSE/RMSE in stated units.","RMSE is not interchangeable with MAE even when this particular sample has similar values.");
      add("R²: compare with this sample's mean","Measure squared-error improvement over a mean reference.","The denominator is target variation around this evaluation sample's mean.","Error sum 3; target mean 2.5; target variation sum 5.","Subtract the ratio of residual error to mean-reference error.","R^2=1-\\frac{\\sum_i e_i^2}{\\sum_i(y_i-\\bar y)^2}","1−3/5 = "+fmt(m.r2)+".","R² "+fmt(m.r2)+".","R² can be negative; a constant-target denominator is zero and requires a declared convention. It is not an accuracy percentage.");
    }else if(c.family==="language"){
      add("Identify the scored targets","The loss mask determines the denominator.","A target probability is the probability assigned to the actual next token.","Authored target probabilities [0.5,0.25,0.1].","Include "+(c.mask==="all"?"all three targets":"only the last two targets")+" in the score.","N=\\sum_t m_t","Mask ["+m.selected.map(Number)+"]; valid count "+m.count+".","A defined scored region.","Do not average padded/user-prompt positions if the declared objective excludes them.");
      add("Calculate negative log-likelihood","A confident wrong token gets a large penalty.","Nats use natural logarithms.","Each selected target probability.","Take −log(p), then sum selected losses.","L_t=-\\log p_t(y_t)","Per-target losses ["+m.losses.map(fmt)+"]; selected sum "+fmt(m.total)+".","A summed prediction cost.","These scores do not measure the truth or usefulness of a generated answer.");
      add("Average, then exponentiate","Perplexity is derived from the mean loss, not an independent quality test.","Mean NLL divides by valid targets; perplexity is exp(mean NLL).","Selected sum and target count.","Divide once, then exponentiate.","NLL=\\frac{\\sum_t m_tL_t}{N},\\quad PPL=e^{NLL}","NLL="+fmt(m.nll)+" nats/target; perplexity="+fmt(m.perplexity)+".","Mean NLL and perplexity.","Comparisons need matched tokenization, text, context/stride and masks. Perplexity is not naturally defined for masked-language-model scoring in this form.");
      add("Bits per byte: use the correct raw-text unit","A byte-normalized score needs an aligned byte count.","Bits divide nats by log(2); the byte denominator belongs to the same scored text.","For all three targets, the authored aligned region has 12 UTF-8 bytes.","Convert summed nats to bits, then divide by aligned bytes.","BPB=\\frac{\\sum_tL_t}{\\log(2)\\,B}",m.bitsPerByte===null?"No byte count is supplied for the response-only region, so BPB is not reported.":"BPB="+fmt(m.bitsPerByte)+" bits/byte.","An aligned BPB value or an explicit unavailable result.","BPB can help compare tokenizers on identical text, but does not equalize context coverage, training compute or contamination.");
    }else{
      add("Fix relevance and the ranked list","Retrieval needs a gold relevance set, not generated answer correctness.","Relevant documents are A/C; the ranked list is B,A,C.","The ordered candidates and k="+c.k+".","Inspect the first k entries.","D_k=(d_1,\\ldots,d_k)","Prefix ["+m.ranked.slice(0,c.k)+"] contains "+m.hits+" relevant item(s).","A ranked prefix and gold set.","An incomplete relevance set can bias the measured result.");
      add("Precision@k and recall@k","Use different denominators for noise and coverage.","Precision divides by retrieved count; recall divides by known relevant count.","Hits="+m.hits+", k="+c.k+", relevant count=2.","Compute both ratios.","P@k=hits/k,\\quad R@k=hits/|Rel|","Precision@k="+fmt(m.precision)+"; recall@k="+fmt(m.recall)+".","Retrieval precision and recall.","Changing k changes the denominator and possibly the retrieved hits.");
      add("Reciprocal rank: first relevant hit","Some tasks value how soon the first useful result appears.","Reciprocal rank is one divided by the first relevant position.","Full ranking B,A,C.","The first relevant document is at rank 2.","RR=1/rank_{first}","RR=1/2=0.5.","Reciprocal rank 0.5.","MRR averages this quantity over queries; one query is not MRR.");
      add("Average precision and nDCG","Evaluate more of the ranked list.","AP rewards precision at relevant hits; nDCG discounts later gains and normalizes by the ideal ranking.","The complete three-document ranking and binary relevance.","Compute both full-list summaries.","AP=\\frac1{|Rel|}\\sum_j P@j\\,rel_j","AP=(1/2+2/3)/2="+fmt(m.averagePrecision)+"; nDCG@3="+fmt(m.ndcg)+".","Full-ranking AP/nDCG.","MAP averages AP over queries. These full-list values stay fixed when the inspected k changes; answer support/citation correctness need additional evaluation.");
    }
    const visible=[controls[0],...(c.family==="classification"?[controls[1]]:c.family==="multiclass"?[controls[2]]:c.family==="language"?[controls[3]]:c.family==="retrieval"?[controls[4]]:[])];
    return {steps,controls:visible,facts:m};
  }});
  function figure(c,selected,sv,el,choose){
    const m=measure(c),fig=el("figure",undefined,"ax-metric-figure"),svg=sv("svg",{viewBox:"0 0 430 340",class:"ax-inset-svg",role:"group","aria-label":"Metric cases and denominators"});
    const text=(x,y,value,size=12)=>svg.append(sv("text",{x,y,"font-size":size,"text-anchor":"middle",fill:"#294258"},String(value)));
    function box(x,y,w,h,value,color,step,active=false){
      const g=sv("g",{...(step?{role:"button",tabindex:0,"aria-label":String(value)}:{})});
      g.append(sv("rect",{x,y,width:w,height:h,rx:5,fill:color,stroke:active?"#1d6099":"#b4c2cf","stroke-width":active?3:1}),sv("text",{x:x+w/2,y:y+h/2+4,"text-anchor":"middle","font-size":12,fill:"#243e55"},String(value)));
      if(step){g.onclick=()=>choose(step);g.onkeydown=e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();choose(step);}};}svg.append(g);
    }
    let caption;
    if(c.family==="classification"&&selected==="s6"){
      text(215,26,"ROC: vary the threshold over the same scores");
      const x=a=>62+280*a,y=a=>270-210*a;
      svg.append(sv("line",{x1:62,y1:270,x2:342,y2:270,stroke:"#9bacbd"}),sv("line",{x1:62,y1:270,x2:62,y2:60,stroke:"#9bacbd"}));
      m.curve.slice(1).forEach((p,i)=>{const a=m.curve[i];svg.append(sv("line",{x1:x(a.fpr),y1:y(a.tpr),x2:x(p.fpr),y2:y(p.tpr),stroke:"#4479a4","stroke-width":2}));});
      svg.append(sv("circle",{cx:x(m.fpr),cy:y(m.recall),r:7,fill:"#b08035"}));
      text(215,310,"False-positive rate (0 → 1)");text(215,46,"Vertical: true-positive rate / recall (0 → 1)",11);
      caption="The amber point is the current threshold. The full curve and its AUC use the ranking, so moving one selected cutoff does not change AUC.";
    }else if(c.family==="classification"){
      text(115,25,"Confusion counts");text(307,25,"The six labelled cases");
      text(76,57,"pred 0",11);text(166,57,"pred 1",11);
      const precision=selected==="s3",recall=selected==="s4",f1=selected==="s5";
      const cells=[[m.tn,m.fp],[m.fn,m.tp]],names=[["True Negative","False Positive"],["False Negative","True Positive"]];
      cells.forEach((row,r)=>row.forEach((n,col)=>{
        const denominator=precision?col===1:recall?r===1:f1?(r===1||col===1):true,numerator=r===1&&col===1;
        box(32+col*90,75+r*77,83,68,n,denominator?(numerator?"#cce6dc":"#f1dfbd"):"#f1f4f7","s1",denominator);
        text(74+col*90,127+r*77,names[r][col],8);
      }));
      text(110,244,"Top row: true 0 · bottom row: true 1",9);
      m.cases.forEach((r,i)=>box(238,43+i*33,177,28,(i+1)+": "+r.score+" · "+r.true+" → "+r.predicted+" · "+r.kind,r.kind==="TP"?"#cce6dc":r.kind==="TN"?"#e5edf5":"#f1dfbd","s0"));
      const equation=precision?"Precision: "+m.tp+" / "+(m.tp+m.fp):recall?"Recall: "+m.tp+" / "+(m.tp+m.fn):f1?"F1: "+(2*m.tp)+" / "+(2*m.tp+m.fp+m.fn):"Accuracy: "+(m.tp+m.tn)+" / 6";
      text(215,282,equation,15);
      text(215,309,precision?fmt(m.precision):recall?fmt(m.recall):f1?fmt(m.f1):fmt(m.accuracy),12);
      caption="Outlined cells form the selected denominator. Green True Positives supply precision/recall's numerator. For F1 they count twice; False Positives and False Negatives count once. The case list reads score · true → predicted.";
    }else if(c.family==="multiclass"){
      text(215,24,"Rows: true class · columns: predicted class");
      ["A","B","C"].forEach((label,i)=>{text(110+i*88,55,label);text(48,102+i*72,label);});
      m.matrix.forEach((row,i)=>row.forEach((n,j)=>box(70+j*88,72+i*72,80,64,n,i===j?"#cce6dc":"#f1e4d1","s0",i===c.classIndex||j===c.classIndex)));
      text(215,319,"Outlined: selected class "+["A","B","C"][c.classIndex]+" row/column");
      caption="For the selected class, its diagonal is True Positive, its row's other cells are False Negative, and its column's other cells are False Positive. The averaging steps combine these class calculations.";
    }else if(c.family==="regression"){
      text(215,24,"Blue: target · amber: prediction");
      svg.append(sv("line",{x1:35,y1:260,x2:400,y2:260,stroke:"#a6b7c6"}));
      m.truth.forEach((truth,i)=>{const x=70+i*90,y=t=>260-t*40;svg.append(sv("line",{x1:x,y1:y(truth),x2:x,y2:y(m.prediction[i]),stroke:"#8f6b33","stroke-width":3}),sv("circle",{cx:x-5,cy:y(truth),r:7,fill:"#4479a4"}),sv("circle",{cx:x+5,cy:y(m.prediction[i]),r:7,fill:"#b08035"}));text(x,285,"Case "+(i+1),11);text(x,312,"error "+m.residual[i],11);});
      text(215,47,"Vertical: original target units (0 at the baseline)",10);
      caption="The vertical separation is the signed residual. MAE averages lengths; MSE averages their squares; RMSE takes the square root. R² adds a separate reference to the evaluation sample's mean.";
    }else if(c.family==="language"){
      text(215,25,"Negative log-probability for each known target");
      m.losses.forEach((loss,i)=>{const h=loss/2.5*175,x=57+i*122;box(x,245-h,70,h,fmt(loss),m.selected[i]?"#c8dff5":"#edf0f3","s1",m.selected[i]);text(x+35,270,"p="+m.probabilities[i],11);text(x+35,291,m.selected[i]?"scored":"excluded",11);});
      text(215,321,"Divide the selected total by "+m.count+" valid targets",12);
      caption="Outlined bars contribute to the mean loss. Perplexity exponentiates that mean; it is not an answer-accuracy score. A different mask changes the denominator and scored region.";
    }else{
      text(215,27,"Ranked results: earlier positions matter");
      m.ranked.forEach((name,i)=>box(62,57+i*67,305,55,"Rank "+(i+1)+": document "+name+(m.relevant.includes(name)?" · relevant":" · not relevant"),m.relevant.includes(name)?"#cce6dc":"#edf0f3","s0",i<c.k));
      text(215,301,"Prefix: "+m.hits+" relevant / "+c.k+" retrieved",13);
      caption="Outline marks the selected top-k prefix. Recall divides the same hits by all known relevant documents (2). Reciprocal rank, AP and nDCG describe the full ranking in the later steps.";
    }
    fig.append(svg,el("figcaption",caption,"ax-caption"));return fig;
  }
  return {binary,multiclass,regression,language,retrieval,measure,fmt,figure};
})();
