const corpus = ['the bank approved the loan','the bank offered the loan','the river bank flooded','the river flooded the path'];
const words = s => s.split(' ');
const unique = values => [...new Set(values)];
const esc = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const chips = items => '<div class="chips">'+items.map(x=>'<span>'+esc(x)+'</span>').join('')+'</div>';
const table = (headers, rows) => '<div class="tablewrap"><table><thead><tr>'+headers.map(x=>'<th scope="col">'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
const pct = x => (100*x).toFixed(1)+'%';
const next = ['the','cat','dog','sleeps','runs','EOS'];
const transitions = {BOS:{the:3},the:{cat:2,dog:1},cat:{sleeps:1,runs:1},dog:{runs:1},sleeps:{EOS:1},runs:{EOS:2}};
function distribution(previous, alpha){
 const counts=transitions[previous];
 const total=Object.values(counts).reduce((a,b)=>a+b,0)+alpha*next.length;
 return table(['Possible next token','Probability'],next.map(w=>[w,pct(((counts[w]||0)+alpha)/total)]));
}
const tokenWords={unhelpfulness:['un','help','ful','ness'],replaying:['re','play','ing'],unhappiness:['un','happi','ness'],connected:['connect','ed']};
const contexts={cat:[1,1,0,1,0,1,0],kitten:[1,1,0,1,0,1,0],dog:[1,0,1,0,1,0,1],puppy:[1,0,1,0,1,0,1]};
const scores=[.93,.82,.76,.68,.61,.55,.44,.36,.24,.12];
const claims=[1,0,1,1,0,1,0,1,0,0];
const shifts=[
 ['Input / covariate shift','Training: “Cities should fund public transport.”','Testing: “We ought to expand municipal transit.”','P_train(x) ≠ P_test(x). The vocabulary and writing style change, while the rule for assigning labels stays fixed.'],
 ['Label-prior shift','Source: Claims are rare.','Target: Claims are common.','P_train(y) ≠ P_test(y). In pure label shift, the distribution of inputs within each label stays fixed.'],
 ['Concept / conditional shift','Source: A type of statement typically signals a claim.','Target: That same type of statement is less likely to signal a claim.','P_train(y | x) ≠ P_test(y | x). The relationship between inputs and labels changes.'],
 ['Annotation-guideline shift','Source: A factual statement can count as a claim when it supports a conclusion.','Target: The guideline reserves the label for recommendations or judgments.','The labeling rule changes. The same text can receive different labels under the two guidelines.']
];
const demos={
 bank: ({sense}) => sense==='financial'
  ? '<p>“The bank approved the loan.”</p><strong>Likely meaning:</strong> financial institution. <strong>Context clues:</strong> approved, loan.'
  : '<p>“The fisherman sat on the river bank.”</p><strong>Likely meaning:</strong> land beside a river. <strong>Context clues:</strong> fisherman, river.',
 tokenize: ({word,level})=>{
   const variants={word:[word],subword:tokenWords[word],character:[...word]};
   return '<p><strong>Resulting tokens · '+variants[level].length+'</strong></p>'+chips(variants[level])+table(['Word','Subword','Character'],[[1,variants.subword.length,variants.character.length]]);
 },
 ngrams: ({n})=>{
   const xs=words('the cat chased the dog'), k=+n;
   return chips(xs)+table(['Position',k+'-gram feature'],xs.slice(0,xs.length-k+1).map((_,i)=>[i+1,xs.slice(i,i+k).join(' ')]));
 },
 tfidf: ({doc,term})=>{
   const selected=words(corpus[doc]);
   const tf=selected.filter(x=>x===term).length, df=corpus.filter(s=>words(s).includes(term)).length, idf=Math.log(corpus.length/df);
   return chips(selected)+table(['TF (count)','DF','IDF = ln(4/DF)','TF × IDF'],[[tf,df+'/4',idf.toFixed(3),(tf*idf).toFixed(3)]]);
 },
 history: ({order})=>{
   const n=+order, xs=words('the small black cat sleeps'), history=xs.slice(xs.length-n,-1);
   return '<p>Target: <strong>sleeps</strong></p>'+chips(history.length?history:['No context'])+'<p>Estimated conditional probability: <strong>P(sleeps'+(history.length?' | '+history.join(' '):'')+')</strong></p>';
 },
 bigram: ({previous})=>distribution(previous,0),
 smoothing: ({alpha})=>distribution('dog',+alpha),
 cooccurrence: ({target})=>table(['Target','drinks','sleeps','runs','milk','water','softly','quickly'],[[target,...contexts[target]]])+'<p>Selected vector: <strong>['+contexts[target].join(', ')+']</strong></p>',
 ner: ({error})=>{
   const tokens=words('Bright Labs acquired Nova Systems in New York .'), gold=words('B-ORG I-ORG O B-ORG I-ORG O B-LOC I-LOC O');
   const predicted=gold.map((tag,i)=>error==='1' && i===4?'O':tag);
   const matches=predicted.filter((tag,i)=>tag===gold[i]).length, entities=error==='1'?2:3;
   return table(['Token','Gold BIO','Predicted BIO'],tokens.map((word,i)=>[word,gold[i],predicted[i]]))
    +'<p><strong>Token accuracy '+pct(matches/9)+'</strong> ('+matches+'/9) · <strong>Exact entity F1 '+pct(entities/3)+'</strong> ('+entities+'/3 spans)</p>'
    +(error==='1'?'<p>The prediction spans “Nova” instead of “Nova Systems”. The other two entities are correct.</p>':'<p>All three entity spans match exactly.</p>');
 },
 bm25: ({tf,k,b,ratio})=>{
   const contribution=+tf*(+k+1)/(+tf + +k*(1-b+b*ratio));
   return '<p><strong>Term contribution (IDF = 1) · '+contribution.toFixed(3)+'</strong></p>';
 },
 rrf: ({k})=>{
   const lists=[['D1','D4','D2'],['D3','D1','D5']], fused={};
   lists.forEach(list=>list.forEach((d,i)=>{fused[d]=(fused[d]||0)+1/(+k+i+1);}));
   const rows=Object.entries(fused).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));
   return table(['Rank','BM25','Dense'],lists[0].map((d,i)=>[i+1,d,lists[1][i]]))+table(['Fused rank','Document','RRF score'],rows.map(([d,s],i)=>[i+1,d,s.toFixed(4)]));
 },
 threshold: ({threshold})=>{
   const predictions=scores.map(s=>s>=+threshold);
   const tp=predictions.filter((p,i)=>p && claims[i]).length, fp=predictions.filter((p,i)=>p && !claims[i]).length, fn=claims.filter((c,i)=>c && !predictions[i]).length;
   const precision=tp/(tp+fp)||0, recall=tp/(tp+fn)||0, f1=2*tp/(2*tp+fp+fn)||0;
   return '<p><strong>Precision '+pct(precision)+' · Recall '+pct(recall)+' · F1 '+pct(f1)+'</strong></p>'
      +table(['Example','Score','Actual','Predicted'],scores.map((s,i)=>[i+1,s.toFixed(2),claims[i]?'Claim':'Non-claim',predictions[i]?'Claim':'Non-claim']));
 },
 shift: ({mode})=>{
   const [label,source,target,explanation]=shifts[+mode];
   return '<p><strong>'+esc(label)+'</strong></p><p>'+esc(source)+'</p><p>'+esc(target)+'</p><p>'+esc(explanation)+'</p>';
 }
};

for(const demo of document.querySelectorAll('[data-demo]')){
 const controls=[...demo.querySelectorAll('select,input')];
 const snapshot=demo.querySelector('.snapshot');
 const result=demo.querySelector('.demo-result');
 const update=()=>{
   const state=Object.fromEntries(controls.map(c=>[c.name,c.value]));
   for(const c of controls){
     const value=c.type==='range'?c.value:c.options[c.selectedIndex].text;
     if(c.type==='range') c.nextElementSibling.textContent=value;
   }
   snapshot.textContent='Snapshot settings · '+controls.map(c=>c.dataset.label+' '+(c.type==='range'?c.value:c.options[c.selectedIndex].text)).join(' · ');
   result.innerHTML=demos[demo.dataset.demo](state);
 };
 demo.addEventListener('input',update);
 update();
}
for(const q of document.querySelectorAll('.quiz')){
 q.querySelector('.check').addEventListener('click',()=>{
   const selected=q.querySelector('input:checked');
   const status=q.querySelector('.quiz-status');
   if(!selected){status.textContent='Choose an answer first.';return;}
   status.textContent=selected.value==='b'?'Correct.':'Review the explanation below.';
   q.querySelector('details').open=true;
 });
}
for(const button of document.querySelectorAll('.copy')){
 button.addEventListener('click',async()=>{
   const code=button.closest('.example').querySelector('code').textContent;
   try{await navigator.clipboard.writeText(code);button.textContent='Copied';}
   catch{const range=document.createRange();range.selectNodeContents(button.closest('.example').querySelector('code'));const selection=getSelection();selection.removeAllRanges();selection.addRange(range);button.textContent='Selected';}
 });
}
