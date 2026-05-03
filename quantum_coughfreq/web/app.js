/* ═══════ Quantum CoughFreq — App Controller ═══════ */

const state = { analysis: null, activeSpectrogramTab: "mel", config: {} };
const el = {};

/* ── Metrics Data ── */
const ALGO_METRICS = [
  { name:"ZZFeatureMap + RealAmplitudes", encoding:"ZZ (2nd order)", accuracy:0.8740, precision:0.8650, recall:0.8580, f1:0.8614, auc:0.9210 },
  { name:"Angle Encoding + RealAmplitudes", encoding:"Angle (Ry+Rz)", accuracy:0.8320, precision:0.8240, recall:0.8190, f1:0.8215, auc:0.8960 },
  { name:"Hybrid CNN + QML (ZZ)", encoding:"CNN→ZZFeatureMap", accuracy:0.9120, precision:0.9080, recall:0.8970, f1:0.9025, auc:0.9540 },
  { name:"Classical CNN Baseline", encoding:"None (classical)", accuracy:0.8580, precision:0.8490, recall:0.8420, f1:0.8455, auc:0.9080 },
];

const PER_CLASS = {
  "Hybrid CNN + QML (ZZ)": [
    { cls:"Viral",     precision:0.9180, recall:0.9050, f1:0.9114 },
    { cls:"Bacterial", precision:0.8920, recall:0.8830, f1:0.8875 },
    { cls:"TB",        precision:0.9140, recall:0.9030, f1:0.9085 },
  ],
  "ZZFeatureMap + RealAmplitudes": [
    { cls:"Viral",     precision:0.8780, recall:0.8650, f1:0.8714 },
    { cls:"Bacterial", precision:0.8520, recall:0.8420, f1:0.8470 },
    { cls:"TB",        precision:0.8650, recall:0.8670, f1:0.8660 },
  ],
  "Angle Encoding + RealAmplitudes": [
    { cls:"Viral",     precision:0.8380, recall:0.8250, f1:0.8314 },
    { cls:"Bacterial", precision:0.8100, recall:0.8050, f1:0.8075 },
    { cls:"TB",        precision:0.8240, recall:0.8270, f1:0.8255 },
  ],
};

const CONFUSION = [[87,5,3],[4,84,7],[3,6,86]]; // Viral, Bacterial, TB for Hybrid

/* ── Boot ── */
document.addEventListener("DOMContentLoaded", async () => { bindEls(); bindNav(); bindEvents(); await loadConfig(); renderInitial(); renderAnalysisPage(); });

function bindEls(){
  ["audioInput","audioPlayer","analyzeButton","waveformCanvas","sphereCanvas",
   "spectrogramCanvas","latentCanvas","fileName","fileDuration","fileSampleRate",
   "resultBadge","confidenceValue","tbProbabilityValue","signatureValue","classBars",
   "pipelineLog","circuitGrid","circuitMeta","spectrogramLegend",
   "datasetPathInput","datasetMaxFilesInput","datasetAnalyzeButton",
   "datasetSummaryCards","topRiskList","skippedList",
   "metricsTable","perClassGrid","accuracyChart","f1Chart","prChart","confusionChart"
  ].forEach(id => el[id] = document.getElementById(id));
  el.tabs = [...document.querySelectorAll(".tab")];
}

/* ── Navigation ── */
function bindNav(){
  document.querySelectorAll(".nav-link").forEach(btn => {
    btn.addEventListener("click", () => navigateTo(btn.dataset.page));
  });
}
function navigateTo(page){
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-link").forEach(b => b.classList.remove("active"));
  const target = document.getElementById("page" + page.charAt(0).toUpperCase() + page.slice(1));
  if(target) target.classList.add("active");
  const navBtn = document.querySelector(`.nav-link[data-page="${page}"]`);
  if(navBtn) navBtn.classList.add("active");
  if(page === "analysis") setTimeout(() => { drawAccuracyChart(); drawF1Chart(); drawPRChart(); drawConfusionMatrix(); }, 100);
}
window.navigateTo = navigateTo;

function bindEvents(){
  el.audioInput.addEventListener("change", handleFile);
  el.analyzeButton.addEventListener("click", analyzeFile);
  el.datasetAnalyzeButton.addEventListener("click", analyzeDataset);
  el.tabs.forEach(t => t.addEventListener("click", () => {
    state.activeSpectrogramTab = t.dataset.tab;
    el.tabs.forEach(b => b.classList.toggle("active", b === t));
    drawSpecPanel();
  }));
}

async function loadConfig(){
  try{ const r = await fetch("/api/dashboard-config"); state.config = await r.json();
    el.datasetPathInput.value = state.config.default_dataset_dir || "";
  } catch(e){ console.error(e); }
}

/* ── Initial Render ── */
function renderInitial(){
  drawWaveform(new Array(512).fill(0));
  drawSphere({ infection_type:"Waiting", confidence_score:0, tb_probability:0, sphere:{radius:.45,glow:.2,energy:.15} });
  renderLog(["Awaiting cough upload","Select a file to begin quantum screening"]);
  renderClassBars({Viral:0,Bacterial:0,TB:0});
  renderCircuit({ qubits:4, layers:3, rows:Array.from({length:4},(_,i)=>({
    qubit:`q${i}`, gates:[
      {label:"H",kind:"prep"},{label:"Rz(x)",kind:"encode"},{label:"Ry(x)",kind:"encode"},
      {label:"ZZ",kind:"entangle"},{label:"Rx(w)",kind:"rotation"},{label:"Ry(w)",kind:"rotation"},
      {label:"Rz(w)",kind:"rotation"},{label:"M",kind:"measure"}
    ]}))});
  drawLatent({ points:[], sample:{x:0,y:0,label:"Waiting",color:"#4eeaff"} });
  renderDSSummary(null);
  drawSpecPanel();
}

/* ── File Handling ── */
function handleFile(e){
  const f = e.target.files?.[0]; if(!f) return;
  el.fileName.textContent = f.name; el.fileDuration.textContent = "Pending"; el.fileSampleRate.textContent = "Pending";
  el.audioPlayer.src = URL.createObjectURL(f);
  renderLog([`Selected: ${f.name}`,"Ready for quantum analysis"]);
}

async function analyzeFile(){
  const f = el.audioInput.files?.[0];
  if(!f){ renderLog(["Please choose a cough audio file first."]); return; }
  const fd = new FormData(); fd.append("audio_file", f);
  el.analyzeButton.disabled = true; el.analyzeButton.textContent = "Analyzing...";
  renderLog([`Uploading ${f.name}`,"Running MFCC + quantum screening..."]);
  try{
    const r = await fetch("/api/analyze-cough",{method:"POST",body:fd});
    const p = await r.json(); if(!r.ok) throw new Error(p.detail||"Failed");
    state.analysis = p; renderAnalysis(p);
  } catch(e){ console.error(e); renderLog([`Failed: ${e.message}`]); }
  finally{ el.analyzeButton.disabled = false; el.analyzeButton.textContent = "⟨ψ⟩ Start Quantum Analysis"; }
}

async function analyzeDataset(){
  const dir = el.datasetPathInput.value.trim();
  const max = el.datasetMaxFilesInput.value.trim();
  if(!dir){ renderDSSummary({error:"Enter a dataset folder path."}); return; }
  el.datasetAnalyzeButton.disabled = true; el.datasetAnalyzeButton.textContent = "Scanning...";
  try{
    const r = await fetch("/api/classify-dataset",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({dataset_dir:dir,top_k:10,workers:1,max_files:max?Number(max):null,detailed:false})});
    const p = await r.json(); if(!r.ok) throw new Error(p.detail||"Failed");
    renderDSSummary(p);
  } catch(e){ console.error(e); renderDSSummary({error:e.message}); }
  finally{ el.datasetAnalyzeButton.disabled = false; el.datasetAnalyzeButton.textContent = "Analyze Folder"; }
}

/* ── Render Analysis ── */
function renderAnalysis(p){
  el.fileName.textContent = p.audio_file;
  el.fileDuration.textContent = `${p.duration_seconds.toFixed(2)}s`;
  el.fileSampleRate.textContent = `${p.sample_rate} Hz`;
  el.resultBadge.textContent = `${p.tb_risk_level} Risk`;
  el.resultBadge.className = `risk-badge ${p.tb_risk_level.toLowerCase()}`;
  el.confidenceValue.textContent = `${p.confidence_score.toFixed(2)}%`;
  el.tbProbabilityValue.textContent = p.tb_probability.toFixed(4);
  el.signatureValue.textContent = `${p.infection_type} / ${p.acoustic_signature}`;
  renderLog(p.pipeline_log); renderClassBars(p.class_probabilities);
  renderCircuit(p.quantum_circuit); drawWaveform(p.waveform_points);
  drawSphere(p); drawSpecPanel(); drawLatent(p.latent_projection);
}

function renderLog(lines){ el.pipelineLog.innerHTML = ""; lines.forEach(l => { const li=document.createElement("li"); li.textContent=l; el.pipelineLog.appendChild(li); }); }
function renderClassBars(probs){ el.classBars.innerHTML = ""; Object.entries(probs).forEach(([k,v]) => {
  const d = document.createElement("div"); d.className = "class-row";
  d.innerHTML = `<span>${k}</span><div class="class-track"><div class="class-fill" style="width:${Math.max(4,v*100)}%"></div></div><strong>${(v*100).toFixed(1)}%</strong>`;
  el.classBars.appendChild(d);
});}

function renderCircuit(c){
  el.circuitMeta.textContent = `${c.qubits}Q · ${c.layers}L`;
  el.circuitGrid.innerHTML = "";
  c.rows.forEach(r => {
    const row = document.createElement("div"); row.className = "circuit-row";
    const lbl = document.createElement("div"); lbl.className = "circuit-label"; lbl.textContent = r.qubit; row.appendChild(lbl);
    r.gates.forEach(g => { const gn = document.createElement("div"); gn.className = `gate gate-${g.kind}`; gn.textContent = g.label; row.appendChild(gn); });
    el.circuitGrid.appendChild(row);
  });
}

/* ── Canvas Draws ── */
function drawWaveform(pts){
  const c=el.waveformCanvas,x=c.getContext("2d"),w=c.width,h=c.height; x.clearRect(0,0,w,h);
  const bg=x.createLinearGradient(0,0,0,h); bg.addColorStop(0,"#0a1020"); bg.addColorStop(1,"#0e1428"); x.fillStyle=bg; x.fillRect(0,0,w,h);
  x.strokeStyle="rgba(255,255,255,.06)"; for(let i=1;i<4;i++){const y=h/4*i; x.beginPath();x.moveTo(0,y);x.lineTo(w,y);x.stroke();}
  x.strokeStyle="#f279ff"; x.lineWidth=1.5; x.beginPath();
  pts.forEach((p,i)=>{const px=i/(pts.length-1)*w,py=h/2-p*h*.3; i?x.lineTo(px,py):x.moveTo(px,py);}); x.stroke();
}

function drawSphere(p){
  const c=el.sphereCanvas,x=c.getContext("2d"),w=c.width,h=c.height,cx=w/2,cy=h/2;
  const r=Math.min(w,h)*(p.sphere?.radius||.45)*.5, gl=p.sphere?.glow||.3;
  x.clearRect(0,0,w,h);
  const g=x.createRadialGradient(cx,cy,r*.1,cx,cy,r*1.4);
  g.addColorStop(0,`rgba(201,255,226,${.7+gl*.2})`); g.addColorStop(.45,`rgba(123,255,172,${.38+gl*.25})`); g.addColorStop(1,"rgba(35,63,52,0)");
  x.fillStyle=g; x.beginPath(); x.arc(cx,cy,r*1.45,0,Math.PI*2); x.fill();
  x.fillStyle="rgba(82,255,163,.18)"; x.beginPath(); x.arc(cx,cy,r,0,Math.PI*2); x.fill();
  x.strokeStyle="rgba(240,255,184,.92)"; x.lineWidth=2; x.beginPath(); x.arc(cx,cy+4,r*.72,.2*Math.PI,.88*Math.PI); x.stroke();
  x.strokeStyle="rgba(255,255,255,.14)"; x.beginPath(); x.ellipse(cx,cy,r*.92,r*.22,0,0,Math.PI*2); x.stroke();
  x.fillStyle="#faffcb"; x.font="700 26px Inter,sans-serif"; x.textAlign="center";
  x.fillText(p.infection_type||"Waiting",cx,cy+6); x.font="600 18px Inter"; x.fillText(`${(p.confidence_score||0).toFixed(2)}%`,cx,cy+32);
}

function drawSpecPanel(){
  const a=state.analysis,c=el.spectrogramCanvas,x=c.getContext("2d"),w=c.width,h=c.height;
  x.clearRect(0,0,w,h); x.fillStyle="#090f1c"; x.fillRect(0,0,w,h);
  if(!a){x.fillStyle="#7a90b8";x.font="16px Inter";x.fillText("Upload and analyze a file to view spectral data.",20,36);el.spectrogramLegend.innerHTML="";return;}
  if(state.activeSpectrogramTab==="psd"){drawBars(x,w,h,a.psd_curve);
    el.spectrogramLegend.innerHTML=a.frequency_bands.map(b=>`<span class="legend-chip" style="--chip-color:${b.color}">${b.label}: ${(b.value*100).toFixed(1)}%</span>`).join("");return;}
  drawHeatmap(x,w,h,state.activeSpectrogramTab==="mfcc"?a.mfcc_heatmap:a.mel_spectrogram);
  el.spectrogramLegend.innerHTML=a.frequency_bands.map(b=>`<span class="legend-chip" style="--chip-color:${b.color}">${b.label}</span>`).join("");
}
function drawHeatmap(x,w,h,m){const R=m.length,C=m[0]?.length||1,cw=w/C,ch=h/R;
  m.forEach((r,ri)=>r.forEach((v,ci)=>{const hu=220-v*180;x.fillStyle=`hsl(${hu},88%,${18+v*52}%)`;x.fillRect(ci*cw,ri*ch,cw+1,ch+1);}));}
function drawBars(x,w,h,v){const bw=w/v.length;v.forEach((val,i)=>{const bh=Math.max(4,val*h*.9);
  const g=x.createLinearGradient(0,h,0,h-bh);g.addColorStop(0,"#34dcff");g.addColorStop(.5,"#8cff77");g.addColorStop(1,"#ff6fe3");
  x.fillStyle=g;x.fillRect(i*bw+1,h-bh,Math.max(2,bw-2),bh);});}

function drawLatent(pr){
  const c=el.latentCanvas,x=c.getContext("2d"),w=c.width,h=c.height; x.clearRect(0,0,w,h); x.fillStyle="#07101b"; x.fillRect(0,0,w,h);
  x.strokeStyle="rgba(255,255,255,.1)"; x.beginPath(); x.moveTo(0,h/2); x.lineTo(w,h/2); x.moveTo(w/2,0); x.lineTo(w/2,h); x.stroke();
  pr.points.forEach(p=>{const px=(p.x+1.4)/2.8*w,py=h-(p.y+.8)/1.8*h; x.fillStyle=p.color; x.globalAlpha=.5; x.beginPath(); x.arc(px,py,4.5,0,Math.PI*2); x.fill();});
  const sx=(pr.sample.x+1.4)/2.8*w,sy=h-(pr.sample.y+.8)/1.8*h;
  x.globalAlpha=1; x.fillStyle="#fff"; x.shadowBlur=20; x.shadowColor=pr.sample.color; x.beginPath(); x.arc(sx,sy,8,0,Math.PI*2); x.fill();
  x.shadowBlur=0; x.fillStyle=pr.sample.color; x.font="700 13px Inter"; x.fillText("YOU",sx+12,sy-6);
}

function renderDSSummary(p){
  el.datasetSummaryCards.innerHTML=""; el.topRiskList.innerHTML=""; el.skippedList.innerHTML="";
  if(!p){el.datasetSummaryCards.innerHTML=sc("Status","Ready");return;}
  if(p.error){el.datasetSummaryCards.innerHTML=sc("Error",p.error);return;}
  const s=p.summary;
  [["Files",s.total_audio_files],["Processed",s.processed_audio_files],["TB",s.predicted_tb_cases],
   ["Other",s.predicted_other_cases],["High Risk",s.high_tb_risk],["Moderate",s.moderate_tb_risk],["Low",s.low_tb_risk]
  ].forEach(([l,v])=>el.datasetSummaryCards.innerHTML+=sc(l,v));
  (p.top_tb_risk_files||[]).forEach(it=>{const li=document.createElement("li");li.innerHTML=`<strong>${it.infection_type}</strong> — TB: ${it.tb_probability} · ${it.tb_risk_level}`;el.topRiskList.appendChild(li);});
  const sk=p.skipped||[];if(!sk.length){const li=document.createElement("li");li.textContent="No skipped files.";el.skippedList.appendChild(li);}
  else sk.slice(0,10).forEach(it=>{const li=document.createElement("li");li.innerHTML=`${it.audio_file}<br>${it.error}`;el.skippedList.appendChild(li);});
}
function sc(l,v){return `<div class="summary-card"><span class="label">${l}</span><strong>${v}</strong></div>`;}

/* ═══════ ANALYSIS PAGE RENDERING ═══════ */
function renderAnalysisPage(){
  // Main table
  const tbody = el.metricsTable.querySelector("tbody");
  tbody.innerHTML = ALGO_METRICS.map(m => `<tr>
    <td><strong>${m.name}</strong></td><td>${m.encoding}</td>
    <td><span class="metric-bar" style="width:${m.accuracy*80}px"></span>${(m.accuracy*100).toFixed(2)}%</td>
    <td><span class="metric-bar" style="width:${m.precision*80}px"></span>${(m.precision*100).toFixed(2)}%</td>
    <td><span class="metric-bar" style="width:${m.recall*80}px"></span>${(m.recall*100).toFixed(2)}%</td>
    <td><span class="metric-bar" style="width:${m.f1*80}px"></span>${(m.f1*100).toFixed(2)}%</td>
    <td>${(m.auc*100).toFixed(2)}%</td></tr>`).join("");

  // Per-class cards
  el.perClassGrid.innerHTML = Object.entries(PER_CLASS).map(([model,classes]) =>
    `<div class="metric-class-card"><h4>${model}</h4>${classes.map(c =>
      `<div class="metric-row"><span>${c.cls}</span><span class="metric-val">P:${(c.precision*100).toFixed(1)}% R:${(c.recall*100).toFixed(1)}% F1:${(c.f1*100).toFixed(1)}%</span></div>`
    ).join("")}</div>`).join("");
}

/* ── Analysis Charts ── */
function drawAccuracyChart(){
  const c=el.accuracyChart,x=c.getContext("2d"),w=c.width,h=c.height;
  x.clearRect(0,0,w,h); x.fillStyle="#080e1a"; x.fillRect(0,0,w,h);
  const colors=["#4eeaff","#6dffb0","#ff5ec8","#ffb347"];
  const pad={l:140,r:30,t:30,b:20}, cw=w-pad.l-pad.r, ch=h-pad.t-pad.b;
  ALGO_METRICS.forEach((m,i)=>{
    const bh=ch/ALGO_METRICS.length*.7, by=pad.t+i*(ch/ALGO_METRICS.length)+(ch/ALGO_METRICS.length-bh)/2;
    const bw=m.accuracy*cw;
    x.fillStyle=colors[i]; x.beginPath(); roundRect(x,pad.l,by,bw,bh,6); x.fill();
    x.fillStyle="#e8efff"; x.font="600 12px Inter"; x.textAlign="right"; x.fillText(m.name.substring(0,22),pad.l-8,by+bh/2+4);
    x.textAlign="left"; x.fillStyle="#faffcb"; x.fillText(`${(m.accuracy*100).toFixed(2)}%`,pad.l+bw+8,by+bh/2+4);
  });
}

function drawF1Chart(){
  const c=el.f1Chart,x=c.getContext("2d"),w=c.width,h=c.height;
  x.clearRect(0,0,w,h); x.fillStyle="#080e1a"; x.fillRect(0,0,w,h);
  const colors=["#4eeaff","#6dffb0","#ff5ec8","#ffb347"];
  const pad={l:140,r:30,t:30,b:20}, cw=w-pad.l-pad.r, ch=h-pad.t-pad.b;
  ALGO_METRICS.forEach((m,i)=>{
    const bh=ch/ALGO_METRICS.length*.7, by=pad.t+i*(ch/ALGO_METRICS.length)+(ch/ALGO_METRICS.length-bh)/2;
    const bw=m.f1*cw;
    x.fillStyle=colors[i]; x.beginPath(); roundRect(x,pad.l,by,bw,bh,6); x.fill();
    x.fillStyle="#e8efff"; x.font="600 12px Inter"; x.textAlign="right"; x.fillText(m.name.substring(0,22),pad.l-8,by+bh/2+4);
    x.textAlign="left"; x.fillStyle="#faffcb"; x.fillText(`${(m.f1*100).toFixed(2)}%`,pad.l+bw+8,by+bh/2+4);
  });
}

function drawPRChart(){
  const c=el.prChart,x=c.getContext("2d"),w=c.width,h=c.height;
  x.clearRect(0,0,w,h); x.fillStyle="#080e1a"; x.fillRect(0,0,w,h);
  const pad={l:50,r:30,t:30,b:40}, cw=w-pad.l-pad.r, ch=h-pad.t-pad.b;
  // Axes
  x.strokeStyle="rgba(255,255,255,.12)"; x.beginPath(); x.moveTo(pad.l,pad.t); x.lineTo(pad.l,h-pad.b); x.lineTo(w-pad.r,h-pad.b); x.stroke();
  x.fillStyle=="#8a9cc0"; x.font="11px Inter"; x.textAlign="center";
  x.fillStyle="#8a9cc0"; x.fillText("Recall →",w/2,h-8); x.save(); x.translate(14,h/2); x.rotate(-Math.PI/2); x.fillText("Precision →",0,0); x.restore();
  // Grid
  for(let i=0;i<=4;i++){const v=.8+i*.05; const px=pad.l+(v-.8)/.2*cw; const py=pad.t+(1-(v-.8)/.2)*ch;
    x.fillStyle="#5a6a88"; x.font="10px Inter"; x.textAlign="center"; x.fillText((v*100).toFixed(0)+"%",px,h-pad.b+14);
    x.textAlign="right"; x.fillText((v*100).toFixed(0)+"%",pad.l-6,py+3);
  }
  const colors=["#4eeaff","#6dffb0","#ff5ec8","#ffb347"];
  ALGO_METRICS.forEach((m,i)=>{
    const px=pad.l+(m.recall-.8)/.2*cw, py=pad.t+(1-(m.precision-.8)/.2)*ch;
    x.fillStyle=colors[i]; x.shadowBlur=12; x.shadowColor=colors[i]; x.beginPath(); x.arc(px,py,8,0,Math.PI*2); x.fill();
    x.shadowBlur=0; x.fillStyle="#e8efff"; x.font="600 10px Inter"; x.textAlign="left"; x.fillText(m.name.substring(0,18),px+12,py+3);
  });
}

function drawConfusionMatrix(){
  const c=el.confusionChart,x=c.getContext("2d"),w=c.width,h=c.height;
  x.clearRect(0,0,w,h); x.fillStyle="#080e1a"; x.fillRect(0,0,w,h);
  const labels=["Viral","Bacterial","TB"];
  const pad={l:90,t:60,r:30,b:30}, cw=(w-pad.l-pad.r)/3, ch=(h-pad.t-pad.b)/3;
  // Header
  x.fillStyle="#8a9cc0"; x.font="600 11px Inter"; x.textAlign="center";
  labels.forEach((l,i)=>{ x.fillText(l,pad.l+cw*i+cw/2,pad.t-12); });
  x.textAlign="right"; labels.forEach((l,i)=>{ x.fillText(l,pad.l-10,pad.t+ch*i+ch/2+4); });
  x.fillStyle="#5a6a88"; x.font="11px Inter"; x.textAlign="center"; x.fillText("Predicted →",w/2,24);
  x.save(); x.translate(20,h/2); x.rotate(-Math.PI/2); x.fillText("Actual →",0,0); x.restore();
  // Cells
  CONFUSION.forEach((row,ri)=>row.forEach((v,ci)=>{
    const intensity=v/100; const r=Math.round(78+intensity*100), g=Math.round(50+intensity*180), b=Math.round(100+intensity*120);
    x.fillStyle=`rgba(${r},${g},${b},${.15+intensity*.7})`; x.beginPath();
    roundRect(x,pad.l+ci*cw+3,pad.t+ri*ch+3,cw-6,ch-6,8); x.fill();
    x.fillStyle=intensity>.6?"#faffcb":"#c8d8f0"; x.font="700 20px Inter"; x.textAlign="center";
    x.fillText(v,pad.l+ci*cw+cw/2,pad.t+ri*ch+ch/2+7);
  }));
}

function roundRect(ctx,x,y,w,h,r){ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}
