// Field keys and units must match the backend (app/main.py). Do not rename keys.
const SECTIONS = [
  {id:"required", title:"Required values", required:true, fields:[
    {key:"wbc",label:"WBC",unit:"×10⁹/L",req:true},
    {key:"hgb",label:"Hemoglobin",unit:"g/dL",req:true},
    {key:"plt",label:"Platelets",unit:"×10⁹/L",req:true}
  ]},
  {id:"diff", title:"White cell breakdown (optional)", fields:[
    {key:"neut",label:"Neutrophils (absolute)",unit:"×10⁹/L"},
    {key:"lymph",label:"Lymphocytes (absolute)",unit:"×10⁹/L"},
    {key:"mono",label:"Monocytes (absolute)",unit:"×10⁹/L"},
    {key:"eo",label:"Eosinophils (absolute)",unit:"×10⁹/L"},
    {key:"baso",label:"Basophils (absolute)",unit:"×10⁹/L"}
  ]},
  {id:"red", title:"Red cell indices (optional)", fields:[
    {key:"rbc",label:"RBC",unit:"×10¹²/L"},
    {key:"hct",label:"Hematocrit",unit:"%"},
    {key:"mcv",label:"MCV",unit:"fL"},
    {key:"mch",label:"MCH",unit:"pg"},
    {key:"mchc",label:"MCHC",unit:"g/dL"},
    {key:"rdw_cv",label:"RDW-CV",unit:"%"},
    {key:"rdw_sd",label:"RDW-SD",unit:"fL"}
  ]},
  {id:"plt-ind", title:"Platelet indices (optional)", fields:[
    {key:"mpv",label:"MPV",unit:"fL"},
    {key:"pdw",label:"PDW",unit:"fL"},
    {key:"plcr",label:"P-LCR",unit:"%"},
    {key:"pct",label:"PCT",unit:"%"}
  ]}
];
const ALL_FIELDS = SECTIONS.flatMap(section => section.fields);

// Backend label kept as-is in the API; shown in plain language here.
const FRIENDLY_NAMES = {
  "Values left blank (assumed typical)": "Fields you left blank (counted as typical)"
};

const EXAMPLES={
  normal:{wbc:6.8,neut:4.1,lymph:1.9,mono:.5,eo:.2,baso:.05,rbc:4.8,hgb:14,hct:42,mcv:90,mch:29.5,mchc:33,rdw_cv:13,rdw_sd:42,plt:250,mpv:10,pdw:12,plcr:25,pct:.25},
  abnormal:{wbc:1.6,neut:.7,lymph:.6,mono:.1,eo:.02,baso:.01,rbc:2.5,hgb:7.6,hct:23,mcv:93,mch:30,mchc:33,rdw_cv:15.5,rdw_sd:52,plt:36,mpv:11,pdw:14,plcr:30,pct:.04}
};

const form=document.getElementById("cbc-form");
const errorEl=document.getElementById("error");
const submitButton=document.getElementById("submit");
const submitLabel=submitButton.querySelector(".btn-label");

// ---- Build the form ----
SECTIONS.forEach(section=>{
  const grid=document.createElement("div");
  grid.className="field-grid";
  let container;
  if(section.required){
    container=document.createElement("div");
    container.className="form-section";
    const heading=document.createElement("h3");
    heading.className="section-title";
    heading.textContent=section.title;
    container.append(heading,grid);
  }else{
    container=document.createElement("details");
    container.className="group";
    container.id=`group-${section.id}`;
    const summary=document.createElement("summary");
    const title=document.createElement("span");
    title.className="group-title";
    title.textContent=section.title;
    const count=document.createElement("span");
    count.className="group-count";
    count.dataset.group=section.id;
    summary.append(title,count);
    container.append(summary,grid);
  }
  section.fields.forEach(field=>{
    const label=document.createElement("label");
    label.htmlFor=field.key;
    const name=document.createElement("span");
    name.className="field-name";
    name.textContent=field.label;
    if(field.req){
      const badge=document.createElement("span");
      badge.className="badge-req";
      badge.textContent="Required";
      name.append(" ",badge);
    }
    const unit=document.createElement("span");
    unit.className="unit";
    unit.id=`${field.key}-unit`;
    unit.textContent=field.unit;
    const input=document.createElement("input");
    input.type="number"; input.step="any"; input.min="0";
    input.id=field.key; input.name=field.key; input.inputMode="decimal";
    input.autocomplete="off";
    input.setAttribute("aria-describedby",unit.id);
    if(field.req){input.required=true;input.setAttribute("aria-required","true");}
    label.append(name,unit,input);
    grid.appendChild(label);
  });
  form.appendChild(container);
});

// ---- Helpers ----
function hideError(){errorEl.hidden=true}
function showError(message){errorEl.textContent=message;errorEl.hidden=false}

function updateCounts(){
  SECTIONS.filter(section=>!section.required).forEach(section=>{
    const filled=section.fields.filter(field=>document.getElementById(field.key).value.trim()!=="").length;
    const count=form.querySelector(`[data-group="${section.id}"]`);
    count.textContent=`${filled} of ${section.fields.length} entered`;
  });
}

function fill(values){
  ALL_FIELDS.forEach(field=>{
    const input=document.getElementById(field.key);
    input.value=values&&values[field.key]!==undefined?values[field.key]:"";
    input.classList.remove("invalid"); input.removeAttribute("aria-invalid");
  });
  // open any optional group that now holds a value, so it is visible
  SECTIONS.filter(section=>!section.required).forEach(section=>{
    const details=document.getElementById(`group-${section.id}`);
    details.open=section.fields.some(field=>document.getElementById(field.key).value.trim()!=="");
  });
  updateCounts();
  hideError();
}

function collect(){
  const body={};
  ALL_FIELDS.forEach(field=>{
    const raw=document.getElementById(field.key).value.trim();
    body[field.key]=raw===""?null:Number(raw);
  });
  return body;
}

function setBusy(busy){
  submitButton.disabled=busy;
  submitButton.setAttribute("aria-busy",String(busy));
  submitLabel.textContent=busy?"Checking values…":"Check results";
}

function renderDrivers(list,items,type){
  list.replaceChildren();
  if(!Array.isArray(items)||items.length===0){
    const li=document.createElement("li"); li.className="empty";
    li.textContent="No notable contributions returned."; list.appendChild(li); return;
  }
  const max=Math.max(...items.map(item=>Math.abs(Number(item.effect)||0)),.001);
  items.forEach(item=>{
    const li=document.createElement("li"); li.className=type;
    const line=document.createElement("div"); line.className="driver-line";
    const name=document.createElement("span"); name.className="driver-name";
    name.textContent=FRIENDLY_NAMES[item.feature]??item.feature??"Feature";
    const value=document.createElement("span"); value.className="driver-value";
    const shown=item.value===null||item.value===undefined?"":`${item.value} · `;
    const effect=Number(item.effect);
    value.textContent=`${shown}${effect>0?"+":""}${item.effect}`;
    line.append(name,value);
    const bar=document.createElement("div"); bar.className="bar";
    const fillBar=document.createElement("div"); fillBar.className="bar-fill";
    fillBar.style.width=`${Math.max(5,100*Math.abs(effect||0)/max)}%`;
    bar.appendChild(fillBar); li.append(line,bar); list.appendChild(li);
  });
}

function renderWarnings(warnings){
  const box=document.getElementById("warnings"); box.replaceChildren();
  const items=Array.isArray(warnings)?warnings:[];
  if(items.length===0) return;
  const title=document.createElement("h3"); title.className="warnings-title";
  title.textContent="Warnings to review";
  box.appendChild(title);
  items.forEach(message=>{
    const item=document.createElement("p"); item.className="warning";
    item.textContent=String(message); box.appendChild(item);
  });
}

function showResult(data){
  const flagged=Boolean(data.flagged);
  const verdict=document.getElementById("verdict");
  verdict.className=`verdict ${flagged?"flag":"noflag"}`;
  document.getElementById("verdict-title").textContent=flagged?"Flag raised":"No flag raised";
  document.getElementById("verdict-sub").textContent=flagged
    ? "The score is at or above the threshold. Further clinical assessment may be appropriate. This is not a diagnosis."
    : "The score is below the threshold. This does not rule out leukemia or replace clinical assessment.";

  const score=Math.min(1,Math.max(0,data.score));
  const threshold=Math.min(1,Math.max(0,data.threshold));
  document.getElementById("score-number").textContent=data.score.toFixed(3);
  document.getElementById("threshold-copy").textContent=`Flag threshold ${data.threshold.toFixed(3)}`;
  document.getElementById("tick").style.left=`${threshold*100}%`;
  document.getElementById("pin").style.left=`${score*100}%`;
  document.getElementById("meter").setAttribute("aria-label",
    `Screening score ${data.score.toFixed(3)} on a scale from 0 to 1. Flag threshold ${data.threshold.toFixed(3)}. ${flagged?"Flag raised.":"No flag raised."}`);

  renderWarnings(data.warnings);
  renderDrivers(document.getElementById("up"),data.raises_score,"up");
  renderDrivers(document.getElementById("down"),data.lowers_score,"down");

  const coverage=document.getElementById("coverage");
  coverage.textContent=typeof data.n_entered==="number"
    ? `Based on ${data.n_entered} of 19 values entered.` : "";

  document.getElementById("placeholder").hidden=true;
  const result=document.getElementById("result");
  result.hidden=false;
  document.getElementById("result-title").focus();
}

// ---- Actions ----
document.getElementById("ex-normal").addEventListener("click",()=>fill(EXAMPLES.normal));
document.getElementById("ex-abn").addEventListener("click",()=>fill(EXAMPLES.abnormal));
document.getElementById("clear").addEventListener("click",()=>{
  fill(null);
  form.querySelectorAll("details.group").forEach(details=>{details.open=false});
  document.getElementById("result").hidden=true;
  document.getElementById("placeholder").hidden=false;
  document.getElementById("wbc").focus();
});

async function runPrediction(){
  hideError();
  const body=collect();
  let invalid=null;
  ALL_FIELDS.forEach(field=>{
    const input=document.getElementById(field.key),value=body[field.key];
    const bad=(field.req&&value===null)||(value!==null&&(!Number.isFinite(value)||value<0));
    input.classList.toggle("invalid",bad);
    if(bad)input.setAttribute("aria-invalid","true");else input.removeAttribute("aria-invalid");
    if(bad&&!invalid)invalid=input;
  });
  if(invalid){
    showError("Please fill in the three required values (WBC, hemoglobin and platelets) and make sure every value is a non-negative number.");
    const group=invalid.closest("details");
    if(group)group.open=true;
    invalid.focus(); return;
  }
  setBusy(true);
  try{
    const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    let data;
    try{data=await response.json()}catch{throw new Error("The server returned an unreadable response. Check that the FastAPI app is running.");}
    if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:"Prediction failed. Please check the values and try again.");
    if(typeof data.score!=="number"||typeof data.threshold!=="number")throw new Error("The prediction response is missing the score or threshold.");
    showResult(data);
  }catch(error){showError(error.message||"Something went wrong. Please try again.");}
  finally{setBusy(false);}
}

form.addEventListener("submit",event=>{event.preventDefault(); runPrediction();});
form.addEventListener("input",event=>{
  if(event.target.matches("input")){
    event.target.classList.remove("invalid");
    event.target.removeAttribute("aria-invalid");
    hideError(); updateCounts();
  }
});
updateCounts();
