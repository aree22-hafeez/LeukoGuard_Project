const GROUPS = [
  {title:"White blood cells",fields:[
    {key:"wbc",label:"WBC",unit:"×10⁹/L",req:true},
    {key:"neut",label:"Neutrophils (abs.)",unit:"×10⁹/L"},
    {key:"lymph",label:"Lymphocytes (abs.)",unit:"×10⁹/L"},
    {key:"mono",label:"Monocytes (abs.)",unit:"×10⁹/L"},
    {key:"eo",label:"Eosinophils (abs.)",unit:"×10⁹/L"},
    {key:"baso",label:"Basophils (abs.)",unit:"×10⁹/L"}
  ]},
  {title:"Red blood cells",fields:[
    {key:"hgb",label:"Hemoglobin",unit:"g/dL",req:true},
    {key:"rbc",label:"RBC",unit:"×10¹²/L"},
    {key:"hct",label:"Hematocrit",unit:"%"},
    {key:"mcv",label:"MCV",unit:"fL"},
    {key:"mch",label:"MCH",unit:"pg"},
    {key:"mchc",label:"MCHC",unit:"g/dL"},
    {key:"rdw_cv",label:"RDW-CV",unit:"%"},
    {key:"rdw_sd",label:"RDW-SD",unit:"fL"}
  ]},
  {title:"Platelets",fields:[
    {key:"plt",label:"Platelets",unit:"×10⁹/L",req:true},
    {key:"mpv",label:"MPV",unit:"fL"},
    {key:"pdw",label:"PDW",unit:"fL"},
    {key:"plcr",label:"P-LCR",unit:"%"},
    {key:"pct",label:"PCT",unit:"%"}
  ]}
];

const EXAMPLES={
  normal:{wbc:6.8,neut:4.1,lymph:1.9,mono:.5,eo:.2,baso:.05,rbc:4.8,hgb:14,hct:42,mcv:90,mch:29.5,mchc:33,rdw_cv:13,rdw_sd:42,plt:250,mpv:10,pdw:12,plcr:25,pct:.25},
  abnormal:{wbc:1.6,neut:.7,lymph:.6,mono:.1,eo:.02,baso:.01,rbc:2.5,hgb:7.6,hct:23,mcv:93,mch:30,mchc:33,rdw_cv:15.5,rdw_sd:52,plt:36,mpv:11,pdw:14,plcr:30,pct:.04}
};

const form=document.getElementById("cbc-form");
const errorEl=document.getElementById("error");

GROUPS.forEach(group=>{
  const fieldset=document.createElement("fieldset");
  const legend=document.createElement("legend");
  legend.textContent=group.title;
  fieldset.appendChild(legend);
  const grid=document.createElement("div");
  grid.className="grid";
  group.fields.forEach(field=>{
    const label=document.createElement("label");
    label.htmlFor=field.key;
    label.append(document.createTextNode(field.label+(field.req?" *":"")));
    const unit=document.createElement("small");
    unit.textContent=field.unit;
    label.appendChild(unit);
    const input=document.createElement("input");
    input.type="number"; input.step="any"; input.min="0";
    input.id=field.key; input.name=field.key; input.inputMode="decimal";
    input.autocomplete="off";
    label.appendChild(input); grid.appendChild(label);
  });
  fieldset.appendChild(grid); form.appendChild(fieldset);
});

function hideError(){errorEl.hidden=true}
function fill(values){
  GROUPS.forEach(group=>group.fields.forEach(field=>{
    const input=document.getElementById(field.key);
    input.value=values&&values[field.key]!==undefined?values[field.key]:"";
    input.classList.remove("invalid"); input.removeAttribute("aria-invalid");
  }));
  hideError();
}
document.getElementById("ex-normal").addEventListener("click",()=>fill(EXAMPLES.normal));
document.getElementById("ex-abn").addEventListener("click",()=>fill(EXAMPLES.abnormal));
document.getElementById("clear").addEventListener("click",()=>{
  fill(null); document.getElementById("result").hidden=true;
  document.getElementById("placeholder").hidden=false;
});
function collect(){
  const body={};
  GROUPS.forEach(group=>group.fields.forEach(field=>{
    const raw=document.getElementById(field.key).value.trim();
    body[field.key]=raw===""?null:Number(raw);
  }));
  return body;
}
function showError(message){errorEl.textContent=message;errorEl.hidden=false}
function renderDrivers(list,items,type){
  list.replaceChildren();
  if(!Array.isArray(items)||items.length===0){
    const li=document.createElement("li"); li.textContent="No notable contributions returned."; list.appendChild(li); return;
  }
  const max=Math.max(...items.map(item=>Math.abs(Number(item.effect)||0)),.001);
  items.forEach(item=>{
    const li=document.createElement("li"); li.className=type;
    const line=document.createElement("div"); line.className="driver-line";
    const name=document.createElement("span"); name.className="driver-name"; name.textContent=item.feature??"Feature";
    const value=document.createElement("span"); value.className="val";
    const rawValue=item.value===null||item.value===undefined?"":` · ${item.value}`;
    value.textContent=`${rawValue} (${Number(item.effect)>0?"+":""}${item.effect})`;
    line.append(name,value);
    const bar=document.createElement("div"); bar.className="bar";
    const fillBar=document.createElement("div"); fillBar.className="bar-fill";
    fillBar.style.width=`${Math.max(5,100*Math.abs(Number(item.effect)||0)/max)}%`;
    bar.appendChild(fillBar); li.append(line,bar); list.appendChild(li);
  });
}
function renderWarnings(warnings){
  const box=document.getElementById("warnings"); box.replaceChildren();
  (Array.isArray(warnings)?warnings:[]).forEach(message=>{
    const item=document.createElement("div"); item.className="warning";
    item.textContent=String(message); box.appendChild(item);
  });
}

document.getElementById("submit").addEventListener("click",async()=>{
  const button=document.getElementById("submit"); hideError();
  const body=collect(); let invalid=false;
  GROUPS.forEach(group=>group.fields.forEach(field=>{
    const input=document.getElementById(field.key),value=body[field.key];
    const bad=(field.req&&value===null)||(value!==null&&(!Number.isFinite(value)||value<0));
    input.classList.toggle("invalid",bad);
    if(bad)input.setAttribute("aria-invalid","true");else input.removeAttribute("aria-invalid");
    if(bad)invalid=true;
  }));
  if(invalid){
    showError("Please complete the required fields (*) and enter valid non-negative numbers.");
    document.querySelector("#cbc-form input.invalid")?.focus(); return;
  }
  button.disabled=true; button.innerHTML="Checking values…";
  try{
    const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    let data;
    try{data=await response.json()}catch{throw new Error("The server returned an unreadable response. Check that the FastAPI app is running.");}
    if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:"Prediction failed. Please check the values and try again.");
    if(typeof data.score!=="number"||typeof data.threshold!=="number")throw new Error("The prediction response is missing the score or threshold.");
    const verdict=document.getElementById("verdict");
    verdict.className=`verdict ${data.flagged?"flag":"noflag"}`; verdict.replaceChildren();
    const title=document.createElement("span"); title.className="big";
    title.textContent=data.flagged?"Flagged: elevated risk":"No Screening Flag Raised";
    const sub=document.createElement("span"); sub.className="sub";
    sub.textContent=data.flagged?"Further clinical evaluation may be appropriate; this is not a diagnosis.":"This result does not rule out leukemia or replace clinical assessment.";
    verdict.append(title,sub);
    document.getElementById("score-number").textContent=data.score.toFixed(3);
    document.getElementById("threshold-copy").textContent=`Flag threshold ${data.threshold.toFixed(3)}`;
    document.getElementById("pin").style.left=`${Math.min(100,Math.max(0,data.score*100))}%`;
    document.getElementById("tick").style.left=`${Math.min(100,Math.max(0,data.threshold*100))}%`;
    const badge=document.getElementById("result-badge");
    badge.textContent=data.flagged?"FLAGGED":"NOT FLAGGED";
    badge.style.color=data.flagged?"#9c1148":"#1f7a4d";
    badge.style.background=data.flagged?"#fdeaf0":"#e8f6ee";
    renderWarnings(data.warnings);
    renderDrivers(document.getElementById("up"),data.raises_score,"up");
    renderDrivers(document.getElementById("down"),data.lowers_score,"down");
    document.getElementById("placeholder").hidden=true;
    document.getElementById("result").hidden=false;
    if(window.innerWidth<950)document.getElementById("result").scrollIntoView({behavior:"smooth",block:"start"});
  }catch(error){showError(error.message||"Something went wrong. Please try again.");}
  finally{button.disabled=false;button.innerHTML='Check Results <span>→</span>';}
});
form.addEventListener("input",event=>{
  if(event.target.matches("input")){event.target.classList.remove("invalid");event.target.removeAttribute("aria-invalid");hideError();}
});