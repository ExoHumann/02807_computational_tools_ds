import {scorePlot} from "./scatter.js";
import "./audio.js";
const get = id => document.getElementById(id);
const number = new Intl.NumberFormat("en");
const optionName = key => ({tiktok:"TikTok-like",queries_tiktok:"TikTok-like",queries:"Standard",standard:"Standard",lsh:"LSH",ivf:"IVF",exact:"Exact cosine"})[String(key).toLowerCase()] || String(key).replace(/^./, c=>c.toUpperCase());
const format = value => Number.isFinite(value) ? value.toFixed(3) : "—";
const element = (tag, text, className) => {
  const el=document.createElement(tag);
  if (text != null) el.textContent=text;
  if (className) el.className=className;
  return el;
};
function populate(id, values) {
  for (const value of [...new Set(values)]) get(id).add(new Option(optionName(value),value));
}

function benchmarks(methods) {
  populate("catalogue", methods.map(row=>row.catalogue));
  populate("family", methods.map(row=>row.family));
  const large=methods.find(row=>row.catalogue.toLowerCase()==="large");
  if(large) get("catalogue").value=large.catalogue;
  function update() {
    const metric=get("metric").value;
    const items=methods.filter(row=>row.catalogue===get("catalogue").value && row.family===get("family").value);
    const val=row=>metric==="accuracy" ? row.correct/row.total*100 : row[metric];
    const max=metric==="accuracy" ? 100 : Math.max(...items.map(val));
    const chart=get("chart");
    chart.replaceChildren();
    for(const row of items){
      const v=val(row);
      const line=element("div",null,"barrow");
      const rail=element("div",null,"bar-track");
      const bar=element("div",null,"bar-fill");
      bar.style.width=(100*v/max)+"%";
      rail.append(bar);
      const text=metric==="accuracy" ? v.toFixed(2)+"%" : metric==="ms" ? v.toFixed(3)+" ms" : number.format(Math.round(v))+"/s";
      line.append(element("b",optionName(row.method)),rail,element("strong",text));
      chart.append(line);
    }
    const notes={accuracy:"Top-1 compares the highest-ranked archive ID with the source. "+number.format(items[0]?.total || 0)+" clips per method; axis 0–100%.",ms:"Warm index and fingerprints on one RTX 4090. Median lookup in milliseconds; shorter is faster. Audio encoding excluded.",qps:"Warm GPU throughput at batch size 128. Longer bars mean more query fingerprints searched per second."};
    get("chart-note").textContent=notes[metric];
  }
  ["catalogue","family","metric"].forEach(id=>get(id).addEventListener("change",update));
  update();
}

function queryExplorer(queries) {
  const outcome=q=>!q.accepted ? "Abstained" : q.correct ? "Correct match" : "Wrong match";
  const label=q=>q.title || "Track "+q.track;
  let page=0, selected, filtered=[];
  const pageSize=6;
  const plot = scorePlot(get("score-plot"), q=>{
    selected=q;page=Math.floor(filtered.indexOf(q)/pageSize);show();
  });
  populate("genre", queries.map(q=>q.genre).filter(Boolean).sort());
  function select(q) {
    selected=q;plot(filtered, q);
    for(const button of get("rows").children)
      button.setAttribute("aria-pressed",String(button.dataset.id===String(q.id)));
    const inspector=get("inspect");
    inspector.replaceChildren(element("p","QUERY "+q.id,"eyebrow"),element("h3",label(q)),element("p",[q.artist,q.genre].filter(Boolean).join(" · ")));
    const values=[
      ["Source track","FMA "+q.track+(q.known ? " · in catalogue" : " · withheld")],
      ["Synthetic clip",[optionName(q.family),q.condition].filter(Boolean).join(" · ")],
      ["First candidate",[q.predicted_title,"FMA "+q.predicted].filter(Boolean).join(" · ")],
      ["Runner-up",[q.runner_up_title,q.runner_up!=null ? "FMA "+q.runner_up : ""].filter(Boolean).join(" · ") || "—"],
      ["Cosine score",format(q.score)],["Runner-up gap",format(q.gap)],["Decision",outcome(q)]
    ];
    const dl=element("dl");
    for(const [key,value] of values) dl.append(element("dt",key),element("dd",value));
    inspector.append(dl);
  }
  function show() {
    const rows=get("rows"), start=page*pageSize;
    rows.replaceChildren();
    const visible=filtered.slice(start,start+pageSize);
    visible.forEach(q=>{
      const button=element("button",null,"queryrow");
      button.dataset.id=q.id;
      const text=element("span");
      text.append(element("b",label(q)),element("small","FMA "+q.track+" · "+optionName(q.family)+" · "+q.id));
      button.append(text,element("span",outcome(q),"tag"));
      button.onclick=()=>select(q);
      rows.append(button);
    });
    if(visible.length) select(visible.includes(selected) ? selected : visible[0]);
    else {
      plot([],null);
      rows.append(element("p","No matching queries. Try another filter.","empty"));
      get("inspect").replaceChildren(element("h3","No query selected"));
    }
    get("count").textContent=visible.length ? (start+1)+"–"+(start+visible.length)+" of "+number.format(filtered.length) : "0 queries";
    get("prev").disabled=page===0;
    get("next").disabled=start+pageSize>=filtered.length;
  }
  function filter() {
    const term=get("search").value.trim().toLowerCase();
    filtered=queries.filter(q=>(get("outcome").selectedIndex===0 || outcome(q)===get("outcome").value)
      && (get("genre").selectedIndex===0 || q.genre===get("genre").value)
      && [q.id,q.track,q.title,q.artist,q.predicted,q.predicted_title,q.runner_up_title].some(v=>v!=null && String(v).toLowerCase().includes(term)));
    page=0;show();
  }
  get("search").oninput=filter;
  get("outcome").onchange=filter;get("genre").onchange=filter;
  get("prev").onclick=()=>{page--;show()};
  get("next").onclick=()=>{page++;show()};
  filter();
}

function coverage(genres) {
  const rows=get("genre-rows");
  for(const g of [...genres].sort((a,b)=>b.tracks-a.tracks)) {
    const tr=element("tr");
    for(const text of [g.genre,number.format(g.tracks),number.format(g.songs),number.format(g.correct)+" / "+number.format(g.clips)]) tr.append(element("td",text));
    rows.append(tr);
  }
}

try {
  const reply=await fetch("/evidence.json");
  if(!reply.ok) throw Error("Evidence load failed.");
  const data=await reply.json();
  benchmarks(data.methods);queryExplorer(data.queries);coverage(data.genres);
} catch {
  get("chart-note").textContent="Could not load the archived measurements. Reload the page to try again.";
  get("chart").textContent="Measurements unavailable";get("rows").textContent="Query archive unavailable";
}


fetch("/listening-catalogue.json").then(r=>r.json()).then(rows=>{
  get("demo-tracks").replaceChildren(...rows.map(r=>element("li",r.title+" · "+r.artist+" · FMA "+r.track_id)));
}).catch(()=>{get("demo-tracks").textContent="Track list unavailable.";});
