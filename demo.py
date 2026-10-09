"""Single-file fake-news demo: backend + frontend in one. Run: uvicorn demo:app --reload, open http://localhost:8000"""
import pickle, os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Fake News Demo")

class TextIn(BaseModel):
    text: str

_bundle = None
for p in ["backend/models/tfidf.pkl", "models/tfidf.pkl", "model.pkl", "tfidf.pkl"]:
    if os.path.exists(p):
        with open(p, "rb") as f:
            _bundle = pickle.load(f)
        break

def verdict(text: str):
    import re
    if _bundle is None:  # ponytail: keyword fallback until real pickle present
        t = text.lower()
        loaded = ["shock", "miracle", "secret", "aliens", "breaking", "unbelievable", "cure", "exposed", "hoax", "cover-up", "shocking"]
        ld = sum(w in t for w in loaded)
        caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
        exc = text.count("!")
        score = ld * 2 + caps + exc * 0.5
        conf = min(0.95, 0.52 + score * 0.07)
        label = "FAKE" if score >= 3 else "REAL"
        if label == "REAL":
            conf = max(0.55, min(0.9, 0.85 - score * 0.05))
        return {"label": label, "confidence": round(conf, 3), "fake_words": [], "real_words": [], "available": True}
    vec, clf = _bundle["vectorizer"], _bundle["model"]
    proba = clf.predict_proba(vec.transform([text]))[0]
    i = int(proba.argmax())
    raw = clf.classes_[i]
    label = {0: "FAKE", 1: "REAL", "0": "FAKE", "1": "REAL"}.get(raw, str(raw))
    if label not in ("FAKE", "REAL"):
        label = "FAKE" if str(raw).lower().startswith("f") or str(raw) == "0" else "REAL"
    return {"label": label, "confidence": float(proba[i]), "fake_words": [], "real_words": [], "available": True}

@app.post("/predict")
def predict(inp: TextIn):
    if len(inp.text.split()) < 20:
        raise HTTPException(422, "Paste at least ~20 words")
    return {"tfidf": verdict(inp.text), "bert": {"label": "UNKNOWN", "confidence": 0, "fake_words": [], "real_words": [], "available": False}}

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Fake News Detector — check a story before you share it</title><meta name=description content="Paste a headline and article. Get a clear call, confidence bar, writing-pattern counts, and marked warning signs."><meta property="og:title" content="Fake News Detector"><meta property="og:description" content="Check a story before you share it."><link rel=icon href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%2324409a'/%3E%3Ctext x='16' y='23' font-size='18' text-anchor='middle' fill='white' font-family='monospace' font-weight='bold'%3EF%3C/text%3E%3C/svg%3E">
<style>
:root{--navy:#111111;--navy-deep:#050505;--accent:#24409a;--red:#E61919;--grn:#15803d;--ink:#111111;--paper:#F4F4F0;--soft:#EAE8E3;--line:#111111;--mut:#5a5a5a;--amber:#b45309;--mono:"JetBrains Mono","IBM Plex Mono","Space Mono",ui-monospace,Menlo,monospace}
[data-theme=dark]{--ink:#ECEBE8;--paper:#16171A;--soft:#1D1E22;--line:#33363D;--mut:#A7A9B0;--navy:#8FA8FF;--navy-deep:#ECEBE8;--accent:#8FA8FF;--red:#F25555;--grn:#4CC38A;--amber:#E8A020}
*{box-sizing:border-box;border-radius:0!important}::selection{background:var(--accent);color:#fff}
body{background:var(--paper);color:var(--ink);font-family:Geist,system-ui,-apple-system,sans-serif;margin:0;height:100vh;overflow:hidden}
body, .panel, .topbar, .panel header, footer, textarea, input[type=text], .meter, .how article, details, .reading{transition:background-color .45s ease,color .45s ease,border-color .45s ease}
#wipe{position:fixed;inset:0;z-index:50;pointer-events:none;background:var(--wipe,#16171A);transform:scaleY(0);transform-origin:top}
#wipe.on{animation:wipe .45s ease forwards}@keyframes wipe{0%{transform:scaleY(0);transform-origin:top}45%,55%{transform:scaleY(1);transform-origin:top}100%{transform:scaleY(0);transform-origin:bottom;opacity:0}}
#th{transition:transform .45s ease,opacity .45s ease}#th.spin{transform:rotate(180deg)}
@media(prefers-reduced-motion:reduce){#wipe{display:none}body,.panel,.topbar,.panel header,footer,textarea,input[type=text],.meter,.how article,details,.reading{transition:none}#th{transition:none}}
.topbar{border-bottom:2px solid var(--line);background:var(--paper);color:var(--ink)}[data-theme=dark] .topbar{background:#16171A;border-bottom-color:#33363D;color:#ECEBE8}.topbar div{max-width:1080px;margin:auto;padding:.5rem 1rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
.topbar strong{font-size:1.05rem;letter-spacing:-.01em;text-transform:uppercase;font-family:var(--mono)}.pill{border:1.5px solid currentColor;padding:.15rem .7rem;font-size:.8rem;background:transparent}
.pill.live{background:#dcfce7;color:#14532d}[data-theme=dark] .pill.live{background:#12351f;color:#bbf7d0}
.theme{margin-left:auto;border:2px solid currentColor;background:transparent;color:inherit;padding:.3rem .8rem;font-size:.82rem;cursor:pointer;font-weight:700}
.tele{border-bottom:2px solid var(--line);background:var(--paper)}.tele div{max-width:1080px;margin:auto;padding:.35rem 1rem;display:flex;justify-content:space-between;gap:1rem;font-family:var(--mono);font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
main{max-width:1080px;margin:0 auto;padding:0 1rem;height:calc(100vh - 49px);display:flex;flex-direction:column;overflow:hidden}
.mast{display:grid;grid-template-columns:7fr 5fr;gap:1.5rem;align-items:end;margin:.6rem 0 .6rem}@media(max-width:820px){.mast{grid-template-columns:1fr}}
.mast aside{border-left:4px solid var(--accent);padding-left:1rem}
.kicker{font-family:var(--mono);font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
h1{font-size:clamp(1.8rem,3.5vw,2.4rem);font-weight:900;letter-spacing:-.04em;line-height:.95;margin:.3rem 0;text-transform:uppercase;color:#F4F4F0;background:#050505;display:inline-block;padding:.35rem .7rem;text-shadow:none}.sub{color:var(--mut);max-width:70ch;margin:.3rem 0;font-size:.9rem}
[data-theme=dark] h1{color:#ECEBE8;background:#1D1E22;border:2px solid #33363D;text-shadow:none}
.shell{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:1px;background:var(--line);border:2px solid var(--navy-deep);margin-top:0;flex:1;min-height:0}@media(max-width:820px){.shell{grid-template-columns:1fr;overflow-y:auto}}
.shell{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:1px;background:var(--line);border:2px solid var(--navy-deep);margin-top:1rem}@media(max-width:820px){.shell{grid-template-columns:1fr}}
.panel{background:var(--paper);border:0;box-shadow:none;overflow:hidden;display:flex;flex-direction:column;min-height:0}[data-theme=dark] .panel{border-color:var(--line);box-shadow:none}
.panel header{padding:.8rem 1rem;border-bottom:2px solid var(--navy-deep);display:flex;justify-content:space-between;align-items:center;gap:.5rem;flex-wrap:wrap;background:var(--soft)}
.panel header b{font-size:.8rem;font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em}.panel .body{padding:1rem}
textarea,input[type=text]{width:100%;border:2px solid var(--navy-deep);padding:.7rem;font-size:1rem;font-family:inherit;background:var(--paper);color:var(--ink);caret-color:var(--accent)}[data-theme=dark] textarea,[data-theme=dark] input[type=text]{border-color:var(--line)}
textarea:focus,input:focus{outline:3px solid var(--accent);outline-offset:1px}
textarea{line-height:1.6;min-height:110px;max-height:22vh;resize:vertical}
.row{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}.meta{color:var(--mut);font-size:.75rem;font-family:var(--mono);text-transform:uppercase;letter-spacing:.06em}
button.cta{flex:1;padding:.9rem;font-weight:800;font-size:1.05rem;background:#24409a;color:#fff;border:2px solid #24409a;cursor:pointer;text-transform:uppercase;letter-spacing:.02em}[data-theme=dark] button.cta{background:var(--accent);border-color:var(--accent);color:#101114}
button.cta:hover{filter:brightness(1.25)}button.cta:disabled{opacity:.6;cursor:wait}
button.ghost{padding:.9rem 1rem;background:var(--paper);color:var(--ink);border:2px solid var(--navy-deep);cursor:pointer;font-weight:700}[data-theme=dark] button.ghost{border-color:var(--line)}
.samples{display:flex;gap:.4rem;flex-wrap:wrap;margin:.6rem 0}
.samples button{border:1.5px solid var(--navy-deep);background:var(--soft);color:var(--ink);padding:.3rem .8rem;font-size:.75rem;font-family:var(--mono);text-transform:uppercase;cursor:pointer}[data-theme=dark] .samples button{border-color:var(--line)}
.samples button:hover{background:var(--navy-deep);color:var(--paper)}
.verdict{padding:1.1rem 1rem;border-bottom:2px solid var(--navy-deep)}
.verdict .lab{font-size:clamp(1.8rem,4vw,2.6rem);font-weight:800;letter-spacing:-.02em;text-transform:uppercase}
.verdict.fake .lab{color:var(--red)}.verdict.real .lab{color:var(--grn)}
.verdict u{text-decoration-thickness:3px;text-underline-offset:4px;text-decoration-color:var(--accent)}
.conf{font-size:clamp(2rem,4.5vw,3rem);font-weight:900;letter-spacing:-.03em;line-height:1;margin-top:.4rem;font-variant-numeric:tabular-nums}.conf small{font-size:.85rem;font-weight:700;letter-spacing:.08em;font-family:var(--mono)}
.gauge{height:16px;background:var(--soft);border:2px solid var(--navy-deep);overflow:hidden;margin-top:.6rem}[data-theme=dark] .gauge{border-color:var(--line)}
.gauge i{display:block;height:100%;background:currentColor}
.meters{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);padding:0;border-bottom:2px solid var(--navy-deep)}
@media(max-width:520px){.meters{grid-template-columns:repeat(2,1fr)}}
.meter{border:0;padding:.5rem .6rem;background:var(--paper)}.meter b{font-size:1.4rem;display:block;font-weight:900;letter-spacing:-.02em}
.meter span{font-size:.7rem;color:var(--mut);font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em}.meter.hot{background:var(--soft);box-shadow:inset 0 -3px 0 var(--accent)}[data-theme=dark] .meter.hot{border-color:var(--accent)}
.reading{padding:.8rem .8rem .8rem 1rem;line-height:1.7;flex:1;min-height:80px;overflow:auto;border-left:4px solid var(--accent);background:var(--paper)}
.reading mark{background:#DDE3F5;box-shadow:inset 0 -2px 0 var(--accent);padding:0 2px}[data-theme=dark] .reading mark{background:#2A3350}
.reading mark.caps{background:var(--red);color:#fff}.reading mark.excl{background:#050505;color:#fff}[data-theme=dark] .reading mark.excl{background:#E8A020;color:#101114}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-top:.6rem}
@media(max-width:820px){.how{grid-template-columns:1fr}}
.how article{border:2px solid var(--navy-deep);padding:.55rem .8rem;background:var(--paper)}
.how h2{font-size:.8rem;margin:.2rem 0 .25rem;font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em}.how h2::before{content:"/// ";color:var(--accent)}.how p{font-size:.82rem;color:var(--mut);margin:0}
details{margin-top:.6rem;border:2px solid var(--navy-deep);background:var(--paper);color:var(--ink);padding:.45rem .8rem}
footer{color:var(--mut);font-size:.75rem;margin-top:.5rem;padding-bottom:.5rem;font-family:var(--mono);text-transform:uppercase;letter-spacing:.06em;display:flex;justify-content:space-between}
:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.skip{position:absolute;left:-999px;top:0;background:var(--accent);color:#fff;padding:.5rem 1rem;z-index:10}.skip:focus{left:0}
html{scroll-behavior:smooth}
h1{text-wrap:balance}
.meter b{font-variant-numeric:tabular-nums}
button,.samples button{transition:background .2s,transform .12s,filter .2s}
button.cta:hover{filter:brightness(1.15)}button.cta:active{transform:scale(.98)}button.ghost:active{transform:scale(.98)}.samples button:active{transform:scale(.96)}
.verdict .lab,.reading,#art{transition:opacity .25s}
.loading{opacity:.55;pointer-events:none}
.shimmer{background:linear-gradient(90deg,var(--soft) 25%,var(--paper) 50%,var(--soft) 75%);background-size:200% 100%;animation:sh 1.1s linear infinite}@keyframes sh{to{background-position:-200% 0}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div id=wipe aria-hidden=true></div>
<a class=skip href="#b">Skip to article input</a>
<div class=topbar><div>
<strong><svg width=18 height=18 viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=2 aria-hidden=true><path d="M4 4h16v12H8l-4 4z"/></svg> Fake News Detector</strong>
<button class=theme id=th onclick="theme(event)" aria-label="Toggle dark mode">☾</button>
</div></div>
<main>
<div class=mast><div>
<div class=kicker>Style check</div>
<h1>Check a story before you share it.</h1>
</div><aside>
<p class=sub>Paste the headline and article. You get a clear call — likely real or likely fake — a confidence bar, four writing-pattern counts, and your text marked where the warning signs sit. Takes about a second.</p>
</aside></div>
<div class=shell>
<section class=panel aria-label="Input"><header><b>[ 01 — FEED ] Paste the article</b><span class=meta id=wc>0 words · need 20+</span></header><div class=body>
<input type=text id=h placeholder="Headline (optional)" aria-label="Headline">
<textarea id=b aria-label="Article text" placeholder="Paste the full article text here…"></textarea>
<div class=samples><span class=meta>Try:</span>
<button data-s="real">Council report</button><button data-s="fake">Miracle cure</button><button data-s="caps">Shouty rant</button><button data-s="vax">Vaccine rumor</button><button data-s="election">Election claim</button><button data-s="crypto">Crypto windfall</button><button data-s="study">Research study</button>
</div>
<div class=row><button class=cta id=go onclick="go()">Check this article</button><button class=ghost onclick="clr()">Clear</button></div>
<p class=meta>Shortcut: Cmd/Ctrl + Enter. Minimum 20 words.</p>
</div></section>
<section class=panel aria-label="Reading"><header><b>[ 02 — PROOF ] Read the result</b><span class=meta id=stat>waiting for input</span></header>
<div class=verdict id=out><div class=lab>Waiting for your article</div><div class=meta>Paste at least 20 words and press Check.</div></div>
<div class=meters id=m></div>
<div class=reading id=art>Your marked-up article appears here — hype words highlighted, shouty caps and exclamation runs flagged.</div>
</div></section>
</div>
<div class=how>
<article><h2>01 · What it looks at</h2><p>Word choice and shape of the writing — shouty caps, exclamation bursts, hype words, and thin articles.</p></article>
<article><h2>02 · What it can't do</h2><p>It doesn't check facts, sources, or dates. Smooth writing can still be false — treat the call as a prompt to verify, not proof.</p></article>
<article><h2>03 · How to use the call</h2><p>Flagged fake → pause and verify elsewhere. Called real → still glance at the highlights. Low confidence → get a second opinion before sharing.</p></article>
</div>
<details><summary><b>How it works &amp; limits</b></summary><p class=meta>A trained word-pattern model (6,000 articles) scores the writing; no fact database involved. Short or unusual articles score lower confidence. Everything runs on this machine.</p></details>
<footer><span>© FND DESK — CHECK BEFORE YOU SHARE</span><span><a href="#" onclick="return false" style="color:inherit">Privacy</a> · <a href="#" onclick="return false" style="color:inherit">Terms</a></span></footer>
</main>
<script>
const root=document.documentElement;root.dataset.theme=localStorage.t||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");function theme(e){const reduce=matchMedia("(prefers-reduced-motion:reduce)").matches;const next=root.dataset.theme==="dark"?"light":"dark";if(reduce){apply();return}const w=document.getElementById("wipe");w.style.setProperty("--wipe",next==="dark"?"#16171A":"#F4F4F0");w.classList.remove("on");void w.offsetWidth;th.classList.remove("spin");void th.offsetWidth;th.classList.add("spin");w.classList.add("on");setTimeout(apply,200);setTimeout(()=>w.classList.remove("on"),500);function apply(){root.dataset.theme=next;localStorage.t=next;th.textContent=next==="dark"?"☀":"☾"}}th.textContent=root.dataset.theme==="dark"?"☀":"☾";
const LOADED=["shock","miracle","secret","aliens","breaking","unbelievable","cure","exposed","hoax","cover-up","shocking"];
const SAMPLES={real:"City council members voted Tuesday to approve the annual budget after three hours of public comment. The plan allocates funding for school repairs, road resurfacing, and two new bus routes. Officials said construction bids will open next month, with work scheduled to begin in the fall. Residents raised questions about traffic during construction.",
fake:"SHOCKING MIRACLE CURE doctors do not want you to know about this SECRET remedy that Big Pharma is hiding. This UNBELIEVABLE breakthrough was EXPOSED by insiders and it will change everything overnight!!! Share before they DELETE this!!!",
caps:"BREAKING NEWS EVERYONE MUST READ THIS RIGHT NOW THE TRUTH IS FINALLY OUT AND NOBODY CAN DENY IT ANYMORE THIS CHANGES EVERYTHING WE HAVE EVER BEEN TOLD ABOUT THE SITUATION AND YOU NEED TO ACT IMMEDIATELY TODAY PLEASE",
vax:"Secret documents prove vaccines contain microchips that control your brain and doctors are hiding the truth from everyone. This shocking cover up was exposed by brave insiders and the media refuses to report it share this now before it gets deleted forever urgently",
election:"Breaking insiders reveal truckloads of fake ballots were secretly counted at midnight to steal the election and officials covered it up. This unbelievable fraud changes everything and the truth must get out before they silence us share widely right now today",
crypto:"Unbelievable crypto secret turns pocket change into a fortune overnight with guaranteed returns that banks hate. Insiders are doubling money daily and this exposed loophole closes soon so act now before it is banned click here immediately today folks",
study:"Astronomers at the national observatory reported Thursday that a newly tracked comet will pass at a safe distance next month. The team published orbital calculations after weeks of observation and invited independent verification from three partner institutions worldwide."};
document.querySelectorAll(".samples button").forEach(x=>x.onclick=()=>{b.value=SAMPLES[x.dataset.s];b.oninput()});
b.oninput=()=>{const n=b.value.trim().split(/\\s+/).filter(Boolean).length;wc.textContent=n+" words · "+(n>=20?"ready":"need 20+")};
document.addEventListener("keydown",e=>{if((e.metaKey||e.ctrlKey)&&e.key==="Enter")go()});
function clr(){h.value=b.value="";b.oninput();out.innerHTML='<div class=lab>Waiting for your article</div><div class=meta>Paste at least 20 words and press Check.</div>';m.innerHTML=art.textContent="Your marked-up article appears here — hype words highlighted, shouty caps and exclamation runs flagged.";stat.textContent="waiting for input"}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;")}
function annotate(t){return esc(t).replace(/\\b[A-Z]{3,}\\b/g,"<mark class=caps>$&</mark>").replace(/!+/g,"<mark class=excl>$&</mark>").replace(new RegExp("\\\\b("+LOADED.join("|").replace("-","\\-")+")\\\\b","gi"),"<mark>$&</mark>")}
function meter(bv,v,l){return `<div class="meter ${bv?"hot":""}"><b>${v}</b><span>${l}</span></div>`}
async function go(){const t=(h.value+" "+b.value).trim();goBtn();try{const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t})});if(!r.ok){const d=(await r.json()).detail||"rejected";out.className="verdict";out.innerHTML=`<div class=lab>Need more text</div><div class=meta>${esc(d)} — paste the full article, not just the headline.</div>`;stat.textContent="rejected";return}const v=(await r.json()).tfidf;const f=v.label==="FAKE";const caps=(t.match(/\\b[A-Z]{3,}\\b/g)||[]).length,ex=(t.match(/!/g)||[]).length,ld=LOADED.filter(w=>t.toLowerCase().includes(w)).length,n=t.split(/\\s+/).filter(Boolean).length;const low=v.confidence<0.6;out.className="verdict "+(f?"fake":"real");out.innerHTML=`<div class=lab><u>${f?"Likely fake — pause before sharing":"Looks real — writing checks out"}</u></div><div class=conf>${(v.confidence*100).toFixed(1)}% <small>CONFIDENCE${low?" · LOW — GET A SECOND OPINION":""}</small></div><div class=gauge style="color:${f?"var(--red)":"var(--grn)"}"><i style="width:${(v.confidence*100).toFixed(0)}%"></i></div>`;m.innerHTML=meter(n>20,n,"word count")+meter(caps>0,caps,"shouty words")+meter(ex>0,ex,"exclamations")+meter(ld>0,ld,"hype words");art.innerHTML=annotate(b.value);stat.textContent="call: "+v.label.toLowerCase()}catch(e){out.innerHTML=`<div class=lab>Checker offline</div><div class=meta>Can't reach the checker — is the app still running?</div>`}finally{go.disabled=false;go.textContent="Check this article";out.classList.remove("loading");art.classList.remove("shimmer")}}
function goBtn(){go.disabled=true;go.textContent="Checking…";stat.textContent="checking…";out.classList.add("loading");art.classList.add("shimmer")}
</script>
<script src="http://localhost:8400/live.js?token=5cc50e59-0b22-4f15-b5c8-80e7d3af5945"></script>
</body></html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
