// APP_MODE is injected by server.py: "single" or "folders"
const { useState, useEffect, useRef, useCallback } = React;

const T = {
  bg:"#09090B",surface:"#111113",surface2:"#18181B",surface3:"#1F1F23",
  border:"#27272A",text:"#FAFAFA",text2:"#A1A1AA",text3:"#71717A",text4:"#52525B",
  blue:"#3B82F6",blueDim:"rgba(59,130,246,.12)",blueGlow:"rgba(59,130,246,.06)",
  green:"#22C55E",greenDim:"rgba(34,197,94,.12)",
  amber:"#F59E0B",amberDim:"rgba(245,158,11,.12)",
  red:"#EF4444",redDim:"rgba(239,68,68,.12)",
  violet:"#8B5CF6",violetDim:"rgba(139,92,246,.12)",
};
const F = {sans:"'DM Sans',system-ui,sans-serif",display:"'Instrument Sans',system-ui,sans-serif",mono:"'JetBrains Mono','Fira Code',monospace"};

// === API ===
const API = {
  single: {
    upload: async (file) => { const fd=new FormData();fd.append("file",file);const r=await fetch("/api/single/upload",{method:"POST",body:fd});if(!r.ok)throw new Error((await r.json()).detail);return r.json(); },
    chat: async (q) => { const r=await fetch("/api/single/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:q})});return r.json(); },
    status: async () => (await fetch("/api/single/status")).json(),
    reset: async () => fetch("/api/single/reset",{method:"POST"}),
  },
  folders: {
    list: async () => (await fetch("/api/folders")).json().then(r=>r.folders),
    create: async (n) => { const r=await fetch("/api/folders",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:n})});if(!r.ok)throw new Error((await r.json()).detail);return r.json(); },
    del: async (n) => fetch(`/api/folders/${encodeURIComponent(n)}`,{method:"DELETE"}),
    upload: async (fn,file) => { const fd=new FormData();fd.append("file",file);const r=await fetch(`/api/folders/${encodeURIComponent(fn)}/documents`,{method:"POST",body:fd});if(!r.ok)throw new Error((await r.json()).detail);return r.json(); },
    delDoc: async (fn,dn) => fetch(`/api/folders/${encodeURIComponent(fn)}/documents/${encodeURIComponent(dn)}`,{method:"DELETE"}),
    chat: async (q) => { const r=await fetch("/api/folders/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:q})});return r.json(); },
    reset: async () => fetch("/api/folders/chat/reset",{method:"POST"}),
  },
};

// === Icons ===
const Ic=({d,size=16,color="currentColor",style:s,...p})=>React.createElement("svg",{width:size,height:size,viewBox:"0 0 24 24",fill:"none",stroke:color,strokeWidth:"1.8",strokeLinecap:"round",strokeLinejoin:"round",style:s,...p},React.createElement("path",{d}));
const I={
  send:p=>React.createElement(Ic,{d:"M22 2L11 13M22 2l-7 20-4-9-9-4z",...p}),
  folder:p=>React.createElement(Ic,{d:"M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z",...p}),
  file:p=>React.createElement(Ic,{d:"M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6",...p}),
  tree:p=>React.createElement(Ic,{d:"M12 3v9m0 0l-4 4m4-4l4 4M4 20h16",...p}),
  check:p=>React.createElement(Ic,{d:"M20 6L9 17l-5-5",...p}),
  plus:p=>React.createElement(Ic,{d:"M12 5v14m-7-7h14",...p}),
  sparkle:p=>React.createElement(Ic,{d:"M12 2l2.09 6.26L20 10.27l-4.91 3.82L16.18 22 12 17.77 7.82 22l1.09-7.91L4 10.27l5.91-2.01z",...p}),
  upload:p=>React.createElement(Ic,{d:"M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12",...p}),
  trash:p=>React.createElement(Ic,{d:"M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2",...p}),
  x:p=>React.createElement(Ic,{d:"M18 6L6 18M6 6l12 12",...p}),
  sidebar:p=>React.createElement(Ic,{d:"M21 3H3v18h18V3zM9 3v18",...p}),
  chevRight:p=>React.createElement(Ic,{d:"M9 18l6-6-6-6",...p}),
  home:p=>React.createElement(Ic,{d:"M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z",...p}),
};

// === CSS ===
const injectCSS=()=>{if(document.getElementById("as-css"))return;const s=document.createElement("style");s.id="as-css";s.textContent=`@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');*{margin:0;padding:0;box-sizing:border-box}::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${T.border};border-radius:3px}@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}@keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}@keyframes spin{to{transform:rotate(360deg)}}@keyframes appear{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}@keyframes slideR{from{opacity:0;transform:translateX(-16px)}to{opacity:1;transform:translateX(0)}}.fadeUp{animation:fadeUp .3s ease-out both}.appear{animation:appear .25s ease-out both}.slideR{animation:slideR .25s ease-out both}`;document.head.appendChild(s);};

const Btn=({children,onClick,variant="ghost",small,disabled,style:sx,...p})=>{
  const base={display:"inline-flex",alignItems:"center",gap:6,padding:small?"5px 10px":"8px 14px",borderRadius:small?6:8,fontSize:small?11:12,fontWeight:500,fontFamily:F.sans,cursor:disabled?"default":"pointer",transition:"all .15s",border:"none",outline:"none",opacity:disabled?.5:1};
  const v={ghost:{background:"transparent",color:T.text3,border:`1px solid ${T.border}`},primary:{background:T.blue,color:"#fff"},danger:{background:T.redDim,color:T.red,border:`1px solid rgba(239,68,68,.2)`}};
  return React.createElement("button",{onClick:disabled?undefined:onClick,style:{...base,...v[variant],...sx},...p},children);
};

// === Markdown ===
const Md=({text})=>React.createElement("div",{style:{fontSize:14,lineHeight:1.8,color:T.text,fontFamily:F.sans}},
  text.split("\n").map((line,i)=>{
    const r=s=>s.replace(/\*\*([^*]+)\*\*/g,"⌜$1⌝").split(/[⌜⌝]/).map((p,j)=>j%2?React.createElement("strong",{key:j,style:{color:T.text,fontWeight:600}},p):React.createElement("span",{key:j},p));
    if(line.startsWith("•"))return React.createElement("div",{key:i,style:{display:"flex",gap:10,marginTop:6,paddingLeft:4}},React.createElement("span",{style:{color:T.blue,fontWeight:700}},"•"),React.createElement("span",{style:{color:T.text2}},r(line.slice(1).trim())));
    return React.createElement("p",{key:i,style:{marginTop:i>0&&line?8:0,color:line?T.text2:undefined}},r(line));
  }));

// === Single Mode Sidebar ===
function SingleSidebar({docInfo,setDocInfo,isOpen,onClose}){
  const [indexing,setIndexing]=useState(false);
  const fileRef=useRef(null);

  const handleUpload=async(e)=>{
    const file=e.target.files?.[0];if(!file)return;
    setIndexing(true);
    try{const info=await API.single.upload(file);setDocInfo(info);}catch(err){alert(err.message);}
    setIndexing(false);e.target.value="";
  };

  const handleReset=async()=>{await API.single.reset();setDocInfo(null);};

  if(!isOpen)return null;
  return React.createElement("div",{className:"slideR",style:{width:290,borderRight:`1px solid ${T.border}`,flexShrink:0,display:"flex",flexDirection:"column",background:T.bg,height:"100%"}},
    React.createElement("div",{style:{padding:"14px 14px 10px",borderBottom:`1px solid ${T.border}`,display:"flex",alignItems:"center",justifyContent:"space-between"}},
      React.createElement("span",{style:{fontSize:13,fontWeight:600,fontFamily:F.display,color:T.text}},"Document"),
      React.createElement(Btn,{small:true,onClick:onClose},React.createElement(I.x,{size:14}))),
    React.createElement("div",{style:{flex:1,overflow:"auto",padding:16,display:"flex",flexDirection:"column",gap:12}},
      docInfo?React.createElement(React.Fragment,null,
        React.createElement("div",{className:"fadeUp",style:{padding:16,borderRadius:12,background:T.surface,border:`1px solid ${T.border}`}},
          React.createElement("div",{style:{display:"flex",alignItems:"center",gap:10,marginBottom:10}},
            React.createElement("div",{style:{width:36,height:36,borderRadius:8,background:T.blueDim,display:"flex",alignItems:"center",justifyContent:"center"}},React.createElement(I.file,{size:16,color:T.blue})),
            React.createElement("div",null,
              React.createElement("div",{style:{fontSize:13,fontWeight:500,color:T.text,fontFamily:F.sans}},docInfo.filename),
              React.createElement("div",{style:{fontSize:11,color:T.text4,fontFamily:F.mono}},`${docInfo.pages} pages`))),
          React.createElement("div",{style:{width:5,height:5,borderRadius:"50%",background:T.green,display:"inline-block",marginRight:6}}),
          React.createElement("span",{style:{fontSize:11,color:T.green,fontFamily:F.mono}},"Indexed"),
          docInfo.description&&React.createElement("p",{style:{fontSize:12,color:T.text3,marginTop:10,lineHeight:1.5}},docInfo.description.slice(0,200)+"...")),
        React.createElement(Btn,{onClick:handleReset,variant:"danger",style:{width:"100%",justifyContent:"center"}},React.createElement(I.trash,{size:13})," Remove & upload new"))
      :React.createElement(React.Fragment,null,
        React.createElement("div",{onClick:()=>fileRef.current?.click(),style:{padding:"40px 20px",borderRadius:12,border:`2px dashed ${T.border}`,textAlign:"center",cursor:"pointer",transition:"all .15s"},
          onMouseEnter:e=>{e.currentTarget.style.borderColor=T.blue;e.currentTarget.style.background=T.blueGlow},
          onMouseLeave:e=>{e.currentTarget.style.borderColor=T.border;e.currentTarget.style.background="transparent"}},
          indexing?React.createElement(React.Fragment,null,
            React.createElement("div",{style:{width:24,height:24,border:`3px solid ${T.blue}`,borderTopColor:"transparent",borderRadius:"50%",animation:"spin .8s linear infinite",margin:"0 auto 12px"}}),
            React.createElement("p",{style:{fontSize:13,color:T.blue,fontFamily:F.mono}},"Indexing..."))
          :React.createElement(React.Fragment,null,
            React.createElement(I.upload,{size:28,color:T.text4,style:{margin:"0 auto 12px",display:"block"}}),
            React.createElement("p",{style:{fontSize:14,fontWeight:500,color:T.text2}},"Upload a PDF"),
            React.createElement("p",{style:{fontSize:12,color:T.text4,marginTop:4}},"Click or drag to upload"))))),
    React.createElement("div",{style:{padding:"12px 16px",borderTop:`1px solid ${T.border}`,fontSize:11,fontFamily:F.mono,color:T.text4}},
      React.createElement("a",{href:"/",style:{color:T.blue,textDecoration:"none"}},"← Back to home")),
    React.createElement("input",{ref:fileRef,type:"file",accept:".pdf",onChange:handleUpload,style:{display:"none"}}));
}

// === Folder Mode Sidebar ===
function FolderSidebar({folders,setFolders,isOpen,onClose,refreshFolders}){
  const[expanded,setExpanded]=useState(null);const[newName,setNewName]=useState("");const[showNew,setShowNew]=useState(false);const[indexing,setIndexing]=useState(null);const newRef=useRef(null);const fileRef=useRef(null);const uploadTarget=useRef(null);
  useEffect(()=>{if(showNew)setTimeout(()=>newRef.current?.focus(),100)},[showNew]);
  const createFolder=async()=>{const n=newName.trim();if(!n)return;try{await API.folders.create(n);setNewName("");setShowNew(false);setExpanded(n);await refreshFolders();}catch(e){alert(e.message);}};
  const delFolder=async n=>{await API.folders.del(n);await refreshFolders();if(expanded===n)setExpanded(null);};
  const delDoc=async(fn,dn)=>{await API.folders.delDoc(fn,dn);await refreshFolders();};
  const handleUpload=fn=>{uploadTarget.current=fn;fileRef.current?.click();};
  const onFiles=async e=>{const files=Array.from(e.target.files||[]);const fn=uploadTarget.current;if(!files.length||!fn)return;for(const file of files){if(!file.name.toLowerCase().endsWith(".pdf"))continue;setIndexing({folder:fn,name:file.name});try{await API.folders.upload(fn,file);}catch(err){alert(`Failed: ${file.name} — ${err.message}`);}setIndexing(null);}await refreshFolders();e.target.value="";};
  if(!isOpen)return null;
  const totalDocs=folders.reduce((a,f)=>a+f.docs.length,0);
  return React.createElement("div",{className:"slideR",style:{width:290,borderRight:`1px solid ${T.border}`,flexShrink:0,display:"flex",flexDirection:"column",background:T.bg,height:"100%"}},
    React.createElement("div",{style:{padding:"14px 14px 10px",borderBottom:`1px solid ${T.border}`,display:"flex",alignItems:"center",justifyContent:"space-between"}},
      React.createElement("span",{style:{fontSize:13,fontWeight:600,fontFamily:F.display,color:T.text}},"Documents"),
      React.createElement(Btn,{small:true,onClick:onClose},React.createElement(I.x,{size:14}))),
    React.createElement("div",{style:{flex:1,overflow:"auto",padding:10}},
      folders.map(folder=>{const isExp=expanded===folder.name;return React.createElement("div",{key:folder.name,style:{marginBottom:4}},
        React.createElement("div",{onClick:()=>setExpanded(isExp?null:folder.name),style:{display:"flex",alignItems:"center",gap:7,padding:"9px 10px",borderRadius:9,cursor:"pointer",background:isExp?T.surface2:"transparent",border:`1px solid ${isExp?T.border:"transparent"}`,transition:"all .15s"},onMouseEnter:e=>{if(!isExp)e.currentTarget.style.background=T.surface},onMouseLeave:e=>{if(!isExp)e.currentTarget.style.background="transparent"}},
          React.createElement(I.chevRight,{size:12,color:T.text4,style:{transform:isExp?"rotate(90deg)":"none",transition:"transform .15s"}}),
          React.createElement(I.folder,{size:14,color:isExp?T.blue:T.text3}),
          React.createElement("div",{style:{flex:1,minWidth:0}},React.createElement("div",{style:{fontSize:12,fontWeight:500,color:T.text,fontFamily:F.sans,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}},folder.name),React.createElement("div",{style:{fontSize:10,color:T.text4,fontFamily:F.mono}},`${folder.docs.length} docs · ${folder.total_pages||0}p`)),
          React.createElement(I.trash,{size:12,color:T.text4,style:{cursor:"pointer",opacity:.5},onClick:e=>{e.stopPropagation();delFolder(folder.name);}})),
        isExp&&React.createElement("div",{className:"fadeUp",style:{padding:"6px 0 6px 18px"}},
          folder.docs.map((doc,di)=>React.createElement("div",{key:doc.name,className:"fadeUp",style:{display:"flex",alignItems:"center",gap:7,padding:"7px 9px",borderRadius:7,marginBottom:3,background:T.surface,border:`1px solid ${T.border}`,animationDelay:`${di*30}ms`}},React.createElement(I.file,{size:12,color:T.blue}),React.createElement("div",{style:{flex:1,minWidth:0}},React.createElement("div",{style:{fontSize:11,color:T.text,fontFamily:F.sans,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}},doc.name),React.createElement("div",{style:{fontSize:9,color:T.text4,fontFamily:F.mono}},`${doc.pages}p · ${doc.time}`)),React.createElement("div",{style:{width:5,height:5,borderRadius:"50%",background:T.green,flexShrink:0}}),React.createElement(I.x,{size:11,color:T.text4,style:{cursor:"pointer"},onClick:()=>delDoc(folder.name,doc.name)}))),
          indexing&&indexing.folder===folder.name&&React.createElement("div",{className:"fadeUp",style:{display:"flex",alignItems:"center",gap:7,padding:"7px 9px",borderRadius:7,marginBottom:3,background:T.amberDim,border:"1px solid rgba(245,158,11,.2)"}},React.createElement("div",{style:{width:12,height:12,border:`2px solid ${T.amber}`,borderTopColor:"transparent",borderRadius:"50%",animation:"spin .8s linear infinite"}}),React.createElement("span",{style:{fontSize:11,color:T.amber,fontFamily:F.mono}},`Indexing ${indexing.name}...`)),
          React.createElement("div",{onClick:()=>handleUpload(folder.name),style:{display:"flex",alignItems:"center",justifyContent:"center",gap:5,padding:9,borderRadius:7,marginTop:3,border:`1.5px dashed ${T.border}`,cursor:"pointer",transition:"all .15s",fontSize:11,color:T.text4,fontFamily:F.sans},onMouseEnter:e=>{e.currentTarget.style.borderColor=T.blue;e.currentTarget.style.color=T.blue;e.currentTarget.style.background=T.blueGlow},onMouseLeave:e=>{e.currentTarget.style.borderColor=T.border;e.currentTarget.style.color=T.text4;e.currentTarget.style.background="transparent"}},React.createElement(I.upload,{size:12})," Upload PDFs")));}),
      showNew?React.createElement("div",{className:"fadeUp",style:{display:"flex",gap:5,alignItems:"center",padding:"7px 9px",borderRadius:9,background:T.surface,border:`1px solid ${T.border}`,marginTop:6}},React.createElement(I.folder,{size:13,color:T.blue}),React.createElement("input",{ref:newRef,value:newName,onChange:e=>setNewName(e.target.value),onKeyDown:e=>{if(e.key==="Enter")createFolder();if(e.key==="Escape")setShowNew(false);},placeholder:"Folder name...",style:{flex:1,background:"transparent",border:"none",outline:"none",color:T.text,fontSize:11,fontFamily:F.sans}}),React.createElement(Btn,{small:true,variant:"primary",onClick:createFolder,disabled:!newName.trim()},React.createElement(I.check,{size:11})),React.createElement(Btn,{small:true,onClick:()=>{setShowNew(false);setNewName("");}},React.createElement(I.x,{size:11})))
      :React.createElement(Btn,{onClick:()=>setShowNew(true),style:{width:"100%",justifyContent:"center",marginTop:6}},React.createElement(I.plus,{size:12})," New Folder")),
    React.createElement("div",{style:{padding:"10px 14px",borderTop:`1px solid ${T.border}`,display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:F.mono,color:T.text4}},
      React.createElement("a",{href:"/",style:{color:T.blue,textDecoration:"none"}},"← Home"),
      React.createElement("span",null,`${folders.length} folders`),
      React.createElement("span",null,`${totalDocs} docs`)),
    React.createElement("input",{ref:fileRef,type:"file",accept:".pdf",multiple:true,onChange:onFiles,style:{display:"none"}}));
}

// === Chat components ===
const FolderBadge=({name,confidence})=>React.createElement("span",{className:"appear",style:{display:"inline-flex",alignItems:"center",gap:5,padding:"4px 10px 4px 7px",borderRadius:16,background:T.violetDim,border:"1px solid rgba(139,92,246,.2)",fontSize:11,fontFamily:F.mono,color:T.violet,fontWeight:500}},React.createElement(I.folder,{size:12,color:T.violet}),name,confidence&&React.createElement("span",{style:{opacity:.6,fontSize:10}},confidence));
const Phase=({text})=>React.createElement("div",{className:"fadeUp",style:{display:"flex",alignItems:"center",gap:8,padding:"8px 14px",borderRadius:8,background:T.surface,border:`1px solid ${T.border}`,fontSize:12,fontFamily:F.mono,color:T.text3}},React.createElement("div",{style:{width:6,height:6,borderRadius:"50%",background:T.blue,animation:"pulse 1s infinite",flexShrink:0}}),text);
const Source=({s,delay})=>React.createElement("div",{className:"fadeUp",style:{padding:"9px 11px",borderRadius:8,background:T.surface,border:`1px solid ${T.border}`,animationDelay:`${delay}ms`,display:"flex",alignItems:"center",gap:9}},React.createElement("div",{style:{width:26,height:26,borderRadius:6,flexShrink:0,background:s.score>.8?T.greenDim:T.blueDim,display:"flex",alignItems:"center",justifyContent:"center"}},React.createElement(I.file,{size:12,color:s.score>.8?T.green:T.blue})),React.createElement("div",{style:{flex:1,minWidth:0}},React.createElement("div",{style:{fontSize:11,fontWeight:500,color:T.text,fontFamily:F.sans}},s.doc||s.document),React.createElement("div",{style:{fontSize:10,color:T.text4,fontFamily:F.mono}},`${s.section} · ${s.pages}`)),React.createElement("span",{style:{padding:"2px 6px",borderRadius:6,fontSize:10,fontWeight:600,fontFamily:F.mono,background:s.score>.8?T.greenDim:T.blueDim,color:s.score>.8?T.green:T.blue}},`${(s.score*100).toFixed(0)}%`));

function Msg({m}){
  if(m.role==="user")return React.createElement("div",{className:"fadeUp",style:{display:"flex",justifyContent:"flex-end",padding:"0 24px",marginBottom:16}},React.createElement("div",{style:{maxWidth:520,padding:"12px 16px",borderRadius:"16px 16px 4px 16px",background:T.blue,color:"#fff",fontSize:14,fontFamily:F.sans,lineHeight:1.6}},m.content));
  return React.createElement("div",{className:"fadeUp",style:{padding:"0 24px",marginBottom:20}},React.createElement("div",{style:{maxWidth:640}},
    m.folder&&React.createElement("div",{style:{marginBottom:8}},React.createElement(FolderBadge,{name:m.folder,confidence:m.stats?.folder_confidence})),
    m.content&&React.createElement("div",{style:{padding:"18px 20px",borderRadius:14,background:T.surface,border:`1px solid ${T.border}`,marginBottom:10}},React.createElement(Md,{text:m.content})),
    m.sources?.length>0&&React.createElement("div",{style:{marginBottom:8}},React.createElement("div",{style:{fontSize:10,fontWeight:600,color:T.text4,textTransform:"uppercase",letterSpacing:.5,marginBottom:6,paddingLeft:2,fontFamily:F.sans}},"Sources"),React.createElement("div",{style:{display:"flex",flexDirection:"column",gap:4}},m.sources.map((s,i)=>React.createElement(Source,{key:i,s,delay:i*60})))),
    m.stats?.total&&React.createElement("div",{className:"fadeUp",style:{display:"flex",gap:5,flexWrap:"wrap",animationDelay:"100ms"}},[{l:"Time",v:m.stats.total},{l:"LLM",v:m.stats.calls||m.stats.llm_calls},{l:"Cost",v:m.stats.cost,c:T.green}].map(({l,v,c})=>React.createElement("span",{key:l,style:{padding:"2px 7px",borderRadius:5,fontSize:10,fontFamily:F.mono,background:T.surface,border:`1px solid ${T.border}`,color:T.text4}},l," ",React.createElement("span",{style:{color:c||T.text3}},v))))));
}

function Welcome({mode,folders,docInfo,onSuggestion}){
  const suggestions=mode==="single"?["What is this document about?","Summarize the key findings","What are the main recommendations?","List all the data sources mentioned"]:["What was the total project cost?","Summarize the financial results","What are the key contract terms?","Compare budgets across phases"];
  const ready=mode==="single"?!!docInfo:folders?.length>0;
  return React.createElement("div",{style:{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",height:"100%",padding:"40px 24px",textAlign:"center"}},
    React.createElement("div",{className:"appear",style:{width:52,height:52,borderRadius:14,marginBottom:18,background:`linear-gradient(135deg,${T.blue},${T.violet})`,display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 8px 32px rgba(59,130,246,.2)"}},React.createElement(I.tree,{size:24,color:"#fff"})),
    React.createElement("h1",{className:"appear",style:{fontSize:24,fontWeight:700,fontFamily:F.display,color:T.text,letterSpacing:-.5,animationDelay:"50ms"}},"AlphaSearch"),
    React.createElement("p",{className:"appear",style:{fontSize:13,color:T.text3,fontFamily:F.sans,marginTop:8,maxWidth:400,lineHeight:1.6,animationDelay:"100ms"}},mode==="single"?"Upload a PDF and ask anything about it. MCTS-powered reasoning finds the exact section.":"Ask anything across your documents. Auto-routes to the right folder using MCTS reasoning."),
    mode==="single"&&docInfo&&React.createElement("div",{className:"appear",style:{marginTop:20,padding:"8px 16px",borderRadius:20,background:T.blueDim,border:`1px solid rgba(59,130,246,.2)`,fontSize:12,fontFamily:F.mono,color:T.blue,animationDelay:"150ms"}},React.createElement(I.file,{size:12,color:T.blue}),` ${docInfo.filename} (${docInfo.pages}p)`),
    mode==="single"&&!docInfo&&React.createElement("p",{className:"appear",style:{fontSize:13,color:T.amber,fontFamily:F.sans,marginTop:20,animationDelay:"150ms"}},"Upload a PDF using the sidebar to get started."),
    mode==="folders"&&folders?.length>0&&React.createElement("div",{className:"appear",style:{display:"flex",gap:6,marginTop:20,flexWrap:"wrap",justifyContent:"center",animationDelay:"150ms"}},folders.map(f=>React.createElement("span",{key:f.name,style:{padding:"4px 10px",borderRadius:16,background:T.surface2,border:`1px solid ${T.border}`,fontSize:11,fontFamily:F.mono,color:T.text4,display:"flex",alignItems:"center",gap:4}},React.createElement(I.folder,{size:11,color:T.text4}),f.name,React.createElement("span",{style:{opacity:.6}},`(${f.docs.length})`)))),
    mode==="folders"&&(!folders||folders.length===0)&&React.createElement("p",{className:"appear",style:{fontSize:13,color:T.amber,fontFamily:F.sans,marginTop:20,animationDelay:"150ms"}},"No folders yet — click Manage to create one and upload PDFs."),
    ready&&React.createElement("div",{className:"appear",style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginTop:28,maxWidth:460,width:"100%",animationDelay:"200ms"}},suggestions.map(s=>React.createElement("button",{key:s,onClick:()=>onSuggestion(s),style:{padding:"11px 14px",borderRadius:10,textAlign:"left",background:T.surface,border:`1px solid ${T.border}`,color:T.text2,fontSize:12,fontFamily:F.sans,cursor:"pointer",transition:"all .15s",lineHeight:1.4},onMouseEnter:e=>{e.currentTarget.style.borderColor=T.blue+"50";e.currentTarget.style.background=T.blueGlow},onMouseLeave:e=>{e.currentTarget.style.borderColor=T.border;e.currentTarget.style.background=T.surface}},React.createElement(I.sparkle,{size:12,color:T.text4,style:{marginBottom:5,display:"block"}}),s))));
}

// === MAIN ===
function TreeRAGApp(){
  const mode=typeof APP_MODE!=="undefined"?APP_MODE:"folders";
  const[folders,setFolders]=useState([]);
  const[docInfo,setDocInfo]=useState(null);
  const[messages,setMessages]=useState([]);
  const[input,setInput]=useState("");
  const[isStreaming,setIsStreaming]=useState(false);
  const[phase,setPhase]=useState(null);
  const[sidebarOpen,setSidebarOpen]=useState(false);
  const scrollRef=useRef(null);

  useEffect(()=>{injectCSS();if(mode==="folders")refreshFolders();else API.single.status().then(s=>{if(s.loaded)setDocInfo(s);});},[]);
  useEffect(()=>{scrollRef.current?.scrollTo({top:scrollRef.current.scrollHeight,behavior:"smooth"});},[messages,phase]);

  const refreshFolders=async()=>{try{setFolders(await API.folders.list());}catch(e){console.error(e);}};

  const send=useCallback(async text=>{
    const q=text||input.trim();if(!q||isStreaming)return;
    setMessages(p=>[...p,{id:Date.now(),role:"user",content:q}]);
    setInput("");setIsStreaming(true);setPhase("Searching...");
    try{
      const result=mode==="single"?await API.single.chat(q):await API.folders.chat(q);
      setPhase(null);setIsStreaming(false);
      setMessages(p=>[...p,{id:Date.now()+1,role:"assistant",content:result.answer,folder:result.folder,sources:result.sources,stats:result.stats}]);
    }catch(err){setPhase(null);setIsStreaming(false);setMessages(p=>[...p,{id:Date.now()+1,role:"assistant",content:`Error: ${err.message}`}]);}
  },[input,isStreaming,mode]);

  const hasMsg=messages.length>0;
  const modeLabel=mode==="single"?"Single PDF":"Folders";
  const modeColor=mode==="single"?T.blue:T.violet;

  return React.createElement("div",{style:{display:"flex",height:"100vh",width:"100%",background:T.bg,color:T.text,fontFamily:F.sans,overflow:"hidden"}},
    mode==="single"?React.createElement(SingleSidebar,{docInfo,setDocInfo,isOpen:sidebarOpen,onClose:()=>setSidebarOpen(false)}):React.createElement(FolderSidebar,{folders,setFolders,isOpen:sidebarOpen,onClose:()=>setSidebarOpen(false),refreshFolders}),
    React.createElement("div",{style:{flex:1,display:"flex",flexDirection:"column",overflow:"hidden",minWidth:0}},
      // Header
      React.createElement("div",{style:{padding:"10px 20px",borderBottom:`1px solid ${T.border}`,display:"flex",alignItems:"center",gap:10,flexShrink:0}},
        React.createElement(Btn,{small:true,onClick:()=>setSidebarOpen(!sidebarOpen)},React.createElement(I.sidebar,{size:15})),
        React.createElement("div",{style:{width:28,height:28,borderRadius:7,background:`linear-gradient(135deg,${T.blue},${T.violet})`,display:"flex",alignItems:"center",justifyContent:"center"}},React.createElement(I.tree,{size:14,color:"#fff"})),
        React.createElement("div",{style:{flex:1}},
          React.createElement("span",{style:{fontSize:14,fontWeight:700,fontFamily:F.display}},"AlphaSearch"),
          React.createElement("span",{style:{fontSize:10,color:T.text4,fontFamily:F.mono,marginLeft:8,padding:"2px 8px",borderRadius:10,background:mode==="single"?T.blueDim:T.violetDim,color:modeColor}},modeLabel)),
        React.createElement(Btn,{small:true,onClick:()=>setSidebarOpen(true)},mode==="single"?React.createElement(I.file,{size:13}):React.createElement(I.folder,{size:13})," Manage")),
      // Chat
      React.createElement("div",{ref:scrollRef,style:{flex:1,overflow:"auto",paddingTop:hasMsg?20:0}},
        !hasMsg?React.createElement(Welcome,{mode,folders,docInfo,onSuggestion:s=>{setInput(s);setTimeout(()=>send(s),50);}})
        :React.createElement(React.Fragment,null,messages.map(m=>React.createElement(Msg,{key:m.id,m})),isStreaming&&React.createElement("div",{style:{padding:"0 24px",marginBottom:16}},React.createElement("div",{style:{maxWidth:640}},phase&&React.createElement(Phase,{text:phase}))))),
      // Input
      React.createElement("div",{style:{padding:"14px 20px 18px",flexShrink:0}},
        React.createElement("div",{style:{display:"flex",gap:8,alignItems:"flex-end",maxWidth:680,margin:"0 auto",padding:"5px 5px 5px 16px",borderRadius:14,background:T.surface,border:`1px solid ${T.border}`,transition:"border-color .15s,box-shadow .15s"},onFocus:e=>{e.currentTarget.style.borderColor=T.blue;e.currentTarget.style.boxShadow=`0 0 0 3px ${T.blueGlow}`},onBlur:e=>{e.currentTarget.style.borderColor=T.border;e.currentTarget.style.boxShadow="none"}},
          React.createElement("textarea",{value:input,onChange:e=>setInput(e.target.value),onKeyDown:e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}},placeholder:mode==="single"?(docInfo?"Ask about your document...":"Upload a PDF first..."):(folders.length?"Ask anything across your documents...":"Create a folder & upload PDFs..."),disabled:isStreaming,rows:1,style:{flex:1,padding:"9px 0",background:"transparent",border:"none",outline:"none",color:T.text,fontSize:14,fontFamily:F.sans,resize:"none",lineHeight:1.5,maxHeight:100,minHeight:22},onInput:e=>{e.target.style.height="auto";e.target.style.height=Math.min(e.target.scrollHeight,100)+"px";}}),
          React.createElement("button",{onClick:()=>send(),disabled:!input.trim()||isStreaming,style:{width:38,height:38,borderRadius:11,flexShrink:0,background:input.trim()&&!isStreaming?T.blue:T.surface3,border:"none",cursor:input.trim()&&!isStreaming?"pointer":"default",display:"flex",alignItems:"center",justifyContent:"center",transition:"background .15s"}},isStreaming?React.createElement("div",{style:{width:14,height:14,border:`2px solid ${T.text4}`,borderTopColor:"transparent",borderRadius:"50%",animation:"spin .8s linear infinite"}}):React.createElement(I.send,{size:15,color:input.trim()?"#fff":T.text4}))),
        React.createElement("p",{style:{textAlign:"center",fontSize:10,color:T.text4,fontFamily:F.mono,marginTop:6}},mode==="single"?"MCTS tree search → deep read → answer":"Auto-routes → selects docs → finds sections → answers"))));
}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(TreeRAGApp));
