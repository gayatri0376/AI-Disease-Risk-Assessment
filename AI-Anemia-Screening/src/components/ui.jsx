import {ResponsiveContainer} from 'recharts'
import {ShieldAlert,Inbox} from 'lucide-react'
const tone={Low:'ok',Verified:'ok',Completed:'ok',Active:'ok',Published:'ok',Upcoming:'in',Medium:'wn',Pending:'wn',Draft:'wn',Cancelled:'dn',Rejected:'dn',Needs:'dn',Inactive:'dn'}
export const Badge=({s})=><span className={'bd '+(tone[String(s).split(' ')[0]]||'in')}>{s}</span>
export const Stat=({icon:I,label,value,c=''})=><div className="card stat"><span className={'ic '+c}><I size={22}/></span><div><div style={{color:'var(--m)',fontSize:13,fontWeight:500}}>{label}</div><b style={{color:'var(--n)'}}>{value}</b></div></div>
export const Head=({title,sub,children})=><div className="ph"><div><h2>{title}</h2><span>{sub}</span></div><div>{children}</div></div>
export const Table=({cols,rows})=><div className="card tw">{rows.length?<table><thead><tr>{cols.map(c=><th key={c.h}>{c.h}</th>)}</tr></thead><tbody>{rows.map((r,i)=><tr key={i}>{cols.map(c=><td key={c.h}>{c.r?c.r(r):r[c.k]}</td>)}</tr>)}</tbody></table>:<Empty title="Nothing to show yet" text="Records will appear here once they are available."/>}</div>
export const Chart=({title,children})=><div className="card"><b style={{color:'var(--n)'}}>{title}</b><div style={{height:240,marginTop:12}}><ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer></div></div>
export const Disclaimer=()=><Alert><ShieldAlert size={18} style={{flex:'none'}}/><span>This platform provides preliminary screening support and is not a medical diagnosis.</span></Alert>
export const Field=({label,...p})=><div><label>{label}</label><input {...p}/></div>
export const Alert=({type='',children})=><div className={'alert '+type}>{children}</div>
export const Empty=({title,text,children})=><div className="empty"><div className="ic"><Inbox size={24}/></div><h3>{title}</h3><p>{text}</p>{children}</div>
export const Loading=({text='Loading…'})=><div className="card empty"><div className="spin"/><p>{text}</p></div>
export const Tabs=({tabs,value,onChange})=><div className="tabs">{tabs.map(t=><button key={t} className={t===value?'on':''} onClick={()=>onChange(t)}>{t}</button>)}</div>
export const Modal=({open,title,onClose,onConfirm,confirmText='Confirm',danger,children})=>open?<div className="modal-bg" onClick={onClose}><div className="modal" onClick={e=>e.stopPropagation()}><h3>{title}</h3><p>{children}</p><div className="foot"><button className="btn o" onClick={onClose}>Cancel</button><button className={'btn '+(danger?'r':'')} onClick={onConfirm}>{confirmText}</button></div></div></div>:null
