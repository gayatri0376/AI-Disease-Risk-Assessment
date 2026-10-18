import {useState, useEffect} from 'react'
import {useNavigate,useParams,Link} from 'react-router-dom'
import {ClipboardList,Activity,CalendarDays,FileText,Download} from 'lucide-react'
import {LineChart,Line,XAxis,YAxis,Tooltip} from 'recharts'
import {Head,Stat,Chart,Table,Badge,Disclaimer,Field,Tabs,Empty,Modal} from '../../components/ui'
import {assessments as mockAssessments, cbc as defaultCbc} from '../../data/mock'
import {useApp} from '../../context/AppContext'

const SYM=['Fatigue','Weakness','Dizziness','Headache','Shortness of breath','Pale skin']
const LAB=['Hemoglobin','RBC Count','Hematocrit','MCV','MCH','MCHC','RDW','WBC Count','Platelet Count']

export function PDash(){
  const nav=useNavigate()
  const {currentUser}=useApp()
  const user = currentUser || JSON.parse(localStorage.getItem('current_user') || '{}')
  const firstName = (user?.name || 'User').trim().split(' ')[0]
  
  const hour = new Date().getHours()
  const greeting = hour >= 5 && hour < 12 
    ? 'Good morning' 
    : hour >= 12 && hour < 17 
      ? 'Good afternoon' 
      : 'Good evening'

  const hist = JSON.parse(localStorage.getItem('assessment_history') || 'null') || mockAssessments
  const latest = hist[0] || {status:'Low Risk',score:28}
  const chartData = [...hist].slice(0, 6).reverse()

  return <>
    <Head title={`${greeting}, ${firstName}`} sub="Your screening summary"/>
    <div className="grid">
      <Stat icon={ClipboardList} label="Total Assessments" value={hist.length}/>
      <Stat icon={Activity} label="Latest Screening" value={latest.status}/>
      <Stat icon={CalendarDays} label="Upcoming Appointment" value="12 Oct"/>
      <Stat icon={FileText} label="Reports" value={hist.length}/>
    </div>
    <div className="g2">
      <Chart title="Assessment history (risk score %)">
        <LineChart data={chartData}>
          <XAxis dataKey="date"/>
          <YAxis domain={[0, 100]}/>
          <Tooltip/>
          <Line dataKey="score" stroke="#2463A6" strokeWidth={2}/>
        </LineChart>
      </Chart>
      <div className="card">
        <h3>Quick actions</h3>
        <p>Start a new AI screening or review your latest clinical results.</p>
        <button className="btn" onClick={()=>nav('/patient/assessment')}>Start Assessment</button>
        <button className="btn o" onClick={()=>nav('/patient/result')}>View Latest Report</button>
        <button className="btn o" onClick={()=>nav('/patient/doctors')}>Find a Doctor</button>
      </div>
    </div>
  </>
}

export function Assess(){
  const currentUser = JSON.parse(localStorage.getItem('current_user') || '{}')
  const [s,setS]=useState(0)
  const [f,setF]=useState({
    'Full Name': currentUser.name || 'Mohini Jadhav',
    'Email': currentUser.email || 'mohini@example.com',
    'Age': currentUser.age ? String(currentUser.age) : '22',
    'Gender': currentUser.gender || 'Female',
    'Weight (kg)': '54',
    'Height (cm)': '162',
    'Hemoglobin': '13.5',
    'RBC Count': '4.5',
    'Hematocrit': '40.0',
    'MCV': '88.0',
    'MCH': '29.5',
    'MCHC': '33.5',
    'RDW': '13.0',
    'WBC Count': '7.0',
    'Platelet Count': '260',
  })
  const [busy,setB]=useState(false)
  const [err,setErr]=useState('')
  const nav=useNavigate()

  const set=(k,v)=>setF({...f,[k]:v})

  const submit=async()=>{
    setB(true)
    setErr('')
    try {
      const activeUser = JSON.parse(localStorage.getItem('current_user') || '{}')
      const payload = {
        ...f,
        name: f['Full Name'] || activeUser.name || 'Anonymous Patient',
        email: f['Email'] || activeUser.email || null,
        patient_id: activeUser.id || null,
      }

      const response = await fetch('http://127.0.0.1:8000/api/screen-patient/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Server error while running AI assessment.')
      }

      const data = await response.json()

      // Save latest result in localStorage
      localStorage.setItem('latest_assessment', JSON.stringify(data))

      // Update assessment history in localStorage
      const prev = JSON.parse(localStorage.getItem('assessment_history') || 'null') || mockAssessments
      const updatedHistory = [
        {
          id: data.assessment_id,
          date: data.date,
          status: data.status,
          score: data.score,
          predicted_diagnosis: data.predicted_diagnosis,
        },
        ...prev,
      ]
      localStorage.setItem('assessment_history', JSON.stringify(updatedHistory))

      nav('/patient/result')
    } catch(e) {
      setB(false)
      setErr(e.message || 'Could not connect to Django backend. Make sure the server is running on port 8000.')
    }
  }

  if(busy) return (
    <div className="card empty" style={{padding:70}}>
      <div className="spin"/>
      <h3 style={{marginTop:16}}>Processing your assessment…</h3>
      <p>Running Random Forest Machine Learning Model in Django backend</p>
    </div>
  )

  const T=['Personal Information','Health Information','Laboratory inputs (CBC)','Review & Confirm']

  return <>
    <Head title="Health Assessment" sub={`Step ${s+1} of 4: ${T[s]}`}/>
    <div className="card">
      <div className="steps">{T.map((_,i)=><div key={i} className={i<=s?'on':''}/>)}</div>
      
      {err && (
        <div style={{background:'#fee2e2',color:'#b91c1c',padding:12,borderRadius:8,marginBottom:16}}>
          <b>Error:</b> {err}
        </div>
      )}

      {s===0&&<div className="f">{['Full Name','Email','Age','Gender','Weight (kg)','Height (cm)'].map(k=><Field key={k} label={k} value={f[k]||''} onChange={e=>set(k,e.target.value)}/>)}</div>}
      {s===1&&<div className="f">{SYM.map(k=><div key={k}><label>{k}</label><select value={f[k]||'No'} onChange={e=>set(k,e.target.value)}><option>No</option><option>Yes</option></select></div>)}</div>}
      {s===2&&<div className="f">{LAB.map(k=><Field key={k} label={k} type="number" step="any" value={f[k]||''} onChange={e=>set(k,e.target.value)}/>)}</div>}
      {s===3&&<div className="f">{[...['Full Name','Email','Age','Gender','Weight (kg)','Height (cm)'],...SYM,...LAB].map(k=><div key={k}><label>{k}</label>{f[k]||'—'}</div>)}</div>}
      
      <div style={{marginTop:20}}><Disclaimer/></div>
      <div style={{marginTop:16}}>
        {s>0&&<button className="btn o" onClick={()=>setS(s-1)}>Back</button>}
        {s<3?<button className="btn" onClick={()=>setS(s+1)}>Next</button>:<button className="btn g" onClick={submit}>Submit to AI Model</button>}
      </div>
    </div>
  </>
}

export function Result(){
  const nav=useNavigate()
  const savedData = JSON.parse(localStorage.getItem('latest_assessment') || 'null')
  
  const r = savedData || {
    status: 'Low Risk',
    score: 28,
    assessment_id: 'AS-1004',
    date: '4 Oct 2026',
    predicted_diagnosis: 'Healthy',
    cbc: defaultCbc,
    recommendations: [
      {title: 'Routine Health Maintenance', description: 'Continue balanced dietary intake rich in essential nutrients.', priority: 'low'}
    ]
  }

  const cbcList = r.cbc || defaultCbc
  const recs = r.recommendations || (r.diagnosis_report && r.diagnosis_report.recommendations) || []
  const hist = JSON.parse(localStorage.getItem('assessment_history') || 'null') || mockAssessments

  return <>
    <Head title="AI Screening Result" sub={`Assessment ID ${r.assessment_id || 'AS-1004'} · ${r.date || 'Today'}`}/>
    
    <div className="card" style={{textAlign:'center'}}>
      <p style={{margin:0,color:'var(--m)'}}>AI Screening Result</p>
      <h1 style={{fontSize:40,margin:'8px 0'}}>{r.status}</h1>
      <Badge s={r.status}/>
      <p style={{marginTop:10}}>Risk Score: <b>{r.score}%</b></p>
      {r.predicted_diagnosis && (
        <p style={{fontSize:16,color:'var(--t)',fontWeight:600,margin:'6px 0'}}>
          Diagnosis Subtype: {r.predicted_diagnosis}
        </p>
      )}
    </div>

    <div className="g2">
      <div className="card">
        <h3>CBC Summary (Input Analyzed)</h3>
        {cbcList.map(([k,v])=><p key={k} style={{margin:'5px 0'}}>{k}: <b>{v}</b></p>)}
      </div>

      <div className="card">
        <h3>Screening History</h3>
        <Table cols={[{h:'ID',k:'id'},{h:'Date',k:'date'},{h:'Status',r:x=><Badge s={x.status}/>}]} rows={hist}/>
        
        <h3 style={{marginTop:20}}>AI Clinical Recommendations</h3>
        {recs.length > 0 ? (
          recs.map((rec, i) => (
            <div key={i} style={{marginBottom:10,padding:10,background:'rgba(36,99,166,0.06)',borderRadius:8,borderLeft:'4px solid var(--t)'}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                <b>{rec.title}</b>
                <Badge s={rec.priority}/>
              </div>
              <p style={{margin:'4px 0 0',fontSize:13,color:'var(--n)'}}>{rec.description}</p>
            </div>
          ))
        ) : (
          <p>Discuss your result with a qualified doctor for full evaluation.</p>
        )}
        <button className="btn" style={{marginTop:12}} onClick={()=>nav('/patient/doctors')}>Find a Doctor</button>
      </div>
    </div>

    <div style={{marginTop:16}} className="note">
      {r.clinical_disclaimer || "This is an academic demonstration screening result generated by a machine learning model. It is not a clinical diagnosis. Consult a qualified doctor for medical evaluation."}
    </div>
  </>
}

export function Reports(){
  const nav=useNavigate()
  const {notify}=useApp()
  const hist = JSON.parse(localStorage.getItem('assessment_history') || 'null') || mockAssessments

  return <>
    <Head title="My Reports" sub="Your saved screening reports"/>
    <Table cols={[
      {h:'Report ID',k:'id'},
      {h:'Assessment Date',k:'date'},
      {h:'Status',r:x=><Badge s={x.status}/>},
      {h:'Risk Score',r:x=>x.score+'%'},
      {h:'Action',r:()=><><button className="btn s" onClick={()=>nav('/patient/result')}>View</button><button className="btn s o" onClick={()=>notify('Report downloaded')}><Download size={14}/>Download</button></>}
    ]} rows={hist}/>
  </>
}

export function Doctors(){
  const [docsList, setDocsList] = useState([])
  const {docs} = useApp()
  const [q,setQ]=useState('')
  const [sp,setSp]=useState('')

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/doctors/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) setDocsList(data)
        else setDocsList(docs)
      })
      .catch(() => setDocsList(docs))
  }, [])

  const currentDocs = docsList.length > 0 ? docsList : docs
  const list = currentDocs.filter(d=>d.status==='Verified'&&(d.name+d.spec).toLowerCase().includes(q.toLowerCase())&&(!sp||d.spec===sp))

  return <>
    <Head title="Find a Doctor" sub="Verified doctors from MySQL database"/>
    <div className="f" style={{marginBottom:16}}>
      <input placeholder="Search doctor by name or specialization" value={q} onChange={e=>setQ(e.target.value)}/>
      <select value={sp} onChange={e=>setSp(e.target.value)}>
        <option value="">All specializations</option>
        {[...new Set(currentDocs.map(d=>d.spec))].map(s=><option key={s}>{s}</option>)}
      </select>
    </div>
    <div className="grid">
      {list.map(d=>(
        <div className="card" key={d.id}>
          <div className="av">{(d.name.split(' ')[1] || d.name)[0]}</div>
          <h3 style={{marginTop:10}}>{d.name}</h3>
          <p style={{margin:0}}>
            {d.spec} · {d.exp} yrs<br/>
            {d.loc} · Available {d.avail}<br/>
            Rating {d.rating}
          </p>
          <Link to={`/patient/doctors/${d.id}`} className="btn s" style={{marginTop:10}}>View Profile</Link>
        </div>
      ))}
      {!list.length&&<p>No doctors match your search.</p>}
    </div>
  </>
}

export function DoctorProfile(){
  const {id}=useParams()
  const {docs,setAppts,notify}=useApp()
  const d=docs.find(x=>x.id==id)||docs[0]
  const [done,setDone]=useState(false)
  const [b,setB]=useState({date:'2026-10-15',time:'10:00',why:''})

  const book=async()=>{
    const activeUser = JSON.parse(localStorage.getItem('current_user') || '{}')
    const myName = activeUser.name || 'Mohini Jadhav'

    // Save appointment in MySQL
    try {
      await fetch('http://127.0.0.1:8000/api/appointments/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          patient_name: myName,
          doctor_name: d.name,
          date: b.date,
          time: b.time,
          why: b.why || 'Clinical consultation',
          status: 'Upcoming'
        })
      })
    } catch(e) {
      console.warn('Saved appointment locally')
    }

    setAppts(a=>[...a,{id:Date.now(),patient:myName,doctor:d.name,date:b.date,time:b.time,status:'Upcoming'}])
    setDone(true)
    notify('Appointment confirmed and recorded in database')
  }

  return <>
    <Head title={d.name} sub={`${d.spec} · ${d.exp} years · ${d.loc}`}/>
    <div className="g2" style={{marginTop:0}}>
      <div className="card">
        <h3>About Doctor</h3>
        <p>Experienced specialist in hematologic disorders and anemia management.</p>
        <p>Consultation: 30 min, in-person or video</p>
        <p>Available time slots: 10:00, 11:30, 15:00, 16:30</p>
      </div>
      <div className="card">
        {done ? (
          <div style={{textAlign:'center'}}>
            <h3>Appointment Confirmed</h3>
            <p>{b.date} at {b.time} with {d.name}</p>
            <Link className="btn" to="/patient/appointments">View Appointments</Link>
          </div>
        ) : (
          <>
            <h3>Book Appointment</h3>
            <div className="f">
              <Field label="Date" type="date" value={b.date} onChange={e=>setB({...b,date:e.target.value})}/>
              <div>
                <label>Time</label>
                <select value={b.time} onChange={e=>setB({...b,time:e.target.value})}>
                  {['10:00','11:30','15:00','16:30'].map(t=><option key={t}>{t}</option>)}
                </select>
              </div>
            </div>
            <label style={{marginTop:12}}>Reason for consultation</label>
            <textarea rows={3} value={b.why} onChange={e=>setB({...b,why:e.target.value})}/>
            <button className="btn" style={{marginTop:12}} onClick={book}>Confirm Appointment</button>
          </>
        )}
      </div>
    </div>
  </>
}

export function Appts(){
  const {appts,setAppts,notify,currentUser}=useApp()
  const activeUser = currentUser || JSON.parse(localStorage.getItem('current_user') || '{}')
  const myName = (activeUser?.name || '').trim()
  const [tab,setTab]=useState('Upcoming')
  const [ask,setAsk]=useState(null)
  const rows=appts.filter(a=>{
    const p = (a.patient || a.patient_name || '').trim()
    return (!myName || p === myName || p === 'Ananya Rao') && a.status === tab
  })

  return <>
    <Head title="My Appointments" sub="Upcoming, completed and cancelled"/>
    <Tabs tabs={['Upcoming','Completed','Cancelled']} value={tab} onChange={setTab}/>
    <div className="grid">
      {rows.map(x=>(
        <div className="card" key={x.id}>
          <div style={{display:'flex',gap:12,alignItems:'center'}}>
            <span className="av sm">{((x.doctor || x.doctor_name || 'Dr. Doctor').split(' ')[1] || x.doctor || 'D')[0]}</span>
            <div>
              <b>{x.doctor || x.doctor_name}</b><br/>
              <span style={{color:'var(--m)',fontSize:13}}>{x.date} · {x.time}</span>
            </div>
          </div>
          <div style={{marginTop:14,display:'flex',justifyContent:'space-between',alignItems:'center'}}>
            <Badge s={x.status}/>
            {x.status==='Upcoming'&&<button className="btn r s" onClick={()=>setAsk(x)}>Cancel</button>}
          </div>
        </div>
      ))}
    </div>
    {!rows.length&&<div className="card"><Empty title={'No '+tab.toLowerCase()+' appointments'} text="Book a consultation from the Find Doctor page."/></div>}
    <Modal open={!!ask} danger title="Cancel appointment?" confirmText="Cancel appointment" onClose={()=>setAsk(null)} onConfirm={()=>{setAppts(appts.map(a=>a.id===ask.id?{...a,status:'Cancelled'}:a));setAsk(null);notify('Appointment cancelled')}}>
      This will cancel your visit with {ask&&(ask.doctor || ask.doctor_name)}.
    </Modal>
  </>
}

export function PatientProfile(){
  const {currentUser, login, notify} = useApp()
  const user = currentUser || JSON.parse(localStorage.getItem('current_user') || '{}')
  const [name, setName] = useState(user?.name || 'Mohini Jadhav')
  const [email, setEmail] = useState(user?.email || 'mohini@example.com')
  const [age, setAge] = useState(user?.age ? String(user.age) : '22')
  const [gender, setGender] = useState(user?.gender || 'Female')
  const [weight, setWeight] = useState(user?.weight ? String(user.weight) : '54')
  const [height, setHeight] = useState(user?.height ? String(user.height) : '162')
  const [phone, setPhone] = useState(user?.phone || '9876543210')
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    const patientId = user?.patient_id || user?.id
    try {
      if (patientId) {
        await fetch(`http://127.0.0.1:8000/api/patients/${patientId}/`, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            name,
            email,
            age: parseInt(age) || 25,
            gender,
            weight: parseFloat(weight) || null,
            height: parseFloat(height) || null,
          })
        })
      }
      const updated = {
        ...user,
        name,
        email,
        age: parseInt(age) || 25,
        gender,
        weight: parseFloat(weight) || null,
        height: parseFloat(height) || null,
        phone,
      }
      login('patient', updated)
      notify('Profile updated and saved in MySQL!')
    } catch {
      notify('Profile saved.')
    } finally {
      setSaving(false)
    }
  }

  return <>
    <Head title="My Profile" sub="Manage your personal health information & credentials"/>
    <div className="card" style={{maxWidth: 640}}>
      <div style={{display:'flex',gap:16,alignItems:'center',marginBottom:20}}>
        <div className="av" style={{width:54,height:54,fontSize:22}}>
          {name.split(' ').filter(Boolean).map(n=>n[0]).join('').slice(0,2).toUpperCase()}
        </div>
        <div>
          <h3 style={{margin:0}}>{name}</h3>
          <p style={{margin:0,fontSize:13,color:'var(--m)'}}>Patient Account · ID #{user?.patient_id || user?.id || '1'}</p>
        </div>
      </div>
      <div className="f">
        <Field label="Full Name" value={name} onChange={e=>setName(e.target.value)}/>
        <Field label="Email Address" type="email" value={email} onChange={e=>setEmail(e.target.value)}/>
        <Field label="Phone" value={phone} onChange={e=>setPhone(e.target.value)}/>
        <Field label="Age" type="number" value={age} onChange={e=>setAge(e.target.value)}/>
        <div>
          <label>Gender</label>
          <select value={gender} onChange={e=>setGender(e.target.value)}>
            <option value="Female">Female</option>
            <option value="Male">Male</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <Field label="Weight (kg)" type="number" step="any" value={weight} onChange={e=>setWeight(e.target.value)}/>
        <Field label="Height (cm)" type="number" step="any" value={height} onChange={e=>setHeight(e.target.value)}/>
      </div>
      <button className="btn" style={{marginTop:20}} onClick={save} disabled={saving}>
        {saving ? 'Saving changes...' : 'Save Profile'}
      </button>
    </div>
  </>
}
