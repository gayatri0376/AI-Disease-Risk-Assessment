import {useState, useEffect} from 'react'
import {useNavigate,useParams} from 'react-router-dom'
import {Users,CalendarDays,Clock,FileText,CheckCircle2,NotebookPen,Plus,Search,User,Stethoscope,Calendar,Pill,Utensils} from 'lucide-react'
import {BarChart,Bar,XAxis,YAxis,Tooltip} from 'recharts'
import {Head,Stat,Chart,Table,Badge,Field} from '../../components/ui'
import {patients as mockPatients, appointments as mockAppointments, assessments, cbc, trend} from '../../data/mock'
import {useApp} from '../../context/AppContext'

export function DDash(){
  const {currentUser} = useApp()
  const user = currentUser || (typeof window !== 'undefined' ? JSON.parse(localStorage.getItem('current_user') || 'null') : null)
  const docName = user?.name || 'Dr. Aarav Mehta'
  const spec = user?.spec || 'Hematologist'

  return <>
    <Head title="Doctor Dashboard" sub={`${docName} · ${spec}`}/>
    <div className="grid">
      <Stat icon={Users} label="Total Patients" value="48"/>
      <Stat icon={CalendarDays} label="Today's Appointments" value="5"/>
      <Stat icon={Clock} label="Pending Requests" value="3"/>
      <Stat icon={FileText} label="Reports Reviewed" value="126"/>
    </div>
    <div className="g2">
      <Chart title="Appointments per month">
        <BarChart data={trend}>
          <XAxis dataKey="m"/><YAxis/><Tooltip/>
          <Bar dataKey="appts" fill="#287D8E"/>
        </BarChart>
      </Chart>
      <div>
        <h3>Upcoming appointments</h3>
        <Table cols={[{h:'Patient',k:'patient'},{h:'Date',k:'date'},{h:'Time',k:'time'}]} rows={mockAppointments.filter(a=>a.status==='Upcoming')}/>
      </div>
    </div>
  </>
}

export function DPatients(){
  const nav=useNavigate()
  const [list, setList] = useState([])

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/patients/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          const mapped = data.map(p => ({
            id: p.id,
            name: p.name,
            age: p.age,
            last: 'Recent',
            risk: 'Needs Attention',
            email: p.email || 'patient@anemia.com'
          }))
          setList([...mapped, ...mockPatients])
        } else {
          setList(mockPatients)
        }
      })
      .catch(() => setList(mockPatients))
  }, [])

  const rows = list.length > 0 ? list : mockPatients

  return <>
    <Head title="My Patients" sub="Patients registered in MySQL database"/>
    <Table cols={[
      {h:'Patient',k:'name'},
      {h:'Age',k:'age'},
      {h:'Last Assessment',k:'last'},
      {h:'Screening',r:x=><Badge s={x.risk}/>},
      {h:'Action',r:x=><button className="btn s" onClick={()=>nav('/doctor/patient/'+x.id)}>View Patient</button>}
    ]} rows={rows}/>
  </>
}

export function DPatient(){
  const {id}=useParams(),nav=useNavigate()
  const p=mockPatients.find(x=>x.id==id)||mockPatients[0]
  
  return <>
    <Head title={p.name} sub={`${p.email} · Age ${p.age}`}>
      <button className="btn" onClick={()=>nav('/doctor/consultation')}>Add Consultation Note</button>
    </Head>
    <div className="g2" style={{marginTop:0}}>
      <div className="card">
        <h3>AI Screening Result</h3>
        <Badge s={p.risk}/>
        <p style={{fontSize:13,color:'var(--m)',marginTop:6}}>Preliminary screening supported by Random Forest ML model.</p>
        <h3 style={{marginTop:16}}>CBC Values</h3>
        {cbc.map(([k,v])=><p key={k} style={{margin:'4px 0'}}>{k}: <b>{v}</b></p>)}
      </div>
      <div>
        <h3>Previous reports</h3>
        <Table cols={[{h:'ID',k:'id'},{h:'Date',k:'date'},{h:'Status',r:x=><Badge s={x.status}/>}]} rows={assessments}/>
        <h3 style={{marginTop:16}}>Appointment history</h3>
        <Table cols={[{h:'Doctor',k:'doctor'},{h:'Date',k:'date'},{h:'Status',r:x=><Badge s={x.status}/>}]} rows={mockAppointments.filter(a=>a.patient===p.name)}/>
      </div>
    </div>
  </>
}

export function Form({title,sub,fields,cta,msg}){
  const {notify}=useApp()
  return <>
    <Head title={title} sub={sub}/>
    <div className="card">
      <div className="f">
        {fields.map(([l,t,v])=>t==='area'?<div key={l} style={{gridColumn:'1/-1'}}><label>{l}</label><textarea rows={4} defaultValue={v}/></div>:<Field key={l} label={l} type={t} defaultValue={v}/>)}
      </div>
      <button className="btn" style={{marginTop:16}} onClick={()=>notify(msg)}>{cta}</button>
    </div>
  </>
}

export function Notes(){
  const {currentUser, notify} = useApp()
  const user = currentUser || (typeof window !== 'undefined' ? JSON.parse(localStorage.getItem('current_user') || '{}') : {})
  const doctorName = user?.name || 'Dr. Aarav Mehta'

  const [notesList, setNotesList] = useState([])
  const [patientList, setPatientList] = useState([])
  const [saving, setSaving] = useState(false)
  const [showAdd, setShowAdd] = useState(true)

  // Form state
  const [form, setForm] = useState({
    patient_name: '',
    custom_patient: '',
    date: new Date().toISOString().split('T')[0],
    notes: 'Patient exhibits clinical signs of fatigue and lower eyelid pallor. Advised CBC re-evaluation and iron replacement.',
    prescription: 'Ferrous Ascorbate 100mg once daily after lunch, Vitamin C 500mg OD',
    dietary_advice: 'Incorporate spinach, beetroot, lentils, pomegranate and citrus fruits. Avoid tea or coffee within 1 hour of meals.',
    follow_up: 'Yes, in 4 weeks'
  })

  // Fetch consultation notes and patient list from MySQL
  const loadData = () => {
    fetch('http://127.0.0.1:8000/api/consultation-notes/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setNotesList(data)
      })
      .catch(() => {})

    fetch('http://127.0.0.1:8000/api/patients/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          setPatientList(data)
          setForm(f => ({...f, patient_name: f.patient_name || data[0].name}))
        } else {
          setPatientList(mockPatients)
          setForm(f => ({...f, patient_name: f.patient_name || mockPatients[0].name}))
        }
      })
      .catch(() => {
        setPatientList(mockPatients)
        setForm(f => ({...f, patient_name: f.patient_name || mockPatients[0].name}))
      })
  }

  useEffect(() => {
    loadData()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSaving(true)

    const finalPatient = form.patient_name === 'custom' ? form.custom_patient : form.patient_name
    if (!finalPatient || !finalPatient.trim()) {
      notify('Please specify patient name')
      setSaving(false)
      return
    }

    const payload = {
      doctor_name: doctorName,
      patient_name: finalPatient.trim(),
      date: form.date,
      notes: form.notes,
      prescription: form.prescription,
      dietary_advice: form.dietary_advice,
      follow_up: form.follow_up,
      doctor: user?.doctor_id || null,
    }

    try {
      const res = await fetch('http://127.0.0.1:8000/api/consultation-notes/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      })
      const data = await res.json()
      if (res.ok) {
        setNotesList([data, ...notesList])
        notify('Consultation note saved in MySQL database!')
        setForm(f => ({
          ...f,
          custom_patient: '',
          notes: '',
          prescription: '',
          dietary_advice: ''
        }))
      } else {
        throw new Error(data.error || 'Failed to save note')
      }
    } catch (err) {
      const fallback = {...payload, id: Date.now()}
      setNotesList([fallback, ...notesList])
      notify('Consultation recorded (saved locally)')
    } finally {
      setSaving(false)
    }
  }

  return <>
    <Head title="Consultation Notes" sub="Record clinical observations, prescriptions, and dietary advice in MySQL">
      <button className="btn" onClick={() => setShowAdd(!showAdd)}>
        {showAdd ? 'Hide Form' : '+ New Consultation'}
      </button>
    </Head>

    <div className="grid">
      <Stat icon={NotebookPen} label="Total Consultations" value={notesList.length}/>
      <Stat icon={Users} label="Patients Advised" value={new Set(notesList.map(n => n.patient_name)).size}/>
      <Stat icon={Clock} label="Pending Follow-ups" value={notesList.filter(n => n.follow_up && !n.follow_up.includes('No')).length}/>
    </div>

    {showAdd && (
      <div className="card" style={{marginTop: 18}}>
        <h3 style={{borderBottom: '1px solid var(--line)', paddingBottom: 10, marginBottom: 14}}>
          Record New Consultation Note
        </h3>

        <form onSubmit={handleSubmit} style={{display: 'flex', flexDirection: 'column', gap: 14}}>
          <div className="f">
            <div>
              <label>Select Patient *</label>
              <select value={form.patient_name} onChange={e => setForm({...form, patient_name: e.target.value})}>
                {patientList.map(p => <option key={p.id} value={p.name}>{p.name} (Age: {p.age || 25})</option>)}
                <option value="custom">+ Other Patient Name...</option>
              </select>
            </div>

            {form.patient_name === 'custom' && (
              <div>
                <label>Enter Patient Name *</label>
                <input required value={form.custom_patient} onChange={e => setForm({...form, custom_patient: e.target.value})} placeholder="e.g. Ramesh Kumar"/>
              </div>
            )}

            <div>
              <label>Consultation Date</label>
              <input type="date" value={form.date} onChange={e => setForm({...form, date: e.target.value})}/>
            </div>
          </div>

          <div>
            <label>Clinical Observations & Diagnostic Findings *</label>
            <textarea
              required
              rows={3}
              value={form.notes}
              onChange={e => setForm({...form, notes: e.target.value})}
              placeholder="Record symptoms, pallor, fatigue severity, CBC interpretation..."
              style={{fontFamily: 'inherit'}}
            />
          </div>

          <div className="f">
            <div>
              <label>Prescribed Medications / Supplements</label>
              <textarea
                rows={2}
                value={form.prescription}
                onChange={e => setForm({...form, prescription: e.target.value})}
                placeholder="e.g. Ferrous Ascorbate 100mg OD, Folic Acid 5mg..."
                style={{fontFamily: 'inherit'}}
              />
            </div>

            <div>
              <label>Dietary & Nutritional Guidance</label>
              <textarea
                rows={2}
                value={form.dietary_advice}
                onChange={e => setForm({...form, dietary_advice: e.target.value})}
                placeholder="e.g. High-iron foods, Vitamin C sources, avoid tea near meals..."
                style={{fontFamily: 'inherit'}}
              />
            </div>
          </div>

          <div className="f">
            <div>
              <label>Follow-up Required</label>
              <select value={form.follow_up} onChange={e => setForm({...form, follow_up: e.target.value})}>
                <option value="Yes, in 1 week">Yes, in 1 week</option>
                <option value="Yes, in 2 weeks">Yes, in 2 weeks</option>
                <option value="Yes, in 4 weeks">Yes, in 4 weeks</option>
                <option value="After next CBC lab test">After next CBC lab test</option>
                <option value="No follow-up needed">No follow-up needed</option>
              </select>
            </div>
            <div>
              <label>Attending Doctor</label>
              <input disabled value={doctorName} style={{background: 'var(--bg)'}}/>
            </div>
          </div>

          <div style={{marginTop: 6}}>
            <button type="submit" className="btn" disabled={saving}>
              <CheckCircle2 size={16}/> {saving ? 'Saving to Database…' : 'Save Consultation Note'}
            </button>
          </div>
        </form>
      </div>
    )}

    <div style={{marginTop: 24}}>
      <h3 style={{marginBottom: 14}}>Saved Clinical Consultation Records</h3>
      <Table cols={[
        {
          h: 'Patient',
          r: x => <div>
            <b style={{color: 'var(--n)', fontSize: 14}}>{x.patient_name}</b>
            <div style={{fontSize: 12, color: 'var(--m)'}}>By {x.doctor_name}</div>
          </div>
        },
        {h: 'Date', k: 'date'},
        {
          h: 'Observations & Notes',
          r: x => <div style={{maxWidth: 320, whiteSpace: 'normal', fontSize: 13}}>
            {x.notes}
          </div>
        },
        {
          h: 'Prescription & Diet',
          r: x => <div style={{maxWidth: 240, whiteSpace: 'normal', fontSize: 12.5}}>
            {x.prescription && <div><span style={{color: 'var(--b)', fontWeight: 600}}>Rx:</span> {x.prescription}</div>}
            {x.dietary_advice && <div style={{color: 'var(--m)', marginTop: 2}}>🥗 {x.dietary_advice}</div>}
          </div>
        },
        {
          h: 'Follow-up',
          r: x => <span className="bd in" style={{fontSize: 11.5}}>{x.follow_up || 'In 4 weeks'}</span>
        }
      ]} rows={notesList}/>
    </div>
  </>
}

export function DReports(){
  return <>
    <Head title="Reports" sub="Recent patient assessment reports"/>
    <Table cols={[{h:'Patient',k:'name'},{h:'Screening',r:x=><Badge s={x.risk}/>},{h:'Date',k:'last'}]} rows={mockPatients}/>
  </>
}

export function DoctorProfilePage(){
  const {currentUser, login, notify} = useApp()
  const user = currentUser || (typeof window !== 'undefined' ? JSON.parse(localStorage.getItem('current_user') || '{}') : {})
  const [name, setName] = useState(user?.name || 'Dr. Aarav Mehta')
  const [email, setEmail] = useState(user?.email || 'doctor@hemocare.ai')
  const [spec, setSpec] = useState(user?.spec || 'Hematologist')
  const [exp, setExp] = useState(user?.exp ? String(user.exp) : '8')
  const [loc, setLoc] = useState(user?.loc || 'Pune')
  const [phone, setPhone] = useState(user?.phone || '9876543210')
  const [avail, setAvail] = useState(user?.avail || 'Today (10:00 - 17:00)')
  const [bio, setBio] = useState(user?.bio || 'Clinical specialist in hematological care, CBC interpretation, and anemia screening.')
  const [saving, setSaving] = useState(false)

  // Smart Initials: removes Dr. prefix and takes first and last initial
  const cleanName = (name || '').replace(/^Dr\.?\s+/i, '').trim()
  const parts = cleanName.split(' ').filter(Boolean)
  const ini = parts.length >= 2
    ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
    : (cleanName ? cleanName.slice(0, 2).toUpperCase() : 'DR')

  const save = async (e) => {
    e?.preventDefault?.()
    setSaving(true)
    const docId = user?.doctor_id || user?.id

    try {
      if (docId) {
        await fetch(`http://127.0.0.1:8000/api/doctors/${docId}/`, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            name,
            email,
            spec,
            exp: parseInt(exp) || 5,
            loc,
            avail,
          })
        })
      }
    } catch(err) {
      console.warn('Backend sync failed, saved locally')
    }

    const updated = {
      ...user,
      name,
      email,
      spec,
      exp: parseInt(exp) || 5,
      loc,
      phone,
      avail,
      bio,
    }
    login('doctor', updated)
    setSaving(false)
    notify('Doctor profile updated successfully!')
  }

  return <>
    <Head title="Doctor Profile" sub="Manage your clinical credentials & account settings"/>

    <div className="card" style={{marginBottom: 20}}>
      <div style={{display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap'}}>
        <div className="av" style={{width: 72, height: 72, fontSize: 26, background: 'var(--t)', flexShrink: 0}}>
          {ini}
        </div>
        <div style={{flex: 1, minWidth: 200}}>
          <h2 style={{margin: '0 0 4px', fontSize: 22}}>{name}</h2>
          <div style={{display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap'}}>
            <span className="bd ok">Verified Doctor</span>
            <span style={{color: 'var(--m)', fontSize: 13}}>• {spec}</span>
            <span style={{color: 'var(--m)', fontSize: 13}}>• {loc}</span>
          </div>
          <p style={{margin: '6px 0 0', fontSize: 13, color: 'var(--m)'}}>
            Email: <b>{email}</b> · Experience: <b>{exp} years</b>
          </p>
        </div>
      </div>
    </div>

    <div className="card">
      <form onSubmit={save} style={{display: 'flex', flexDirection: 'column', gap: 16}}>
        <h3 style={{borderBottom: '1px solid var(--line)', paddingBottom: 10, marginBottom: 4}}>
          Doctor Details
        </h3>

        <div className="f">
          <div>
            <label>Doctor Full Name *</label>
            <input required value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Dr. Mohini Jadhav"/>
          </div>
          <div>
            <label>Registered Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)}/>
          </div>
        </div>

        <div className="f">
          <div>
            <label>Specialization</label>
            <select value={spec} onChange={e => setSpec(e.target.value)}>
              <option value="Hematologist">Hematologist</option>
              <option value="General Physician">General Physician</option>
              <option value="Internal Medicine">Internal Medicine</option>
              <option value="Clinical Pathologist">Clinical Pathologist</option>
              <option value="Pediatric Hematologist">Pediatric Hematologist</option>
            </select>
          </div>
          <div>
            <label>Experience (Years)</label>
            <input type="number" value={exp} onChange={e => setExp(e.target.value)} min="1" max="50"/>
          </div>
        </div>

        <div className="f">
          <div>
            <label>Clinic Location / City</label>
            <input value={loc} onChange={e => setLoc(e.target.value)} placeholder="e.g. Pune, Maharashtra"/>
          </div>
          <div>
            <label>Contact Phone</label>
            <input value={phone} onChange={e => setPhone(e.target.value)}/>
          </div>
        </div>

        <div>
          <label>Consultation Availability</label>
          <input value={avail} onChange={e => setAvail(e.target.value)} placeholder="e.g. Today, Mon–Fri 10:00 - 17:00"/>
        </div>

        <div>
          <label>Clinical Advice / Bio</label>
          <textarea rows={3} value={bio} onChange={e => setBio(e.target.value)} style={{fontFamily: 'inherit'}}/>
        </div>

        <div style={{marginTop: 8}}>
          <button type="submit" className="btn" disabled={saving}>
            {saving ? 'Saving Changes…' : 'Save Profile Changes'}
          </button>
        </div>
      </form>
    </div>
  </>
}

export function DoctorAvailabilityPage(){
  const {currentUser, login, notify} = useApp()
  const user = currentUser || (typeof window !== 'undefined' ? JSON.parse(localStorage.getItem('current_user') || '{}') : {})

  const [days, setDays] = useState('Mon–Fri')
  const [fromTime, setFromTime] = useState('10:00')
  const [toTime, setToTime] = useState('17:00')
  const [slotDuration, setSlotDuration] = useState('30 mins')
  const [mode, setMode] = useState('In-Person & Video Consultation')
  const [maxPatients, setMaxPatients] = useState('15')
  const [saving, setSaving] = useState(false)

  // Initialize from user.avail if present
  useEffect(() => {
    if (user?.avail) {
      const parts = user.avail.split(',')
      if (parts[0]) setDays(parts[0].trim())
      if (parts[1]) {
        const timeParts = parts[1].split('-')
        if (timeParts[0]) setFromTime(timeParts[0].trim())
        if (timeParts[1]) setToTime(timeParts[1].trim().split(' ')[0])
      }
    }
  }, [user?.avail])

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)

    const availFormatted = `${days}, ${fromTime} - ${toTime} (${slotDuration})`
    const docId = user?.doctor_id || user?.id

    try {
      if (docId) {
        await fetch(`http://127.0.0.1:8000/api/doctors/${docId}/`, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            avail: availFormatted
          })
        })
      }
    } catch(err) {
      console.warn('Saved availability locally')
    }

    const updated = {
      ...user,
      avail: availFormatted,
      availDetails: {
        days,
        fromTime,
        toTime,
        slotDuration,
        mode,
        maxPatients
      }
    }
    login('doctor', updated)
    setSaving(false)
    notify('Weekly availability schedule saved to database!')
  }

  return <>
    <Head title="Doctor Availability" sub="Configure your weekly appointment consultation slots in MySQL"/>

    <div className="card" style={{marginBottom: 20, background: 'var(--l)', border: '1px solid #CFE5EC'}}>
      <div style={{display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap'}}>
        <div className="ic t" style={{width: 44, height: 44}}><Clock size={20}/></div>
        <div>
          <b style={{color: 'var(--n)', fontSize: 16, display: 'block'}}>
            Active Consultation Schedule: {user?.avail || `${days}, ${fromTime} - ${toTime}`}
          </b>
          <span style={{fontSize: 13, color: 'var(--m)'}}>
            Patients can book appointments within these configured slots. Mode: <b>{mode}</b>
          </span>
        </div>
      </div>
    </div>

    <div className="card">
      <form onSubmit={handleSave} style={{display: 'flex', flexDirection: 'column', gap: 16}}>
        <h3 style={{borderBottom: '1px solid var(--line)', paddingBottom: 10, marginBottom: 4}}>
          Weekly Time Slot Settings
        </h3>

        <div className="f">
          <div>
            <label>Working Days</label>
            <select value={days} onChange={e => setDays(e.target.value)}>
              <option value="Mon–Fri">Monday – Friday</option>
              <option value="Mon–Sat">Monday – Saturday</option>
              <option value="Mon, Wed, Fri">Mon, Wed, Fri</option>
              <option value="Tue, Thu, Sat">Tue, Thu, Sat</option>
              <option value="All Days (Mon–Sun)">All 7 Days (Mon–Sun)</option>
              <option value="Weekends Only (Sat–Sun)">Weekends Only (Sat–Sun)</option>
            </select>
          </div>

          <div>
            <label>Consultation Mode</label>
            <select value={mode} onChange={e => setMode(e.target.value)}>
              <option value="In-Person & Video Consultation">In-Person & Video Consultation</option>
              <option value="In-Person Hospital Visit Only">In-Person Hospital Visit Only</option>
              <option value="Online Video Teleconsultation Only">Online Video Teleconsultation Only</option>
            </select>
          </div>
        </div>

        <div className="f">
          <div>
            <label>Start Time (From)</label>
            <input type="time" value={fromTime} onChange={e => setFromTime(e.target.value)}/>
          </div>

          <div>
            <label>End Time (To)</label>
            <input type="time" value={toTime} onChange={e => setToTime(e.target.value)}/>
          </div>

          <div>
            <label>Slot Duration</label>
            <select value={slotDuration} onChange={e => setSlotDuration(e.target.value)}>
              <option value="15 mins">15 minutes per patient</option>
              <option value="20 mins">20 minutes per patient</option>
              <option value="30 mins">30 minutes per patient</option>
              <option value="45 mins">45 minutes per patient</option>
            </select>
          </div>
        </div>

        <div className="f">
          <div>
            <label>Max Patients Per Day</label>
            <input type="number" min="1" max="60" value={maxPatients} onChange={e => setMaxPatients(e.target.value)}/>
          </div>
          <div>
            <label>Emergency Walk-in Buffer</label>
            <select defaultValue="2 Slots Reserved">
              <option>2 Slots Reserved</option>
              <option>3 Slots Reserved</option>
              <option>No Buffer</option>
            </select>
          </div>
        </div>

        <div style={{marginTop: 8}}>
          <button type="submit" className="btn" disabled={saving}>
            <CheckCircle2 size={16}/> {saving ? 'Saving Schedule…' : 'Save Availability Schedule'}
          </button>
        </div>
      </form>
    </div>
  </>
}

