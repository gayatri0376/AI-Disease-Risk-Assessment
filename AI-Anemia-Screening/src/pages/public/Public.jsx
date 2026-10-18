import {useState} from 'react'
import {Link,useNavigate} from 'react-router-dom'
import {Droplet} from 'lucide-react'
import {useApp} from '../../context/AppContext'
import {Disclaimer,Field} from '../../components/ui'
const Nav=()=><header className="pub"><Link to="/" style={{fontWeight:700,fontSize:19,color:'var(--n)',display:'flex',gap:6,alignItems:'center'}}><Droplet/>HemoCare AI</Link>
<nav>{[['/about','About'],['/how-it-works','How It Works'],['/education','Health Education'],['/login','Login']].map(([p,l])=><Link key={p} to={p}>{l}</Link>)}</nav></header>
const Foot=()=><footer className="sec" style={{textAlign:'center'}}><Disclaimer/><p>© 2026 HemoCare AI · College major project demo</p></footer>
const Cards=({items})=><div className="grid">{items.map(([t,d])=><div className="card" key={t}><h3 style={{fontSize:17}}>{t}</h3><p>{d}</p></div>)}</div>
const STEPS=[['Register','Create your free patient account.'],['Enter Health Information','Share symptoms and basic details.'],['Submit CBC Values','Add your lab report numbers.'],['AI Screening','A demo model estimates a risk indication.'],['View Result','See your report and next action.'],['Connect With Doctor','Book a consultation with a verified doctor.']]
const EDU=[['What is Anemia?','A condition where the body has fewer healthy red blood cells than usual.'],['Common Symptoms','Tiredness, weakness, dizziness, headache, pale skin, shortness of breath.'],['Possible Risk Factors','Diet, chronic conditions, blood loss, pregnancy, family history.'],['CBC Basics','A complete blood count measures hemoglobin, red cells, white cells and platelets.'],['When to Consult a Doctor','If symptoms persist or your lab values concern you, talk to a qualified doctor.']]
export function Landing(){return <><Nav/>
<section className="hero"><h1>HemoCare AI</h1><h3 style={{color:'#9fd3dc'}}>AI-Powered Anemia Screening & Connected Care</h3>
<p>Early Screening. Better Awareness. Connected Care. Get preliminary screening support and connect with doctors.</p>
<Link to="/register" className="btn" style={{background:'var(--t)'}}>Get Started</Link><Link to="/login" className="btn o">Login</Link></section>
<div className="sec"><Disclaimer/><h2 style={{marginTop:30}}>How HemoCare AI works</h2><Cards items={STEPS}/>
<h2 style={{marginTop:30}}>Key features</h2><Cards items={[['AI Screening','Preliminary risk indication from your inputs.'],['Doctor Consultation','Find and book verified doctors.'],['Reports','Keep every screening report in one place.'],['Health Education','Learn the basics in plain language.']]}/>
<h2 style={{marginTop:30}}>FAQ</h2><Cards items={[['Is this a diagnosis?','No. It is a demo screening and not a medical diagnosis.'],['Is my data real?','No. All data in this prototype is mock data.']]}/></div><Foot/></>}
export function Info({kind}){const T={about:['About HemoCare AI',[['The problem','Anemia often goes unnoticed because symptoms are subtle and lab reports are hard to read.'],['Our solution','A platform that screens early using health inputs and CBC values.'],['AI screening','A demo model returns a preliminary risk indication.'],['Doctor consultation','Users can share reports with doctors and book appointments.'],['Benefits','Awareness, faster follow-up and connected care.']]],how:['How It Works',STEPS.map(([a,b],i)=>['Step '+(i+1)+': '+a,b])],edu:['Health Education',EDU]}[kind]
return <><Nav/><div className="sec"><h1>{T[0]}</h1><Cards items={T[1]}/></div><Foot/></>}
export function Auth({reg}){
  const {login,notify}=useApp()
  const nav=useNavigate()
  const [role,setRole]=useState('patient')
  const [name, setName]=useState('Mohini Jadhav')
  const [email, setEmail]=useState('mohini@example.com')
  const [age, setAge]=useState('22')
  const [gender, setGender]=useState('Female')
  const [phone, setPhone]=useState('9876543210')
  const [password, setPassword]=useState('demo1234')
  const [spec, setSpec]=useState('Hematologist')
  const [exp, setExp]=useState('8')
  const [loc, setLoc]=useState('Pune')
  const [busy, setBusy]=useState(false)
  const [err, setErr]=useState('')

  const handleRoleChange = (newRole) => {
    setRole(newRole)
    if (reg) {
      if (newRole === 'doctor') {
        if (!name.toLowerCase().startsWith('dr.')) {
          setName('Dr. ' + (name.replace(/^Dr\.?\s+/i, '') || 'Mohini Jadhav'))
        }
        if (email === 'mohini@example.com') setEmail('dr.mohini@hospital.com')
      } else if (newRole === 'patient') {
        setName(name.replace(/^Dr\.?\s+/i, ''))
        if (email === 'dr.mohini@hospital.com') setEmail('mohini@example.com')
      }
    }
  }

  const handleAuth = async () => {
    setBusy(true)
    setErr('')
    try {
      if (reg) {
        let finalName = name.trim() || (role === 'doctor' ? 'Dr. Doctor' : 'Demo User')
        if (role === 'doctor' && !finalName.toLowerCase().startsWith('dr.') && !finalName.toLowerCase().startsWith('dr ')) {
          finalName = 'Dr. ' + finalName
        }

        const payload = {
          name: finalName,
          email: email.trim().toLowerCase(),
          password: password,
          role: role,
          age: parseInt(age) || 25,
          gender: gender || 'Female',
          phone: phone || null,
          spec: spec || 'Hematologist',
          exp: parseInt(exp) || 5,
          loc: loc || 'Pune',
        }

        const res = await fetch('http://127.0.0.1:8000/api/auth/register/', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        })
        const data = await res.json()
        if (!res.ok) {
          throw new Error(data.error || 'Failed to register account.')
        }
        login(role, data.user)
        notify(`Registration successful! Welcome, ${data.user.name}.`)
        nav(`/${role}/dashboard`)
      } else {
        // Login API Call
        const res = await fetch('http://127.0.0.1:8000/api/auth/login/', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            email: email.trim().toLowerCase(),
            password: password,
            role: role,
          })
        })
        const data = await res.json()
        if (!res.ok) {
          throw new Error(data.error || 'Invalid email or password.')
        }
        login(role, data.user)
        notify(`Welcome back, ${data.user.name}!`)
        nav(`/${role}/dashboard`)
      }
    } catch (e) {
      setBusy(false)
      setErr(e.message || 'Could not connect to Django backend. Check if server is running.')
    }
  }

  const go = async (r) => {
    const demoEmail = r === 'patient' ? 'demo@hemocare.ai' : (r === 'doctor' ? 'doctor@hemocare.ai' : 'admin@hemocare.ai')
    try {
      const res = await fetch('http://127.0.0.1:8000/api/auth/login/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ email: demoEmail, password: 'demo1234', role: r })
      })
      const data = await res.json()
      if (res.ok && data.user) {
        login(r, data.user)
      } else {
        login(r, { name: r === 'doctor' ? 'Dr. Aarav Mehta' : (r === 'admin' ? 'Admin User' : 'Demo Patient'), role: r })
      }
    } catch {
      login(r, { name: r === 'doctor' ? 'Dr. Aarav Mehta' : (r === 'admin' ? 'Admin User' : 'Demo Patient'), role: r })
    }
    notify(`Logged in as ${r}`)
    nav(`/${r}/dashboard`)
  }

  return (
    <>
      <Nav/>
      <div className="sec" style={{maxWidth:480}}>
        <div className="card">
          <h2>{reg?'Create Account':'Welcome Back'}</h2>
          
          {err && (
            <div style={{background:'#fee2e2',color:'#b91c1c',padding:10,borderRadius:8,marginBottom:12,fontSize:13}}>
              <b>Error:</b> {err}
            </div>
          )}

          <div className="f" style={{gridTemplateColumns:'1fr'}}>
            <div>
              <label>Role</label>
              <select value={role} onChange={e=>handleRoleChange(e.target.value)}>
                <option value="patient">Patient (Anemia Screening)</option>
                <option value="doctor">Doctor (Clinical Portal)</option>
                <option value="admin">Administrator</option>
              </select>
            </div>

            {reg && (
              <Field
                label={role === 'doctor' ? 'Doctor Full Name' : 'Full Name'}
                value={name}
                onChange={e=>setName(e.target.value)}
                placeholder={role === 'doctor' ? 'e.g. Dr. Mohini Jadhav' : 'e.g. Mohini Jadhav'}
              />
            )}

            <Field label="Email" type="email" value={email} onChange={e=>setEmail(e.target.value)} />
            
            {reg && role === 'patient' && (
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
                <Field label="Age" type="number" value={age} onChange={e=>setAge(e.target.value)} />
                <div>
                  <label>Gender</label>
                  <select value={gender} onChange={e=>setGender(e.target.value)}>
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
            )}

            {reg && role === 'doctor' && (
              <>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
                  <div>
                    <label>Specialization</label>
                    <select value={spec} onChange={e=>setSpec(e.target.value)}>
                      <option value="Hematologist">Hematologist</option>
                      <option value="General Physician">General Physician</option>
                      <option value="Internal Medicine">Internal Medicine</option>
                      <option value="Clinical Pathologist">Clinical Pathologist</option>
                      <option value="Pediatric Hematologist">Pediatric Hematologist</option>
                    </select>
                  </div>
                  <Field label="Experience (Yrs)" type="number" value={exp} onChange={e=>setExp(e.target.value)} />
                </div>
                <Field label="Clinic City / Location" value={loc} onChange={e=>setLoc(e.target.value)} placeholder="e.g. Pune, Maharashtra" />
              </>
            )}

            {reg && <Field label="Phone" value={phone} onChange={e=>setPhone(e.target.value)} />}
            <Field label="Password" type="password" value={password} onChange={e=>setPassword(e.target.value)} />

            <button className="btn" onClick={handleAuth} disabled={busy} style={{marginTop:8}}>
              {busy ? 'Saving to Database…' : (reg ? 'Register' : 'Login')}
            </button>
          </div>

          <p style={{marginTop:16}}>Quick demo login:</p>
          {['patient','doctor','admin'].map(r=><button key={r} className="btn o s" style={{marginRight:6}} onClick={()=>go(r)}>{r}</button>)}
          <p style={{marginTop:14}}>
            <Link to={reg?'/login':'/register'} style={{color:'var(--b)'}}>
              {reg?'Have an account? Login':'New here? Register'}
            </Link>
          </p>
        </div>
      </div>
      <Foot/>
    </>
  )
}

