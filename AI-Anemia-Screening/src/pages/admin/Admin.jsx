import {useState, useEffect} from 'react'
import {Users,Stethoscope,ShieldCheck,ClipboardList,CalendarDays,BookOpen,Eye,Edit2,Trash2,Plus,Search,FileText,X,Clock,User,CheckCircle2} from 'lucide-react'
import {LineChart,Line,BarChart,Bar,PieChart,Pie,Cell,XAxis,YAxis,Tooltip,Legend} from 'recharts'
import {Head,Stat,Chart,Table,Badge,Field,Modal} from '../../components/ui'
import {patients as mockPatients, appointments as mockAppointments, assessments, trend, dist, articles} from '../../data/mock'
import {useApp} from '../../context/AppContext'

const COL=['#16A34A','#F59E0B','#DC2626']
const Pie3=()=><Chart title="Screening result distribution"><PieChart><Pie data={dist} dataKey="value" nameKey="name" outerRadius={80}>{dist.map((_,i)=><Cell key={i} fill={COL[i]}/>)}</Pie><Tooltip/><Legend/></PieChart></Chart>

export function ADash(){
  const {docs}=useApp()
  return <>
    <Head title="Admin Dashboard" sub="Platform Overview & Clinical Metrics"/>
    <div className="grid">
      <Stat icon={Users} label="Total Patients" value="96"/>
      <Stat icon={Stethoscope} label="Verified Doctors" value={docs.filter(d=>d.status==='Verified').length}/>
      <Stat icon={ShieldCheck} label="Pending Doctors" value={docs.filter(d=>d.status==='Pending').length}/>
      <Stat icon={ClipboardList} label="Total Assessments" value="150"/>
      <Stat icon={CalendarDays} label="Appointments" value="63"/>
    </div>
    <div className="g2">
      <Chart title="Patient registration trend">
        <LineChart data={trend}><XAxis dataKey="m"/><YAxis/><Tooltip/><Line dataKey="patients" stroke="#2463A6" strokeWidth={2}/></LineChart>
      </Chart>
      <Chart title="Assessments over time">
        <BarChart data={trend}><XAxis dataKey="m"/><YAxis/><Tooltip/><Bar dataKey="assess" fill="#287D8E"/></BarChart>
      </Chart>
      <Pie3/>
      <Chart title="Appointments">
        <BarChart data={trend}><XAxis dataKey="m"/><YAxis/><Tooltip/><Bar dataKey="appts" fill="#123D66"/></BarChart>
      </Chart>
    </div>
  </>
}

export function APatients(){
  const [rows,setRows]=useState(mockPatients)
  const [ask,setAsk]=useState(null)
  const {notify}=useApp()

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/patients/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          const mapped = data.map(p => ({
            id: p.id,
            name: p.name,
            email: p.email || 'patient@anemia.com',
            age: p.age,
            joined: '2026-10-15',
            count: 1,
            status: 'Active',
            risk: 'Needs Attention'
          }))
          setRows([...mapped, ...mockPatients])
        }
      })
      .catch(() => {})
  }, [])

  const flip=x=>{
    setRows(rows.map(p=>p.id===x.id?{...p,status:p.status==='Active'?'Inactive':'Active'}:p))
    setAsk(null)
    notify('Status updated')
  }

  return <>
    <Head title="Patients" sub="Manage registered patients in MySQL"/>
    <Table cols={[
      {h:'Name',k:'name'},
      {h:'Email',k:'email'},
      {h:'Age',k:'age'},
      {h:'Registered',k:'joined'},
      {h:'Assessments',k:'count'},
      {h:'Status',r:x=><Badge s={x.status}/>},
      {h:'Actions',r:x=><><button className="btn s o" onClick={()=>notify('Viewing '+x.name)}>View</button><button className="btn s r" onClick={()=>x.status==='Active'?setAsk(x):flip(x)}>{x.status==='Active'?'Deactivate':'Activate'}</button></>}
    ]} rows={rows}/>
    <Modal open={!!ask} danger title="Deactivate patient?" confirmText="Deactivate" onClose={()=>setAsk(null)} onConfirm={()=>flip(ask)}>
      {ask&&ask.name} will no longer be able to use the platform.
    </Modal>
  </>
}

export function ADoctors(){
  const [doctorList, setDoctorList] = useState([])
  const {docs} = useApp()

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/doctors/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) setDoctorList(data)
        else setDoctorList(docs)
      })
      .catch(() => setDoctorList(docs))
  }, [])

  const rows = doctorList.length > 0 ? doctorList : docs

  return <>
    <Head title="Doctors" sub="All registered doctors in MySQL"/>
    <Table cols={[
      {h:'Doctor',k:'name'},
      {h:'Email',r:x=>x.email || 'doctor@hemocare.ai'},
      {h:'Specialization',k:'spec'},
      {h:'Experience',r:x=>x.exp+' yrs'},
      {h:'Verification',r:x=><Badge s={x.status}/>},
      {h:'Patients',k:'patients_count'},
      {h:'Status',r:()=><Badge s="Active"/>}
    ]} rows={rows}/>
  </>
}

export function Verify(){
  const {docs,setDocs,notify}=useApp()
  const set=(id,s)=>{
    setDocs(docs.map(d=>d.id===id?{...d,status:s}:d))
    notify(`Doctor ${s.toLowerCase()}`)
  }

  return <>
    <Head title="Doctor Verification" sub="Review submitted doctor credentials"/>
    <Table cols={[
      {h:'Doctor',k:'name'},
      {h:'Specialization',k:'spec'},
      {h:'Submitted',r:()=>'2026-10-15'},
      {h:'Status',r:x=><Badge s={x.status}/>},
      {h:'Actions',r:x=>x.status==='Pending'?<><button className="btn s g" onClick={()=>set(x.id,'Verified')}>Approve</button><button className="btn s r" onClick={()=>set(x.id,'Rejected')}>Reject</button></>:'—'}
    ]} rows={docs.filter(d=>d.id>3||d.status!=='Verified')}/>
  </>
}

export function AAppts(){
  const [apptsList, setApptsList] = useState([])
  const {appts} = useApp()
  const [f,setF]=useState('All')

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/appointments/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          const mapped = data.map(a => ({
            id: a.id,
            patient: a.patient_name,
            doctor: a.doctor_name,
            date: a.date,
            time: a.time,
            status: a.status
          }))
          setApptsList([...mapped, ...appts])
        } else {
          setApptsList(appts)
        }
      })
      .catch(() => setApptsList(appts))
  }, [])

  const rows = apptsList.length > 0 ? apptsList : appts

  return <>
    <Head title="Appointments" sub="All recorded platform appointments">
      <select value={f} onChange={e=>setF(e.target.value)}>
        {['All','Upcoming','Completed','Cancelled'].map(s=><option key={s}>{s}</option>)}
      </select>
    </Head>
    <Table cols={[
      {h:'Patient',k:'patient'},
      {h:'Doctor',k:'doctor'},
      {h:'Date',k:'date'},
      {h:'Time',k:'time'},
      {h:'Status',r:x=><Badge s={x.status}/>}
    ]} rows={rows.filter(a=>f==='All'||a.status===f)}/>
  </>
}

export function AAnalytics(){
  return <>
    <Head title="Assessment Analytics" sub="Clinical distribution of platform screenings"/>
    <div className="grid">
      <Stat icon={ClipboardList} label="Total Assessments" value="150"/>
      <Stat icon={ShieldCheck} label="Low Risk (Healthy)" value="87"/>
      <Stat icon={ShieldCheck} label="Medium Risk" value="40"/>
      <Stat icon={ShieldCheck} label="Needs Attention" value="23"/>
    </div>
    <div className="g2">
      <Pie3/>
      <Chart title="Assessments over time">
        <BarChart data={trend}><XAxis dataKey="m"/><YAxis/><Tooltip/><Bar dataKey="assess" fill="#2463A6"/></BarChart>
      </Chart>
    </div>
  </>
}

export function AAssess(){
  const [list, setList] = useState([])

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/screening-results/')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          const mapped = data.map((r, i) => {
            let pName = 'Patient #' + (r.patient || (i + 1))
            let risk = r.prediction_label || 'Screened'
            let score = r.confidence_score ? Math.round(r.confidence_score * 100) : 85
            let date = r.created_at ? r.created_at.split('T')[0] : '2026-10-15'
            return {
              id: 'AS-' + (1000 + r.id),
              patient: pName,
              date: date,
              score: score,
              status: risk.includes('Healthy') ? 'Low Risk' : (risk.includes('High') ? 'Needs Attention' : 'Medium Risk')
            }
          })
          setList([...mapped, ...assessments.map(a => ({...a, patient: 'Patient User'}))])
        } else {
          setList(assessments.map(a => ({...a, patient: 'Patient User'})))
        }
      })
      .catch(() => {
        setList(assessments.map(a => ({...a, patient: 'Patient User'})))
      })
  }, [])

  const rows = list.length > 0 ? list : assessments

  return <>
    <Head title="Assessments" sub="All submitted AI screenings"/>
    <Table cols={[
      {h:'ID',k:'id'},
      {h:'Patient',k:'patient'},
      {h:'Date',k:'date'},
      {h:'Score',r:x=>x.score+'%'},
      {h:'Status',r:x=><Badge s={x.status}/>}
    ]} rows={rows}/>
  </>
}

const DEFAULT_ARTICLES = [
  {
    id: 1,
    title: 'What is Anemia? Causes, Symptoms & Types',
    cat: 'Basics',
    author: 'Dr. Aarav Mehta (Hematology Lead)',
    readTime: '4 min read',
    date: '2026-05-01',
    status: 'Published',
    summary: 'Anemia is a condition where your blood lacks enough healthy red blood cells or hemoglobin, leading to reduced oxygen flow to body tissues.',
    content: `Anemia is defined as a reduction in the total circulating red blood cell (RBC) mass or hemoglobin concentration below physiological values for age and sex. Hemoglobin is the iron-rich protein in red blood cells that transports oxygen from your lungs to tissues throughout your body.

Key Symptoms:
• Persistent fatigue, lethargy, and unexplained daytime drowsiness
• Pale, yellowish, or dull skin tone (inner eyelids, nail beds, lips)
• Shortness of breath (dyspnea) or dizziness upon minimal physical exertion
• Cold hands, feet, and heightened sensitivity to ambient cold
• Frequent headaches, lightheadedness, and heart palpitations (rapid or irregular heartbeats)

Major Types of Anemia:
1. Iron Deficiency Anemia (Microcytic Hypochromic): The most common nutritional disorder worldwide, caused by poor dietary intake, chronic occult blood loss, or impaired gut absorption.
2. Megaloblastic Anemia (Macrocytic): Triggered by deficiencies in Vitamin B12 (Cobalamin) or Folate (B9), leading to abnormally large, fragile red blood cells.
3. Hemolytic Anemia: Conditions where red blood cells are prematurely destroyed in the bloodstream faster than bone marrow can regenerate them.
4. Aplastic Anemia: Rare condition involving bone marrow stem cell depletion resulting in pancytopenia (deficiency of RBCs, WBCs, and platelets).
5. Anemia of Chronic Disease: Secondary to chronic inflammatory conditions, autoimmune disorders, or renal insufficiency.

When to Seek Clinical Care:
If you notice persistent fatigue alongside dizziness or paleness, an objective Complete Blood Count (CBC) is the first recommended clinical screening step.`
  },
  {
    id: 2,
    title: 'Understanding Your CBC (Complete Blood Count) Report',
    cat: 'Diagnostic Tests',
    author: 'Dr. Sana Iyer (Pathologist & Physician)',
    readTime: '5 min read',
    date: '2026-06-10',
    status: 'Published',
    summary: 'A complete blood count is the definitive first-line diagnostic test for blood disorders. Learn how to interpret critical parameters like HGB, MCV, MCH, and RDW.',
    content: `A Complete Blood Count (CBC) is an automated laboratory profile that evaluates cellular elements in peripheral blood. Modern hematology analyzers quantify 14+ essential parameters:

Key CBC Indices Explained:
• Hemoglobin (HGB): Normal reference is 13.8–17.2 g/dL (adult males) and 12.1–15.1 g/dL (adult females). Values below 12 g/dL indicate anemia.
• Red Blood Cell Count (RBC): Normal: 4.5–5.9 M/μL. Quantifies circulating erythrocytes.
• Hematocrit (HCT): The volumetric percentage of whole blood occupied by erythrocytes (normal: 40%–50% in males, 36%–44% in females).
• Mean Corpuscular Volume (MCV): Measures average erythrocyte volume in femtoliters (fL).
  - Microcytic (MCV < 80 fL): Strongly suggests iron deficiency anemia, thalassemia, or lead toxicity.
  - Normocytic (MCV 80–100 fL): Seen in acute blood loss, hemolysis, or renal disease.
  - Macrocytic (MCV > 100 fL): Characteristic of Vitamin B12 or folate deficiency.
• Mean Corpuscular Hemoglobin (MCH) & MCHC: Measure average hemoglobin weight and concentration per red cell. Low values (hypochromia) appear pale under microscopic examination.
• Red Cell Distribution Width (RDW): Measures variation in red blood cell volume (anisocytosis). High RDW (>15%) is a hallmark of iron deficiency.
• White Blood Cells (WBC) & Platelets (PLT): Monitor immune status and clotting capability.

HemoCare AI utilizes machine-learning classifiers trained on these exact 14 parameters to offer an immediate, automated risk assessment.`
  },
  {
    id: 3,
    title: 'Dietary Strategies to Boost Iron & Prevent Nutritional Anemia',
    cat: 'Diet & Nutrition',
    author: 'Clinical Nutrition Dept.',
    readTime: '3 min read',
    date: '2026-08-15',
    status: 'Published',
    summary: 'Mastering iron absorption: Discover how pairing heme and non-heme iron sources with Vitamin C boosts bio-availability and reverses nutritional deficits.',
    content: `Dietary modification is an essential pillar in managing and preventing nutritional iron deficiency. However, not all dietary iron is absorbed equally by the intestinal brush border.

The Two Types of Dietary Iron:
1. Heme Iron (High Bio-availability: 15%–35% absorbed):
   • Found in animal sources: lean poultry, fish, eggs, and red meats.
   • Readily absorbed without requiring dietary co-factors.
2. Non-Heme Iron (Variable Bio-availability: 2%–20% absorbed):
   • Found in plant sources: spinach, moringa leaves, fenugreek, lentils (dal), chickpeas, beans, pumpkin seeds, raisins, and fortified cereals.

Absorption Enhancers:
• Vitamin C (Ascorbic Acid): Consuming citrus fruits (oranges, lemons), amla (Indian gooseberry), guava, or bell peppers alongside plant-based iron meals increases iron absorption up to threefold.
• Fermented and sprouted foods: Sprouting beans and grains breaks down phytates, increasing mineral bioavailability.

Absorption Inhibitors (Avoid Near Meals):
• Tannins and Polyphenols: Found in tea and coffee. Avoid drinking chai or coffee 1 hour before and 1 hour after meals.
• Calcium: Competes for the same divalent metal transporters. Do not take calcium supplements or large glasses of milk directly with iron-rich main meals.`
  },
  {
    id: 4,
    title: 'When to See a Doctor: Red Flag Symptoms in Anemia',
    cat: 'Clinical Care',
    author: 'Dr. Rohan Das (Internal Medicine)',
    readTime: '4 min read',
    date: '2026-09-01',
    status: 'Draft',
    summary: 'Recognize emergent warning signs such as chest angina, orthostatic syncope, tachypnea, and melena that demand immediate clinical intervention.',
    content: `While mild anemia might initially present as manageable lethargy, severe or acute anemia can compromise vital organ perfusion and oxygenation.

Urgent Clinical Red Flags:
• Resting chest pain, angina, or irregular rapid heart rates (tachycardia)
• Severe shortness of breath (dyspnea) even while sitting or lying flat
• Syncope (fainting), loss of consciousness, or severe dizzy spells upon standing
• Melena (dark, tarry stools) or hematemesis (vomiting blood), signaling acute gastrointestinal hemorrhage
• Unexplained extensive petechiae, spontaneous mucosal bleeding, or severe purpura
• Progressive jaundice (yellowing of sclera or skin) accompanying profound weakness

Clinical Takeaway:
AI screening tools provide rapid triaging, but patients exhibiting acute or worsening red-flag symptoms must be evaluated immediately in an emergency department or specialized hematology clinic.`
  },
  {
    id: 5,
    title: 'Iron Supplementation: Formulations, Dosages & Safe Practices',
    cat: 'Clinical Care',
    author: 'Dr. Aarav Mehta (Hematology Lead)',
    readTime: '6 min read',
    date: '2026-10-02',
    status: 'Published',
    summary: 'An evidence-based guide to oral iron formulations, managing gastrointestinal side effects, and monitoring serum ferritin restoration.',
    content: `Oral iron replacement therapy remains the frontline therapy for documented iron deficiency anemia under professional clinical guidance.

Common Oral Formulations:
• Ferrous Sulfate: Contains ~65 mg elemental iron per 325 mg tablet. Highly accessible and effective.
• Ferrous Fumarate: Contains ~106 mg elemental iron per 325 mg tablet.
• Ferrous Ascorbate / Bisglycinate Chelate: Formulated with organic chelates to minimize gastric irritation and nausea.

Best Administration Practices:
• Optimal Timing: Take on an empty stomach with a glass of water or fresh citrus juice 1 hour before meals.
• Managing Side Effects: Nausea, constipation, dark stools, and epigastric discomfort are well-documented. If intolerance occurs, taking supplements with a light meal or switching to alternate-day dosing improves compliance without compromising overall absorption.
• Course Duration: Hemoglobin levels typically normalize within 6–8 weeks, but supplementation must often continue for 3–6 months to fully replenish bone marrow ferritin reserves.

Always consult your attending physician before initiating iron therapy, as unmonitored supplementation can mask occult pathology or risk iron overload in non-deficiency states.`
  }
]

const STORAGE_KEY = 'hemocare_education_articles'

export function AEdu(){
  const {notify}=useApp()
  const [rows, setRows] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed) && parsed.length > 0) return parsed
      }
    } catch (e) {
      console.error(e)
    }
    return DEFAULT_ARTICLES
  })

  const [query, setQuery] = useState('')
  const [filterCat, setFilterCat] = useState('All')
  const [filterStatus, setFilterStatus] = useState('All')
  const [viewArticle, setViewArticle] = useState(null)
  const [editArticle, setEditArticle] = useState(null)
  const [form, setForm] = useState({
    id: null,
    title: '',
    cat: 'Basics',
    author: 'Medical Team',
    readTime: '4 min read',
    date: new Date().toISOString().split('T')[0],
    status: 'Published',
    summary: '',
    content: ''
  })

  const saveArticles = (newRows) => {
    setRows(newRows)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newRows))
    } catch (e) {
      console.error(e)
    }
  }

  const openCreateModal = () => {
    setForm({
      id: null,
      title: '',
      cat: 'Basics',
      author: 'Medical Team',
      readTime: '4 min read',
      date: new Date().toISOString().split('T')[0],
      status: 'Published',
      summary: '',
      content: ''
    })
    setEditArticle({ isNew: true })
  }

  const openEditModal = (article) => {
    setForm({
      id: article.id,
      title: article.title,
      cat: article.cat || 'Basics',
      author: article.author || 'Medical Team',
      readTime: article.readTime || '4 min read',
      date: article.date || new Date().toISOString().split('T')[0],
      status: article.status || 'Published',
      summary: article.summary || '',
      content: article.content || ''
    })
    setEditArticle(article)
  }

  const handleSave = (e) => {
    e.preventDefault()
    if (!form.title.trim()) {
      notify('Title cannot be empty')
      return
    }
    if (!form.content.trim()) {
      notify('Article content cannot be empty')
      return
    }

    if (editArticle.isNew) {
      const newArticle = {
        ...form,
        id: Date.now()
      }
      const updated = [newArticle, ...rows]
      saveArticles(updated)
      notify('New article published successfully!')
    } else {
      const updated = rows.map(r => r.id === form.id ? { ...form } : r)
      saveArticles(updated)
      notify('Article updated successfully!')
      if (viewArticle && viewArticle.id === form.id) {
        setViewArticle({ ...form })
      }
    }
    setEditArticle(null)
  }

  const handleDelete = (id) => {
    const updated = rows.filter(r => r.id !== id)
    saveArticles(updated)
    if (viewArticle && viewArticle.id === id) {
      setViewArticle(null)
    }
    notify('Article removed')
  }

  // Categories list
  const categories = ['All', ...new Set(rows.map(r => r.cat))]

  // Filtered rows
  const filtered = rows.filter(r => {
    const matchesCat = filterCat === 'All' || r.cat === filterCat
    const matchesStatus = filterStatus === 'All' || r.status === filterStatus
    const q = query.toLowerCase().trim()
    const matchesQuery = !q ||
      r.title.toLowerCase().includes(q) ||
      (r.summary && r.summary.toLowerCase().includes(q)) ||
      (r.content && r.content.toLowerCase().includes(q)) ||
      (r.cat && r.cat.toLowerCase().includes(q))
    return matchesCat && matchesStatus && matchesQuery
  })

  return <>
    <Head title="Health Education" sub="Manage and publish patient health articles">
      <button className="btn" onClick={openCreateModal}><Plus size={16}/> Add Article</button>
    </Head>

    <div className="grid">
      <Stat icon={BookOpen} label="Total Articles" value={rows.length}/>
      <Stat icon={CheckCircle2} label="Published Articles" value={rows.filter(r=>r.status==='Published').length}/>
      <Stat icon={FileText} label="Draft Articles" value={rows.filter(r=>r.status==='Draft').length}/>
      <Stat icon={ShieldCheck} label="Categories" value={new Set(rows.map(r=>r.cat)).size}/>
    </div>

    <div className="card" style={{marginTop: 18, marginBottom: 18, padding: '16px 20px'}}>
      <div style={{display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between'}}>
        <div style={{position: 'relative', flex: '1 1 280px', maxWidth: 420}}>
          <Search size={16} style={{position: 'absolute', left: 12, top: 12, color: 'var(--m)'}}/>
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search articles by title, topic, or keyword..."
            style={{paddingLeft: 38}}
          />
        </div>
        <div style={{display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 6}}>
            <span style={{fontSize: 13, color: 'var(--m)', fontWeight: 600}}>Category:</span>
            <select value={filterCat} onChange={e => setFilterCat(e.target.value)} style={{width: 'auto', minWidth: 140}}>
              {categories.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div style={{display: 'flex', alignItems: 'center', gap: 6}}>
            <span style={{fontSize: 13, color: 'var(--m)', fontWeight: 600}}>Status:</span>
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} style={{width: 'auto', minWidth: 120}}>
              <option value="All">All Status</option>
              <option value="Published">Published</option>
              <option value="Draft">Draft</option>
            </select>
          </div>
        </div>
      </div>
    </div>

    <Table cols={[
      {
        h:'Article Details',
        r: x => <div style={{maxWidth: 380}}>
          <b style={{color: 'var(--n)', fontSize: 14.5, display: 'block', cursor: 'pointer', marginBottom: 3}} onClick={() => setViewArticle(x)}>
            {x.title}
          </b>
          <span style={{display: 'inline-block', background: 'var(--l)', color: 'var(--b)', fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 99, marginRight: 8}}>
            {x.cat}
          </span>
          <span style={{color: 'var(--m)', fontSize: 12.5, display: 'inline-block', verticalAlign: 'middle', whiteSpace: 'normal'}}>
            {x.summary ? (x.summary.length > 70 ? x.summary.slice(0, 70) + '…' : x.summary) : 'No summary'}
          </span>
        </div>
      },
      {
        h:'Author & Read Time',
        r: x => <div>
          <div style={{fontSize: 13, fontWeight: 500, color: 'var(--tx)'}}>{x.author || 'Medical Staff'}</div>
          <span style={{fontSize: 12, color: 'var(--m)'}}>{x.readTime || '4 min read'}</span>
        </div>
      },
      {h:'Date', k:'date'},
      {h:'Status', r: x => <Badge s={x.status}/>},
      {
        h:'Actions',
        r: x => <div style={{display: 'flex', gap: 6}}>
          <button className="btn s o" title="View Full Article" onClick={() => setViewArticle(x)}>
            <Eye size={13}/> View
          </button>
          <button className="btn s" title="Edit Article" onClick={() => openEditModal(x)}>
            <Edit2 size={13}/> Edit
          </button>
          <button className="btn s r" title="Delete Article" onClick={() => handleDelete(x.id)}>
            <Trash2 size={13}/>
          </button>
        </div>
      }
    ]} rows={filtered}/>

    {/* VIEW ARTICLE MODAL */}
    {viewArticle && (
      <div className="modal-bg" onClick={() => setViewArticle(null)}>
        <div className="modal" style={{maxWidth: 720, maxHeight: '90vh', overflowY: 'auto', textAlign: 'left'}} onClick={e => e.stopPropagation()}>
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--line)', paddingBottom: 14, marginBottom: 16}}>
            <div style={{display: 'flex', gap: 8, alignItems: 'center'}}>
              <span style={{background: 'var(--l)', color: 'var(--b)', padding: '4px 10px', borderRadius: 99, fontSize: 12, fontWeight: 600}}>{viewArticle.cat}</span>
              <Badge s={viewArticle.status}/>
            </div>
            <button className="iconbtn" onClick={() => setViewArticle(null)} style={{width: 32, height: 32}}>
              <X size={16}/>
            </button>
          </div>

          <h2 style={{fontSize: 22, color: 'var(--n)', marginBottom: 8, lineHeight: 1.3}}>{viewArticle.title}</h2>

          <div style={{display: 'flex', gap: 16, color: 'var(--m)', fontSize: 13, marginBottom: 18, flexWrap: 'wrap', alignItems: 'center'}}>
            <span style={{display: 'flex', gap: 6, alignItems: 'center'}}><User size={14}/> {viewArticle.author || 'Medical Staff'}</span>
            <span style={{display: 'flex', gap: 6, alignItems: 'center'}}><Clock size={14}/> {viewArticle.readTime || '4 min read'}</span>
            <span>• Published: {viewArticle.date}</span>
          </div>

          {viewArticle.summary && (
            <div style={{background: 'var(--l)', borderLeft: '4px solid var(--t)', padding: '12px 16px', borderRadius: '0 8px 8px 0', marginBottom: 20}}>
              <b style={{color: 'var(--n)', display: 'block', fontSize: 13, marginBottom: 4}}>Key Takeaway / Summary</b>
              <p style={{margin: 0, color: 'var(--tx)', fontSize: 13.5, lineHeight: 1.55}}>{viewArticle.summary}</p>
            </div>
          )}

          <div style={{color: 'var(--tx)', fontSize: 14.5, lineHeight: 1.7, whiteSpace: 'pre-line', marginBottom: 24}}>
            {viewArticle.content}
          </div>

          <div className="foot" style={{borderTop: '1px solid var(--line)', paddingTop: 16, marginTop: 12}}>
            <button className="btn o" onClick={() => setViewArticle(null)}>Close</button>
            <button className="btn" onClick={() => { const toEdit = viewArticle; setViewArticle(null); openEditModal(toEdit); }}>
              <Edit2 size={15}/> Edit Article
            </button>
          </div>
        </div>
      </div>
    )}

    {/* EDIT / CREATE ARTICLE MODAL */}
    {editArticle && (
      <div className="modal-bg" onClick={() => setEditArticle(null)}>
        <div className="modal" style={{maxWidth: 680, maxHeight: '90vh', overflowY: 'auto', textAlign: 'left'}} onClick={e => e.stopPropagation()}>
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--line)', paddingBottom: 12, marginBottom: 18}}>
            <h3 style={{margin: 0, fontSize: 18, color: 'var(--n)'}}>
              {editArticle.isNew ? 'Create New Health Education Article' : 'Edit Health Education Article'}
            </h3>
            <button className="iconbtn" onClick={() => setEditArticle(null)} style={{width: 32, height: 32}}>
              <X size={16}/>
            </button>
          </div>

          <form onSubmit={handleSave} style={{display: 'flex', flexDirection: 'column', gap: 14}}>
            <div>
              <label>Article Title *</label>
              <input
                required
                value={form.title}
                onChange={e => setForm({...form, title: e.target.value})}
                placeholder="e.g. Recognizing Symptoms of Iron Deficiency Anemia"
              />
            </div>

            <div className="f">
              <div>
                <label>Category</label>
                <select value={form.cat} onChange={e => setForm({...form, cat: e.target.value})}>
                  <option value="Basics">Basics</option>
                  <option value="Diagnostic Tests">Diagnostic Tests</option>
                  <option value="Diet & Nutrition">Diet & Nutrition</option>
                  <option value="Clinical Care">Clinical Care</option>
                  <option value="Awareness">Awareness</option>
                  <option value="General">General</option>
                </select>
              </div>
              <div>
                <label>Publication Status</label>
                <select value={form.status} onChange={e => setForm({...form, status: e.target.value})}>
                  <option value="Published">Published</option>
                  <option value="Draft">Draft</option>
                </select>
              </div>
            </div>

            <div className="f">
              <div>
                <label>Author / Reviewer</label>
                <input
                  value={form.author}
                  onChange={e => setForm({...form, author: e.target.value})}
                  placeholder="e.g. Dr. Aarav Mehta (Hematology Lead)"
                />
              </div>
              <div>
                <label>Read Time</label>
                <input
                  value={form.readTime}
                  onChange={e => setForm({...form, readTime: e.target.value})}
                  placeholder="e.g. 4 min read"
                />
              </div>
            </div>

            <div>
              <label>Summary / Abstract</label>
              <textarea
                rows={2}
                value={form.summary}
                onChange={e => setForm({...form, summary: e.target.value})}
                placeholder="A concise 1-2 sentence overview of the article..."
              />
            </div>

            <div>
              <label>Full Article Content *</label>
              <textarea
                required
                rows={9}
                value={form.content}
                onChange={e => setForm({...form, content: e.target.value})}
                placeholder="Write full article content here. Use paragraphs, bullet points (•), or numbered lists..."
                style={{fontFamily: 'inherit', resize: 'vertical'}}
              />
            </div>

            <div className="foot" style={{borderTop: '1px solid var(--line)', paddingTop: 16, marginTop: 8}}>
              <button type="button" className="btn o" onClick={() => setEditArticle(null)}>Cancel</button>
              <button type="submit" className="btn">
                <CheckCircle2 size={16}/> {editArticle.isNew ? 'Publish Article' : 'Save Changes'}
              </button>
            </div>
          </form>
        </div>
      </div>
    )}
  </>
}

export function ASettings(){
  const {notify} = useApp()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [form, setForm] = useState({
    platform_name: 'HemoCare AI',
    support_email: 'support@hemocare.ai',
    contact_phone: '+91 98765 43210',
    institution: 'AI Health Research Center',
    allow_patient_registration: true,
    enable_email_alerts: true,
    updated_at: ''
  })

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/settings/')
      .then(res => res.json())
      .then(data => {
        if (data && data.platform_name) {
          setForm({
            platform_name: data.platform_name,
            support_email: data.support_email,
            contact_phone: data.contact_phone || '',
            institution: data.institution || '',
            allow_patient_registration: data.allow_patient_registration ?? true,
            enable_email_alerts: data.enable_email_alerts ?? true,
            updated_at: data.updated_at || ''
          })
        }
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load settings', err)
        setLoading(false)
      })
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      const res = await fetch('http://127.0.0.1:8000/api/settings/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      })
      const result = await res.json()
      if (res.ok) {
        notify('Platform settings saved successfully to MySQL database!')
        if (result.data) {
          setForm(prev => ({ ...prev, ...result.data }))
        }
      } else {
        notify('Failed to save settings: ' + JSON.stringify(result))
      }
    } catch (err) {
      console.error(err)
      notify('Network error saving settings to database.')
    } finally {
      setSaving(false)
    }
  }

  return <>
    <Head title="Platform Settings" sub="Configure system identity, support contacts, and clinical operational parameters stored in MySQL."/>
    
    <div className="card" style={{ maxWidth: 800 }}>
      {loading ? (
        <p style={{ color: 'var(--m)', padding: '24px 0', textAlign: 'center' }}>Loading system settings from database...</p>
      ) : (
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--line)', paddingBottom: 12 }}>
            <h3 style={{ margin: 0, fontSize: 16, color: 'var(--tx)', display: 'flex', alignItems: 'center', gap: 8 }}>
              General Platform Identity
            </h3>
            {form.updated_at && (
              <span style={{ fontSize: 11, color: 'var(--m)' }}>
                Last updated: {new Date(form.updated_at).toLocaleString()}
              </span>
            )}
          </div>

          <div className="g2">
            <div>
              <label>Platform Name *</label>
              <input
                type="text"
                required
                value={form.platform_name}
                onChange={e => setForm({ ...form, platform_name: e.target.value })}
                placeholder="e.g. HemoCare AI"
              />
            </div>
            <div>
              <label>Support / Admin Email *</label>
              <input
                type="email"
                required
                value={form.support_email}
                onChange={e => setForm({ ...form, support_email: e.target.value })}
                placeholder="e.g. support@hemocare.ai"
              />
            </div>
          </div>

          <div className="g2">
            <div>
              <label>Contact / Helpline Number</label>
              <input
                type="text"
                value={form.contact_phone}
                onChange={e => setForm({ ...form, contact_phone: e.target.value })}
                placeholder="e.g. +91 98765 43210"
              />
            </div>
            <div>
              <label>Affiliated Medical Institution / Department</label>
              <input
                type="text"
                value={form.institution}
                onChange={e => setForm({ ...form, institution: e.target.value })}
                placeholder="e.g. AI Health Research Center"
              />
            </div>
          </div>

          <div style={{ borderTop: '1px solid var(--line)', paddingTop: 16 }}>
            <h3 style={{ margin: '0 0 12px', fontSize: 15, color: 'var(--tx)' }}>Access & Clinical Alert Policies</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', fontSize: 14 }}>
                <input
                  type="checkbox"
                  checked={form.allow_patient_registration}
                  onChange={e => setForm({ ...form, allow_patient_registration: e.target.checked })}
                  style={{ width: 16, height: 16 }}
                />
                <span><b>Allow Self-Registration for Patients:</b> Enable public registration via the login/register portal.</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', fontSize: 14 }}>
                <input
                  type="checkbox"
                  checked={form.enable_email_alerts}
                  onChange={e => setForm({ ...form, enable_email_alerts: e.target.checked })}
                  style={{ width: 16, height: 16 }}
                />
                <span><b>Enable High-Risk Screening Reminders:</b> Dispatch automatic clinical follow-up notices to patients flagged with microcytic indices.</span>
              </label>
            </div>
          </div>

          <div style={{ borderTop: '1px solid var(--line)', paddingTop: 16, display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
            <button
              type="submit"
              className="btn"
              disabled={saving}
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <CheckCircle2 size={16}/> {saving ? 'Saving to Database...' : 'Save Settings to MySQL'}
            </button>
          </div>
        </form>
      )}
    </div>
  </>
}

