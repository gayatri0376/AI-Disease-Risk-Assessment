import {Routes,Route,Navigate} from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import {Landing,Info,Auth} from './pages/public/Public'
import {PDash,Assess,Result,Reports,Doctors,DoctorProfile,Appts,PatientProfile} from './pages/patient/Patient'
import {DDash,DPatients,DPatient,Notes,DReports,Form,DoctorProfilePage,DoctorAvailabilityPage} from './pages/doctor/Doctor'
import {ADash,APatients,ADoctors,Verify,AAppts,AAnalytics,AAssess,AEdu,ASettings} from './pages/admin/Admin'
const R=(p,e)=><Route path={p} element={e}/>
const prof=<Form title="Profile" sub="Manage your details" cta="Save changes" msg="Profile saved" fields={[['Full Name','text','Demo User'],['Email','email','demo@hemocare.ai'],['Phone','text','9876543210']]}/>
export default function App(){return <Routes>
{R('/',<Landing/>)}{R('/about',<Info kind="about"/>)}{R('/how-it-works',<Info kind="how"/>)}{R('/education',<Info kind="edu"/>)}{R('/login',<Auth/>)}{R('/register',<Auth reg/>)}
<Route path="/patient" element={<AppLayout role="patient"/>}><Route index element={<Navigate to="dashboard"/>}/>
{R('dashboard',<PDash/>)}{R('assessment',<Assess/>)}{R('result',<Result/>)}{R('reports',<Reports/>)}{R('doctors',<Doctors/>)}{R('doctors/:id',<DoctorProfile/>)}{R('appointments',<Appts/>)}{R('profile',<PatientProfile/>)}</Route>
<Route path="/doctor" element={<AppLayout role="doctor"/>}><Route index element={<Navigate to="dashboard"/>}/>
{R('dashboard',<DDash/>)}{R('patients',<DPatients/>)}{R('appointments',<AAppts/>)}{R('patient/:id',<DPatient/>)}{R('reports',<DReports/>)}{R('consultation',<Notes/>)}
{R('availability',<DoctorAvailabilityPage/>)}{R('profile',<DoctorProfilePage/>)}</Route>
<Route path="/admin" element={<AppLayout role="admin"/>}><Route index element={<Navigate to="dashboard"/>}/>
{R('dashboard',<ADash/>)}{R('patients',<APatients/>)}{R('doctors',<ADoctors/>)}{R('verification',<Verify/>)}{R('appointments',<AAppts/>)}{R('assessments',<AAssess/>)}{R('analytics',<AAnalytics/>)}{R('education',<AEdu/>)}
{R('settings',<ASettings/>)}</Route>
<Route path="*" element={<Navigate to="/"/>}/></Routes>}
