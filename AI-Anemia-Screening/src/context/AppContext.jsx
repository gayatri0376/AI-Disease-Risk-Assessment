import {createContext,useContext,useState} from 'react'
import {doctors,appointments} from '../data/mock'
const C=createContext()
export const useApp=()=>useContext(C)
export function AppProvider({children}){
  const [role,setRole]=useState(localStorage.getItem('role'))
  const [currentUser, setCurrentUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('current_user') || 'null') }
    catch { return null }
  })
  const [toast,setToast]=useState('')
  const [appts,setAppts]=useState(appointments)
  const [docs,setDocs]=useState(doctors)

  const notify=m=>{setToast(m);setTimeout(()=>setToast(''),3000)}

  const login=(r, u)=>{
    localStorage.setItem('role', r)
    setRole(r)
    if(u){
      localStorage.setItem('current_user', JSON.stringify(u))
      setCurrentUser(u)
    }
  }

  const logout=()=>{
    localStorage.removeItem('role')
    localStorage.removeItem('current_user')
    localStorage.removeItem('latest_assessment')
    setRole(null)
    setCurrentUser(null)
    try {
      fetch('http://127.0.0.1:8000/api/auth/logout/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
      }).catch(()=>{})
    } catch(e){}
    window.location.href = '/login'
  }

  return (
    <C.Provider value={{role,currentUser,login,logout,notify,appts,setAppts,docs,setDocs}}>
      {children}
      {toast&&<div className="toast">{toast}</div>}
    </C.Provider>
  )
}
