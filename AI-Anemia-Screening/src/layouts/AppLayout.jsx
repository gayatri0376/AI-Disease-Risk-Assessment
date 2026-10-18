import {useState, useEffect} from 'react'
import {NavLink,Outlet,Navigate,Link,useNavigate} from 'react-router-dom'
import {LayoutDashboard,ClipboardList,Brain,FileText,Stethoscope,CalendarDays,User,Users,Clock,NotebookPen,ShieldCheck,BarChart3,BookOpen,Settings,LogOut,Menu,HeartPulse,Bell,ChevronDown,CheckCheck,X,ExternalLink} from 'lucide-react'
import {useApp} from '../context/AppContext'
const L=(p,l,I)=>[p,l,I]
const GROUPS={
patient:[['Overview',[L('dashboard','Dashboard',LayoutDashboard)]],['Screening',[L('assessment','Health Assessment',ClipboardList),L('result','AI Screening',Brain),L('reports','My Reports',FileText)]],['Care',[L('doctors','Find Doctor',Stethoscope),L('appointments','Appointments',CalendarDays)]],['Account',[L('profile','Profile',User)]]],
doctor:[['Overview',[L('dashboard','Dashboard',LayoutDashboard)]],['Clinic',[L('patients','My Patients',Users),L('appointments','Appointments',CalendarDays),L('patient/1','Patient Assessments',ClipboardList),L('reports','Reports',FileText),L('consultation','Consultation Notes',NotebookPen)]],['Account',[L('availability','Availability',Clock),L('profile','Profile',User)]]],
admin:[['Overview',[L('dashboard','Dashboard',LayoutDashboard),L('analytics','Analytics',BarChart3)]],['People',[L('patients','Patients',Users),L('doctors','Doctors',Stethoscope),L('verification','Doctor Verification',ShieldCheck)]],['Operations',[L('appointments','Appointments',CalendarDays),L('assessments','Assessments',ClipboardList),L('education','Health Education',BookOpen)]],['System',[L('settings','Settings',Settings)]]]}
const WHO={patient:['Ananya Rao','AR'],doctor:['Dr. Aarav Mehta','AM'],admin:['Admin User','AU']}
export default function AppLayout({role}){
  const {role:cur,currentUser,logout,notify}=useApp()
  const nav=useNavigate()
  const [open,setOpen]=useState(false)
  const [pop,setPop]=useState('')
  const [notifications, setNotifications] = useState([])

  if(cur!==role) return <Navigate to="/login"/>

  const user = currentUser || (typeof window !== 'undefined' ? JSON.parse(localStorage.getItem('current_user') || 'null') : null)
  const defaultWho = WHO[role] || ['User','U']
  const name = user?.name || defaultWho[0]
  
  // Smart initials calculation: strips Dr. prefix to get actual doctor initials (e.g. Dr. Mohini -> MJ or MO)
  const cleanName = (name || '').replace(/^Dr\.?\s+/i, '').trim()
  const parts = cleanName.split(' ').filter(Boolean)
  const ini = parts.length >= 2
    ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
    : (cleanName ? cleanName.slice(0, 2).toUpperCase() : (defaultWho[1] || 'DR'))
  const prof = role==='admin' ? 'settings' : 'profile'

  const fetchNotifications = async () => {
    try {
      const emailParam = user?.email ? `&email=${encodeURIComponent(user.email)}` : ''
      const res = await fetch(`http://127.0.0.1:8000/api/notifications/?role=${role}${emailParam}`)
      if (res.ok) {
        const data = await res.json()
        setNotifications(Array.isArray(data) ? data : [])
      }
    } catch (err) {
      console.error('Failed to load notifications:', err)
    }
  }

  useEffect(() => {
    fetchNotifications()
  }, [role, user?.email])

  const unreadCount = notifications.filter(n => !n.is_read).length

  const handleMarkAsRead = async (id, e) => {
    e?.stopPropagation?.()
    try {
      await fetch(`http://127.0.0.1:8000/api/notifications/${id}/`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_read: true }),
      })
      setNotifications(prev => prev.map(n => n.id === id ? { ...n, is_read: true } : n))
    } catch (err) {
      console.error('Failed to mark read', err)
    }
  }

  const handleMarkAllRead = async (e) => {
    e?.stopPropagation?.()
    try {
      await fetch('http://127.0.0.1:8000/api/notifications/mark-all-read/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, email: user?.email }),
      })
      setNotifications(prev => prev.map(n => ({ ...n, is_read: true })))
    } catch (err) {
      console.error('Failed to mark all read', err)
    }
  }

  const handleDeleteNotification = async (id, e) => {
    e?.stopPropagation?.()
    try {
      await fetch(`http://127.0.0.1:8000/api/notifications/${id}/`, { method: 'DELETE' })
      setNotifications(prev => prev.filter(n => n.id !== id))
    } catch (err) {
      console.error('Failed to delete notification', err)
    }
  }

  const handleNotificationClick = (item) => {
    if (!item.is_read) {
      handleMarkAsRead(item.id)
    }
    setPop('')
    if (item.link) {
      nav(item.link)
    }
  }

  const getNoteIcon = (type) => {
    switch (type) {
      case 'appointment':
        return <CalendarDays size={15} style={{ color: '#0284c7' }} />
      case 'screening':
        return <Brain size={15} style={{ color: '#7c3aed' }} />
      case 'reminder':
        return <Clock size={15} style={{ color: '#d97706' }} />
      default:
        return <HeartPulse size={15} style={{ color: '#e11d48' }} />
    }
  }

  const handleProfileClick = (e) => {
    e?.stopPropagation?.()
    setPop('')
    nav(`/${role}/${prof}`)
  }

  const handleLogoutClick = (e) => {
    e?.stopPropagation?.()
    setPop('')
    logout()
  }

  return <div className="app">
    {open&&<div className="scrim" onClick={()=>setOpen(false)}/>}
    <aside className={'side'+(open?' open':'')} onClick={()=>setOpen(false)}>
      <Link to="/" className="logo"><i><HeartPulse size={20}/></i>HemoCare AI</Link>
      {GROUPS[role].map(([t,items])=><div key={t}><div className="nav-sec">{t}</div>{items.map(([p,l,I])=><NavLink key={p} to={`/${role}/${p}`}><I size={18}/>{l}</NavLink>)}</div>)}
      <div className="grow"/>
      <button type="button" onClick={handleLogoutClick}><LogOut size={18}/>Logout</button>
    </aside>
    <div className="main">
      {pop&&<div className="scrim" style={{background:'none',zIndex:45}} onClick={()=>setPop('')}/>}
      <header className="top">
        <button className="iconbtn menu" aria-label="Menu" onClick={()=>setOpen(true)}><Menu size={18}/></button>
        <div className="sp"/>
        
        {/* Dynamic Notification & Clinical Reminders Bell */}
        <div className="rel">
          <button
            className="iconbtn"
            aria-label="Notifications"
            onClick={() => setPop(pop === 'n' ? '' : 'n')}
            title={unreadCount > 0 ? `${unreadCount} unread reminders` : 'Notifications'}
          >
            <Bell size={18} />
            {unreadCount > 0 && (
              <span
                style={{
                  position: 'absolute',
                  top: 3,
                  right: 3,
                  minWidth: 16,
                  height: 16,
                  borderRadius: 10,
                  background: 'var(--d)',
                  color: '#fff',
                  fontSize: 10,
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0 3px',
                  lineHeight: 1,
                  boxShadow: '0 0 0 2px #fff',
                }}
              >
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>

          {pop === 'n' && (
            <div
              className="pop"
              style={{
                zIndex: 60,
                width: 330,
                maxHeight: 420,
                display: 'flex',
                flexDirection: 'column',
                padding: 0,
                overflow: 'hidden',
                boxShadow: '0 16px 40px rgba(18,61,102,0.18)',
                borderRadius: 12,
              }}
            >
              <div
                style={{
                  padding: '12px 14px',
                  borderBottom: '1px solid var(--line)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  background: '#f8fafc',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <b style={{ fontSize: 13, color: 'var(--tx)' }}>Reminders & Alerts</b>
                  {unreadCount > 0 && (
                    <span
                      style={{
                        background: '#fee2e2',
                        color: '#b91c1c',
                        padding: '1px 6px',
                        borderRadius: 10,
                        fontSize: 11,
                        fontWeight: 600,
                      }}
                    >
                      {unreadCount} new
                    </span>
                  )}
                </div>
                {unreadCount > 0 && (
                  <button
                    type="button"
                    onClick={handleMarkAllRead}
                    style={{
                      width: 'auto',
                      padding: '3px 8px',
                      fontSize: 11,
                      color: 'var(--p)',
                      background: 'none',
                      border: 0,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      fontWeight: 600,
                    }}
                  >
                    <CheckCheck size={13} /> Mark all read
                  </button>
                )}
              </div>

              <div style={{ overflowY: 'auto', flex: 1, maxHeight: 330 }}>
                {notifications.length === 0 ? (
                  <div style={{ padding: '32px 16px', textAlign: 'center', color: 'var(--m)' }}>
                    <Bell size={26} style={{ opacity: 0.3, marginBottom: 8, margin: '0 auto' }} />
                    <p style={{ margin: 0, fontSize: 13 }}>No notifications right now.</p>
                    <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.7 }}>You are completely caught up!</p>
                  </div>
                ) : (
                  notifications.map((item) => (
                    <div
                      key={item.id}
                      onClick={() => handleNotificationClick(item)}
                      style={{
                        padding: '11px 13px',
                        borderBottom: '1px solid var(--line)',
                        cursor: 'pointer',
                        background: item.is_read ? '#fff' : '#f0fdf4',
                        transition: 'background 0.15s ease',
                        position: 'relative',
                        display: 'flex',
                        gap: 10,
                        alignItems: 'flex-start',
                      }}
                    >
                      <div
                        style={{
                          marginTop: 2,
                          width: 28,
                          height: 28,
                          borderRadius: 8,
                          background: item.is_read ? '#f1f5f9' : '#fff',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
                        }}
                      >
                        {getNoteIcon(item.type)}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, marginBottom: 2 }}>
                          <span
                            style={{
                              fontSize: 12,
                              fontWeight: item.is_read ? 600 : 700,
                              color: item.is_read ? 'var(--tx)' : '#0f172a',
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            }}
                          >
                            {item.title}
                          </span>
                          {!item.is_read && (
                            <span
                              style={{
                                width: 7,
                                height: 7,
                                borderRadius: '50%',
                                background: '#22c55e',
                                flexShrink: 0,
                              }}
                            />
                          )}
                        </div>
                        <p
                          style={{
                            margin: 0,
                            fontSize: 12,
                            color: item.is_read ? '#64748b' : '#334155',
                            lineHeight: 1.4,
                            wordBreak: 'break-word',
                          }}
                        >
                          {item.message}
                        </p>
                        <div
                          style={{
                            marginTop: 4,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            fontSize: 10,
                            color: '#94a3b8',
                          }}
                        >
                          <span>{item.created_at ? new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Today'}</span>
                          <div style={{ display: 'flex', gap: 8 }}>
                            {item.link && (
                              <span style={{ color: 'var(--p)', display: 'flex', alignItems: 'center', gap: 2, fontWeight: 500 }}>
                                View <ExternalLink size={10} />
                              </span>
                            )}
                            <button
                              type="button"
                              onClick={(e) => handleDeleteNotification(item.id, e)}
                              title="Dismiss"
                              style={{
                                border: 0,
                                background: 'none',
                                padding: 0,
                                color: '#94a3b8',
                                cursor: 'pointer',
                              }}
                            >
                              <X size={12} />
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        <div className="rel">
          <button className="me" onClick={()=>setPop(pop==='u'?'':'u')}>
            <span className="av sm">{ini}</span>
            <span className="hide"><b>{name}</b><span className="bd in" style={{padding:'1px 8px',fontSize:11,textTransform:'capitalize'}}>{role}</span></span>
            <ChevronDown size={16} className="hide"/>
          </button>
          {pop==='u'&&(
            <div className="pop" style={{zIndex:60}}>
              <button type="button" onClick={handleProfileClick}><User size={16}/>{role==='admin'?'Settings':'My profile'}</button>
              <button type="button" onClick={handleLogoutClick}><LogOut size={16}/>Logout</button>
            </div>
          )}
        </div>
      </header>
      <div className="page"><Outlet/></div>
    </div></div>
}
