export const doctors=[
{id:1,name:'Dr. Aarav Mehta',spec:'Hematologist',exp:12,loc:'Pune',avail:'Today',rating:4.8,status:'Verified',patients:48},
{id:2,name:'Dr. Sana Iyer',spec:'General Physician',exp:8,loc:'Mumbai',avail:'Tomorrow',rating:4.6,status:'Verified',patients:35},
{id:3,name:'Dr. Rohan Das',spec:'Internal Medicine',exp:15,loc:'Delhi',avail:'Today',rating:4.9,status:'Verified',patients:62},
{id:4,name:'Dr. Meera Nair',spec:'Hematologist',exp:5,loc:'Kochi',avail:'Monday',rating:0,status:'Pending',patients:0},
{id:5,name:'Dr. Kabir Shah',spec:'General Physician',exp:3,loc:'Surat',avail:'Wednesday',rating:0,status:'Pending',patients:0}]
export const patients=[
{id:1,name:'Ananya Rao',email:'ananya@demo.com',age:24,joined:'2026-01-12',count:3,status:'Active',risk:'Low Risk',last:'2026-09-20'},
{id:2,name:'Vikram Joshi',email:'vikram@demo.com',age:41,joined:'2026-02-03',count:2,status:'Active',risk:'Needs Attention',last:'2026-09-11'},
{id:3,name:'Priya Nambiar',email:'priya@demo.com',age:33,joined:'2026-03-18',count:4,status:'Active',risk:'Medium Risk',last:'2026-09-02'},
{id:4,name:'Rahul Verma',email:'rahul@demo.com',age:57,joined:'2026-04-22',count:1,status:'Inactive',risk:'Low Risk',last:'2026-07-15'}]
export const appointments=[
{id:1,patient:'Ananya Rao',doctor:'Dr. Aarav Mehta',date:'2026-10-12',time:'10:30',status:'Upcoming'},
{id:2,patient:'Vikram Joshi',doctor:'Dr. Aarav Mehta',date:'2026-10-05',time:'11:00',status:'Upcoming'},
{id:3,patient:'Priya Nambiar',doctor:'Dr. Sana Iyer',date:'2026-09-25',time:'15:00',status:'Completed'},
{id:4,patient:'Ananya Rao',doctor:'Dr. Rohan Das',date:'2026-08-14',time:'09:30',status:'Cancelled'}]
export const assessments=[
{id:'AS-1003',date:'2026-09-20',status:'Low Risk',score:28},
{id:'AS-1002',date:'2026-08-02',status:'Medium Risk',score:46},
{id:'AS-1001',date:'2026-06-15',status:'Needs Attention',score:71}]
export const cbc=[['Hemoglobin','13.2 g/dL'],['RBC Count','4.6 M/uL'],['Hematocrit','40 %'],['MCV','88 fL'],['MCH','29 pg'],['MCHC','33 g/dL'],['RDW','13 %'],['WBC Count','6.8 K/uL'],['Platelets','250 K/uL']]
export const trend=[{m:'May',patients:20,assess:35,appts:12},{m:'Jun',patients:34,assess:52,appts:21},{m:'Jul',patients:48,assess:80,appts:30},{m:'Aug',patients:65,assess:104,appts:41},{m:'Sep',patients:82,assess:131,appts:55},{m:'Oct',patients:96,assess:150,appts:63}]
export const dist=[{name:'Low Risk',value:58},{name:'Medium Risk',value:27},{name:'Needs Attention',value:15}]
export const articles=[
{id:1,title:'What is Anemia?',cat:'Basics',date:'2026-05-01',status:'Published'},
{id:2,title:'Understanding Your CBC',cat:'Lab Tests',date:'2026-06-10',status:'Published'},
{id:3,title:'When to See a Doctor',cat:'Awareness',date:'2026-09-01',status:'Draft'}]
