import { useEffect, useState } from 'react';
import { Link, Route, Routes, useNavigate } from 'react-router-dom';
import { Activity, CalendarDays, HeartPulse, ShieldCheck, Stethoscope } from 'lucide-react';
import { createAppointment, fetchDashboard, fetchDoctors, fetchNotifications, loginUser, registerUser, setAuthToken } from './api';

const statCards = [
  { label: 'Patients Served', value: '2.4k+' },
  { label: 'Reports Summarized', value: '98%' },
  { label: 'Appointments Managed', value: '450+' },
  { label: 'Clinical Reviews', value: '24/7' },
];

const features = [
  'FHIR-style structured summaries',
  'Appointment management with doctor availability',
  'Role-aware patient, doctor, and admin dashboards',
  'Secure uploads and audit-friendly workflows',
];

function formatDisplayDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function getDisplayName(user) {
  if (!user) return 'Guest';
  return [user.first_name, user.last_name].filter(Boolean).join(' ') || user.username || 'User';
}

function getDashboardRoute(user) {
  if (user?.role === 'doctor') return '/doctor';
  if (user?.role === 'admin') return '/owner';
  if (user?.role === 'patient') return '/patient';
  return '/auth';
}

function App() {
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      const storedUser = localStorage.getItem('carepulse_user');
      return storedUser ? JSON.parse(storedUser) : null;
    } catch {
      return null;
    }
  });
  const [dashboardData, setDashboardData] = useState(null);
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    const savedToken = localStorage.getItem('carepulse_token');
    if (savedToken) {
      setAuthToken(savedToken);
    }
  }, []);

  useEffect(() => {
    const loadDashboard = async () => {
      const token = localStorage.getItem('carepulse_token');
      if (!currentUser || !token) {
        setDashboardData(null);
        setNotifications([]);
        return;
      }

      try {
        setAuthToken(token);
        const [dashboardResponse, notificationsResponse] = await Promise.all([
          fetchDashboard(),
          fetchNotifications(),
        ]);
        setDashboardData(dashboardResponse.data);
        setNotifications(notificationsResponse.data.results || notificationsResponse.data || []);
      } catch {
        setDashboardData(null);
        setNotifications([]);
      }
    };

    loadDashboard();
  }, [currentUser]);

  const dashboardPath = getDashboardRoute(currentUser);

  const handleLogout = () => {
    localStorage.removeItem('carepulse_token');
    localStorage.removeItem('carepulse_user');
    setAuthToken(null);
    setCurrentUser(null);
    setDashboardData(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-cyan-100 p-2 text-cyan-700"><HeartPulse size={20} /></div>
            <div>
              <div className="text-lg font-semibold">CarePulse</div>
              <div className="text-xs text-slate-500">Medical Summarization Platform</div>
            </div>
          </div>
          <nav className="hidden gap-6 md:flex">
            <Link to="/" className="text-sm font-medium text-slate-600">Home</Link>
            <Link to="/appointments" className="text-sm font-medium text-slate-600">Appointments</Link>
            <Link to={dashboardPath} className="text-sm font-medium text-slate-600">Dashboard</Link>
            <Link to="/help" className="text-sm font-medium text-slate-600">Help Center</Link>
            {currentUser ? (
              <button type="button" onClick={handleLogout} className="text-sm font-medium text-slate-600">Logout</button>
            ) : (
              <Link to="/auth" className="text-sm font-medium text-slate-600">Auth</Link>
            )}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        <Routes>
          <Route path="/" element={<LandingPage currentUser={currentUser} />} />
          <Route path="/doctor" element={<DoctorPortal currentUser={currentUser} dashboardData={dashboardData} notifications={notifications} />} />
          <Route path="/owner" element={<OwnerPortal currentUser={currentUser} dashboardData={dashboardData} />} />
          <Route path="/patient" element={<PatientPortal currentUser={currentUser} dashboardData={dashboardData} />} />
          <Route path="/appointments" element={<AppointmentPortal currentUser={currentUser} dashboardData={dashboardData} setDashboardData={setDashboardData} />} />
          <Route path="/auth" element={<AuthPortal onAuthChange={setCurrentUser} />} />
          <Route path="/help" element={<HelpCenterPage />} />
        </Routes>
      </main>
    </div>
  );
}

function LandingPage({ currentUser }) {
  const displayName = getDisplayName(currentUser);
  const roleTitle = currentUser?.role === 'doctor'
    ? 'Doctor Workspace'
    : currentUser?.role === 'admin'
      ? 'Owner Control Center'
      : 'Patient Portal';
  const roleSubtext = currentUser?.role === 'doctor'
    ? 'Review the clinical queue, patient records, and upcoming care actions in one place.'
    : currentUser?.role === 'admin'
      ? 'Manage the platform, approve clinic operations, update business settings, and control the full website workflow from one owner console.'
      : 'Manage your reports, appointments, and follow-ups in a secure patient workspace.';
  const roleQuickAction = currentUser?.role === 'doctor'
    ? 'Open doctor dashboard'
    : currentUser?.role === 'admin'
      ? 'Open owner dashboard'
      : 'Open patient dashboard';

  return (
    <div className="space-y-10">
      <section className="grid gap-8 rounded-3xl bg-gradient-to-br from-cyan-900 via-sky-800 to-slate-900 p-10 text-white shadow-xl md:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-5">
          <div className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-sm">{currentUser ? `Welcome back, ${displayName}` : 'Understand Your Medical Reports, Clearly.'}</div>
          <h1 className="text-4xl font-bold leading-tight md:text-6xl">{currentUser ? `${roleTitle} for ${displayName}` : 'Transform complex reports into clear, structured healthcare insights.'}</h1>
          <p className="max-w-2xl text-slate-200">{currentUser ? roleSubtext : 'CarePulse helps patients upload reports, receive informational summaries, and connect with specialists through a secure, modern medical workflow.'}</p>
          <div className="flex flex-wrap gap-3">
            <Link to={currentUser ? getDashboardRoute(currentUser) : '/auth'} className="rounded-full bg-cyan-400 px-5 py-3 font-semibold text-slate-900">{currentUser ? roleQuickAction : 'Get Started'}</Link>
            <Link to="/appointments" className="rounded-full border border-white/30 px-5 py-3 font-semibold">Book an Appointment</Link>
          </div>
        </div>
        <div className="grid gap-4">
          {statCards.map((card) => (
            <div key={card.label} className="rounded-2xl bg-white/10 p-4 backdrop-blur">
              <div className="text-3xl font-bold">{card.value}</div>
              <div className="text-sm text-slate-200">{card.label}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {features.map((feature) => (
          <div key={feature} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-3 inline-flex rounded-xl bg-cyan-50 p-2 text-cyan-700"><Stethoscope size={18} /></div>
            <p className="text-sm text-slate-700">{feature}</p>
          </div>
        ))}
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Card title="Medical Summarization" icon={<Activity size={18} />} text="Upload PDF, DOCX, or TXT reports and review structured, informational findings with clear disclaimers." />
        <Card title="Appointment Workflow" icon={<CalendarDays size={18} />} text="Search doctors, view availability, and book secure appointments with real-time status tracking." />
        <Card title="Protected Access" icon={<ShieldCheck size={18} />} text="Role-based protection ensures patients, doctors, and admins only see the data appropriate to them." />
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold">Help Center</h2>
            <p className="text-slate-500">Get quick answers about booking appointments, security, and common workflow steps.</p>
          </div>
          <Link to="/help" className="rounded-full bg-cyan-500 px-4 py-2 text-sm font-semibold text-white">Open Help Center</Link>
        </div>
      </section>
    </div>
  );
}

function HelpCenterPage() {
  const helpTopics = [
    { title: 'How do I book an appointment?', text: 'Go to the appointments area, choose a doctor, select a date and time, and submit the request. The assigned doctor will receive a new notification automatically.' },
    { title: 'How is access controlled?', text: 'Patients, doctors, and admins each see only the data relevant to their role. Dashboards are role-aware and protected by the JWT session.' },
    { title: 'What happens after booking?', text: 'The appointment is stored with the selected doctor in the backend and the doctor receives a new notification in the notifications feed.' },
    { title: 'How do I spot ransomware or fake messages?', text: 'Never open unexpected attachments, don’t follow suspicious links, and verify requests by calling your doctor or support line directly. Ransomware warnings or payment demands in emails and texts are almost always malicious.' },
    { title: 'What should I do about fake phone numbers?', text: 'If a caller asks for sensitive medical details, passwords, or payment information, hang up and contact your doctor through the official app or your known clinic number. Fake numbers and unknown callers should not be trusted.' },
    { title: 'What else should I look out for?', text: 'Check that emails and messages are from trusted sources, avoid sharing report files with unverified contacts, and use the official CarePulse site for uploads and appointments.' },
  ];

  return (
    <div className="space-y-6 rounded-3xl bg-white p-6 shadow-sm">
      <div>
        <h1 className="text-3xl font-bold">Help Center</h1>
        <p className="text-slate-500">Support information for patients, doctors, and administrators using CarePulse.</p>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {helpTopics.map((topic) => (
          <Card key={topic.title} title={topic.title} icon={<ShieldCheck size={18} />} text={topic.text} />
        ))}
      </div>
      <div className="rounded-2xl bg-slate-900 p-5 text-white">
        <h3 className="text-lg font-semibold">Support contact</h3>
        <p className="mt-2 text-sm text-slate-300">If you need direct assistance, contact the CarePulse operations desk at support@carepulse.demo or call +1 (555) 123-4567.</p>
        <p className="mt-2 text-sm text-slate-300">For urgent support, use the official app contact page or open the appointment and dashboard workflow from the top navigation.</p>
      </div>
    </div>
  );
}

function PatientPortal({ currentUser, dashboardData }) {
  const displayName = getDisplayName(currentUser);
  const recentReports = dashboardData?.recent_reports?.length
    ? dashboardData.recent_reports
    : [];
  const recentSummaries = dashboardData?.recent_summaries?.length
    ? dashboardData.recent_summaries
    : [];
  const patientAppointments = dashboardData?.appointments?.length
    ? dashboardData.appointments
    : [];

  return (
    <div className="space-y-6 rounded-3xl border border-cyan-100 bg-gradient-to-br from-cyan-50 to-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-cyan-900">Patient Dashboard</h2>
          <p className="text-slate-500">Welcome {displayName} — review your reports, summaries, and appointment activity.</p>
        </div>
        <Link to="/appointments" className="rounded-xl bg-cyan-50 px-4 py-2 text-sm font-semibold text-cyan-700">Book appointment</Link>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <Metric title="Reports" value={String(dashboardData?.stats?.reports_uploaded ?? 0)} />
        <Metric title="Summaries" value={String(dashboardData?.stats?.summaries_generated ?? 0)} />
        <Metric title="Appointments" value={String(dashboardData?.stats?.upcoming_appointments ?? 0)} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <h3 className="mb-3 text-lg font-semibold">Recent reports</h3>
          <div className="space-y-2 text-sm text-slate-600">
            {recentReports.length ? recentReports.map((report) => (
              <div key={report.id} className="rounded-xl bg-slate-50 px-3 py-2">{report.title} • {report.status}</div>
            )) : <div className="rounded-xl bg-slate-50 px-3 py-2">No reports uploaded yet.</div>}
          </div>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <h3 className="mb-3 text-lg font-semibold">Recent summaries</h3>
          <div className="space-y-2 text-sm text-slate-600">
            {recentSummaries.length ? recentSummaries.map((summary) => (
              <div key={summary.id} className="rounded-xl bg-slate-50 px-3 py-2">Summary ready: {summary.report__title}</div>
            )) : <div className="rounded-xl bg-slate-50 px-3 py-2">No summaries generated yet.</div>}
          </div>
        </div>
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <h3 className="mb-3 text-lg font-semibold">Appointment timeline</h3>
        <div className="space-y-2 text-sm text-slate-600">
          {patientAppointments.length ? patientAppointments.map((appointment) => (
            <div key={appointment.id} className="rounded-xl bg-slate-50 px-3 py-2">
              <div className="font-semibold text-slate-800">{appointment.doctor_name || 'Selected doctor'} • {appointment.status}</div>
              <div>{formatDisplayDate(appointment.appointment_date)} at {appointment.start_time}</div>
            </div>
          )) : <div className="rounded-xl bg-slate-50 px-3 py-2">No appointments booked yet.</div>}
        </div>
      </div>
    </div>
  );
}

function DoctorPortal({ currentUser, dashboardData, notifications }) {
  const displayName = getDisplayName(currentUser);
  const [selectedNotification, setSelectedNotification] = useState(null);
  const [selectedAppointment, setSelectedAppointment] = useState(null);
  const doctorAppointments = dashboardData?.appointments?.length ? dashboardData.appointments : [];

  useEffect(() => {
    if (!selectedAppointment && doctorAppointments.length) {
      setSelectedAppointment(doctorAppointments[0]);
    }
  }, [doctorAppointments, selectedAppointment]);

  return (
    <div className="space-y-6 rounded-3xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-emerald-900">Doctor Portal</h2>
          <p className="text-slate-500">Welcome {displayName} — today’s schedule, reports, and patient visibility.</p>
        </div>
        <div className="rounded-xl bg-emerald-50 px-4 py-2 text-sm font-semibold text-emerald-700">{dashboardData?.stats?.today_appointments ?? 12} appointments today</div>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <Metric title="Patients" value={String(dashboardData?.stats?.total_patients ?? 184)} />
        <Metric title="Pending Reports" value={String(dashboardData?.stats?.pending_reports ?? 7)} />
        <Metric title="Upcoming" value={String(dashboardData?.stats?.today_appointments ?? 12)} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-2xl border border-emerald-100 bg-white p-4">
          <h3 className="mb-3 text-lg font-semibold text-emerald-900">Today’s agenda</h3>
          <div className="space-y-2 text-sm text-slate-600">
            {doctorAppointments.length ? (
              doctorAppointments.map((appointment) => (
                <button
                  key={appointment.id}
                  type="button"
                  onClick={() => setSelectedAppointment(appointment)}
                  className={`w-full rounded-xl px-3 py-3 text-left transition ${selectedAppointment?.id === appointment.id ? 'bg-emerald-100 ring-1 ring-emerald-200' : 'bg-emerald-50 hover:bg-emerald-100'}`}
                >
                  <div className="font-semibold text-emerald-900">{appointment.patient_name || 'Unknown patient'}</div>
                  <div className="text-sm text-slate-700">{formatDisplayDate(appointment.appointment_date)} at {appointment.start_time} • {appointment.status}</div>
                  <div className="mt-1 text-sm text-slate-600">Issue: {appointment.notes || 'No issue details'}</div>
                </button>
              ))
            ) : (
              <div className="rounded-xl bg-emerald-50 px-3 py-2">No appointments scheduled yet.</div>
            )}
          </div>
        </div>
        <div className="rounded-2xl border border-emerald-100 bg-white p-4">
          <h3 className="mb-3 text-lg font-semibold text-emerald-900">Notifications</h3>
          <div className="grid gap-3 lg:grid-cols-[1fr_0.9fr]">
            <div className="space-y-2 text-sm text-slate-600">
              {notifications.length ? (
                notifications.map((notification) => (
                  <button
                    key={notification.id}
                    type="button"
                    onClick={() => setSelectedNotification(notification)}
                    className="w-full rounded-xl bg-emerald-50 px-3 py-2 text-left transition hover:bg-emerald-100"
                  >
                    <div className="font-semibold text-emerald-900">{notification.title}</div>
                    <div>{notification.message}</div>
                    <div className="text-xs text-slate-500">{formatDisplayDate(notification.created_at)}</div>
                  </button>
                ))
              ) : (
                <div className="rounded-xl bg-slate-50 px-3 py-2">No new notifications yet.</div>
              )}
            </div>
            <div className="rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
              <div className="mb-2 font-semibold text-slate-900">Selected notification</div>
              {selectedNotification ? (
                <div className="space-y-2">
                  <div><span className="font-semibold">Patient:</span> {selectedNotification.patient_name || 'Unknown patient'}</div>
                  <div><span className="font-semibold">Problem:</span> {selectedNotification.problem || selectedNotification.message || 'No details provided.'}</div>
                  <div><span className="font-semibold">Appointment:</span> {formatDisplayDate(selectedNotification.appointment_date)} at {selectedNotification.start_time || '—'} to {selectedNotification.end_time || '—'}</div>
                </div>
              ) : (
                <div>Select a notification to see the patient and problem details.</div>
              )}
            </div>
          </div>
          <div className="mt-4 rounded-xl border border-emerald-100 bg-emerald-50 p-3 text-sm text-slate-700">
            <div className="mb-2 font-semibold text-slate-900">Selected appointment</div>
            {selectedAppointment ? (
              <div className="space-y-2">
                <div><span className="font-semibold">Patient:</span> {selectedAppointment.patient_name || 'Unknown patient'}</div>
                <div><span className="font-semibold">Scheduled:</span> {formatDisplayDate(selectedAppointment.appointment_date)} at {selectedAppointment.start_time}</div>
                <div><span className="font-semibold">Status:</span> {selectedAppointment.status}</div>
                <div><span className="font-semibold">Issue:</span> {selectedAppointment.notes || 'No issue details provided.'}</div>
                {selectedAppointment.follow_up_recommendation ? (
                  <div><span className="font-semibold">Follow-up:</span> {selectedAppointment.follow_up_recommendation}</div>
                ) : null}
              </div>
            ) : (
              <div>Select an appointment to view patient issue details.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function OwnerPortal({ currentUser, dashboardData }) {
  const displayName = getDisplayName(currentUser);
  const metrics = [
    { title: 'Users', value: String(dashboardData?.stats?.total_users ?? 420) },
    { title: 'Patients', value: String(dashboardData?.stats?.total_patients ?? 310) },
    { title: 'Doctors', value: String(dashboardData?.stats?.total_doctors ?? 98) },
    { title: 'Today’s Appointments', value: String(dashboardData?.stats?.appointments_today ?? 52) },
  ];
  const activityItems = dashboardData?.activity?.length
    ? dashboardData.activity
    : [
        { actor: 'system', action: 'seeded_demo_data', details: 'Seed data refresh complete', created_at: new Date().toISOString() },
      ];
  const quickActions = [
    { title: 'Add New User', href: 'http://127.0.0.1:8000/admin/users/user/add/', description: 'Create new owner, staff, or patient credentials.' },
    { title: 'Add Doctor Profile', href: 'http://127.0.0.1:8000/admin/doctors/doctorprofile/add/', description: 'Register a new doctor and configure specialist details.' },
    { title: 'Add Patient Profile', href: 'http://127.0.0.1:8000/admin/patients/patientprofile/add/', description: 'Create a patient profile with contact and emergency information.' },
    { title: 'Add Appointment', href: 'http://127.0.0.1:8000/admin/appointments/appointment/add/', description: 'Manually create or review appointment records.' },
  ];

  return (
    <div className="space-y-6 rounded-3xl border border-indigo-100 bg-gradient-to-br from-indigo-50 to-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-indigo-900">Owner Control Center</h2>
          <p className="text-slate-500">Welcome {displayName} — use the Django administration UI to manage users, doctors, patients, appointments, reports, and full site settings.</p>
        </div>
        <a href="http://127.0.0.1:8000/admin/" target="_blank" rel="noreferrer" className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white">Open Django Admin</a>
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        {metrics.map((metric) => (
          <Metric key={metric.title} title={metric.title} value={metric.value} />
        ))}
      </div>
      <div className="rounded-2xl border border-indigo-100 bg-white p-4 text-sm text-slate-600">
        Recent operations: {dashboardData?.stats?.reports_processed ?? 14} medical reports processed, {dashboardData?.stats?.pending_appointments ?? 3} new appointment requests, {dashboardData?.stats?.summaries_generated ?? 2} summaries generated.
      </div>
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-2xl border border-indigo-100 bg-slate-900 p-4 text-white">
          <h3 className="mb-3 text-lg font-semibold">Superuser credential</h3>
          <div className="space-y-2 text-sm text-slate-200">
            <div><span className="font-semibold text-white">Username:</span> {currentUser?.username || 'admin'}</div>
            <div><span className="font-semibold text-white">Email:</span> {currentUser?.email || 'admin@example.com'}</div>
            <div><span className="font-semibold text-white">Role:</span> Owner / Superuser</div>
          </div>
        </div>
        <div className="rounded-2xl border border-indigo-100 bg-white p-4">
          <h3 className="mb-3 text-lg font-semibold text-indigo-900">Recent activity</h3>
          <div className="space-y-2 text-sm text-slate-600">
            {activityItems.map((item) => (
              <div key={`${item.actor}-${item.action}-${item.created_at}`} className="rounded-xl bg-indigo-50 px-3 py-2">
                <div className="font-medium text-slate-800">{item.actor}</div>
                <div>{item.action} • {item.details}</div>
                <div className="text-xs text-slate-500">{formatDisplayDate(item.created_at)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="rounded-2xl border border-indigo-100 bg-white p-4">
        <h3 className="mb-3 text-lg font-semibold text-indigo-900">Owner actions</h3>
        <div className="grid gap-3 md:grid-cols-2">
          {quickActions.map((action) => (
            <a key={action.title} href={action.href} target="_blank" rel="noreferrer" className="rounded-2xl border border-indigo-100 bg-indigo-50 p-4 text-sm text-slate-700 transition hover:bg-indigo-100">
              <div className="font-semibold text-indigo-900">{action.title}</div>
              <div className="mt-1">{action.description}</div>
            </a>
          ))}
        </div>
      </div>
      <div className="rounded-2xl border border-indigo-100 bg-white p-4">
        <h3 className="mb-3 text-lg font-semibold text-indigo-900">Owner tools</h3>
        <div className="grid gap-3 md:grid-cols-3">
          <a href="http://127.0.0.1:8000/admin/" target="_blank" rel="noreferrer" className="rounded-2xl bg-slate-900 p-4 text-sm font-semibold text-white">Open full Django admin</a>
          <a href="http://127.0.0.1:8000/swagger/" target="_blank" rel="noreferrer" className="rounded-2xl bg-cyan-600 p-4 text-sm font-semibold text-white">Open API docs</a>
          <a href="http://127.0.0.1:8000/redoc/" target="_blank" rel="noreferrer" className="rounded-2xl bg-emerald-600 p-4 text-sm font-semibold text-white">Open API reference</a>
        </div>
      </div>
    </div>
  );
}

function AuthPortal({ onAuthChange }) {
  const navigate = useNavigate();
  const [mode, setMode] = useState('login');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
    first_name: '',
    last_name: '',
    phone: '',
    role: 'patient',
    password2: '',
  });
  const [message, setMessage] = useState('');

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({
      ...current,
      [name]: value,
      ...(name === 'email' && !current.username ? { username: value } : {}),
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setMessage('');

    try {
      if (mode === 'login') {
        const response = await loginUser({ username: form.username, password: form.password });
        const token = response.data.access;
        const user = response.data.user;
        setAuthToken(token);
        localStorage.setItem('carepulse_token', token);
        localStorage.setItem('carepulse_user', JSON.stringify(user));
        onAuthChange(user);
        setMessage(`Logged in as ${[user.first_name, user.last_name].filter(Boolean).join(' ') || user.username} (${user.role}).`);
        const nextRoute = getDashboardRoute(user);
        navigate(nextRoute);
      } else {
        const username = (form.username || form.email).trim();
        await registerUser({
          username,
          email: form.email,
          first_name: form.first_name,
          last_name: form.last_name,
          phone: form.phone,
          role: form.role,
          password: form.password,
          password2: form.password2,
        });
        setForm((current) => ({ ...current, username }));
        setMode('login');
        setMessage('Registration succeeded. You can now log in.');
      }
    } catch (error) {
      const backendMessage = error?.response?.data?.detail || error?.response?.data?.non_field_errors?.[0] || error?.response?.data?.password || error?.response?.data?.username?.[0] || 'Authentication failed.';
      setMessage(backendMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
      <div className="rounded-3xl bg-white p-6 shadow-sm">
        <div className="mb-4 flex gap-2">
          <button type="button" onClick={() => setMode('login')} className={`rounded-full px-4 py-2 text-sm font-semibold ${mode === 'login' ? 'bg-cyan-500 text-white' : 'bg-slate-100 text-slate-700'}`}>Login</button>
          <button type="button" onClick={() => setMode('register')} className={`rounded-full px-4 py-2 text-sm font-semibold ${mode === 'register' ? 'bg-cyan-500 text-white' : 'bg-slate-100 text-slate-700'}`}>Register</button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === 'register' ? (
            <>
              <div className="grid gap-4 md:grid-cols-2">
                <input name="first_name" value={form.first_name} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder="First name" />
                <input name="last_name" value={form.last_name} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder="Last name" />
              </div>
              <input name="email" type="email" value={form.email} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder="Email" />
              <input name="phone" value={form.phone} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder="Phone" />
              <select name="role" value={form.role} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2">
                <option value="patient">Patient</option>
                <option value="doctor">Doctor</option>
                <option value="admin">Owner</option>
              </select>
            </>
          ) : null}

          {form.role === 'admin' ? (
            <div className="rounded-xl bg-indigo-50 px-3 py-2 text-sm text-indigo-800">
              Owner access requires the Django superuser credentials for the website owner account.
            </div>
          ) : null}

          <input name="username" value={form.username} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder={form.role === 'admin' ? 'Superuser username or email' : mode === 'register' ? 'Username (auto-filled from email)' : 'Username or email'} />
          <input name="password" type="password" value={form.password} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder={form.role === 'admin' ? 'Superuser password' : 'Password'} />

          {mode === 'register' ? (
            <input name="password2" type="password" value={form.password2} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder="Confirm password" />
          ) : null}

          <button type="submit" disabled={isSubmitting} className="rounded-full bg-cyan-500 px-5 py-3 font-semibold text-white disabled:cursor-not-allowed disabled:bg-cyan-300">{isSubmitting ? (mode === 'login' ? 'Signing in...' : 'Creating account...') : (mode === 'login' ? 'Login' : 'Create account')}</button>
          {message ? <p className="text-sm text-slate-700">{message}</p> : null}
        </form>
      </div>

      <div className="rounded-3xl bg-slate-900 p-6 text-white shadow-sm">
        <h2 className="text-2xl font-bold">Patient access preview</h2>
        <p className="mt-3 text-slate-300">Use the auth screen to log in with a seeded account and then move into the appointment flow using the real JWT-backed API.</p>
        <div className="mt-4 rounded-2xl bg-white/10 p-4 text-sm">
          Demo credentials:<br />
          Owner: admin / adminpass<br />
          Doctor: doctor1 / doctorpass<br />
          Patient: patient1 / patientpass
        </div>
      </div>
    </div>
  );
}

function AppointmentPortal({ currentUser, dashboardData, setDashboardData }) {
  const displayName = getDisplayName(currentUser);
  const recentReports = dashboardData?.recent_reports?.length
    ? dashboardData.recent_reports
    : [];
  const recentSummaries = dashboardData?.recent_summaries?.length
    ? dashboardData.recent_summaries
    : [];
  const appointments = dashboardData?.appointments?.length
    ? dashboardData.appointments
    : [];
  const [form, setForm] = useState({
    doctor: '',
    appointment_date: '',
    start_time: '09:00',
    end_time: '09:30',
    notes: '',
  });
  const [message, setMessage] = useState('');
  const [doctors, setDoctors] = useState([]);

  useEffect(() => {
    fetchDoctors()
      .then((response) => setDoctors(response.data.results || response.data))
      .catch(() => setDoctors([]));
  }, []);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const token = localStorage.getItem('carepulse_token');
      if (!token) {
        setMessage('Please log in first to create an appointment.');
        return;
      }

      if (!form.doctor || !form.appointment_date) {
        setMessage('Please choose a doctor and a date before creating the appointment.');
        return;
      }

      setAuthToken(token);
      const response = await createAppointment({
        doctor: Number(form.doctor),
        appointment_date: form.appointment_date,
        start_time: form.start_time,
        end_time: form.end_time,
        notes: form.notes,
      });

      const dashboardResponse = await fetchDashboard();
      const selectedDoctor = doctors.find((doctor) => Number(doctor.id) === Number(response.data.doctor));
      setDashboardData(dashboardResponse.data);
      setMessage(`Appointment created successfully with ${selectedDoctor?.doctor_name || 'the selected doctor'}.`);
      setForm({
        doctor: '',
        appointment_date: '',
        start_time: '09:00',
        end_time: '09:30',
        notes: '',
      });
    } catch (error) {
      setMessage(error?.response?.data?.detail || 'Unable to create appointment.');
    }
  };

  return (
    <div className="space-y-6 rounded-3xl border border-cyan-100 bg-gradient-to-br from-cyan-50 to-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-cyan-900">Book an Appointment</h2>
          <p className="text-slate-500">Welcome {displayName} — choose a doctor and reserve your next consultation slot.</p>
        </div>
        <Link to="/doctor" className="rounded-xl bg-cyan-50 px-4 py-2 text-sm font-semibold text-cyan-700">View doctor schedule</Link>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_0.9fr]">
        <form onSubmit={handleSubmit} className="space-y-4 rounded-2xl border border-slate-200 p-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-700">Doctor</label>
            <select name="doctor" value={form.doctor} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2">
              <option value="">Select a doctor</option>
              {doctors.map((doctor) => (
                <option key={doctor.id} value={doctor.id}>{doctor.doctor_name} — {doctor.specialization?.name || 'General'}</option>
              ))}
            </select>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-700">Date</label>
              <input type="date" name="appointment_date" value={form.appointment_date} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-700">Start time</label>
              <input type="time" name="start_time" value={form.start_time} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" />
            </div>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-slate-700">End time</label>
            <input type="time" name="end_time" value={form.end_time} onChange={handleChange} className="w-full rounded-xl border border-slate-300 px-3 py-2" />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-slate-700">Notes</label>
            <textarea name="notes" value={form.notes} onChange={handleChange} rows="3" className="w-full rounded-xl border border-slate-300 px-3 py-2" placeholder="Reason for visit or symptoms" />
          </div>

          <button type="submit" className="rounded-full bg-cyan-500 px-5 py-3 font-semibold text-white">Create Appointment</button>
          {message ? <p className="text-sm text-emerald-700">{message}</p> : null}
        </form>

        <div className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-1">
            {doctors.slice(0, 3).map((doctor) => (
              <Card key={doctor.id} title={doctor.doctor_name} icon={<Stethoscope size={18} />} text={`${doctor.specialization?.name || 'General'} • ${doctor.hospital || 'CarePulse Hospital'}`} />
            ))}
          </div>

          <div className="rounded-2xl border border-slate-200 p-4">
            <h3 className="mb-3 text-lg font-semibold">Recent requests</h3>
            <div className="space-y-2 text-sm text-slate-600">
              {appointments.map((appointment) => (
                <div key={`${appointment.patientName}-${appointment.date}-${appointment.time}`} className="rounded-xl bg-slate-50 px-3 py-2">
                  {appointment.patientName} • {appointment.doctor} • {appointment.date} at {appointment.time}
                </div>
              ))}
            </div>
            <div className="mt-3 rounded-xl bg-cyan-50 px-3 py-2 text-cyan-800">
              Upcoming appointments: {dashboardData?.stats?.upcoming_appointments ?? 0}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 p-4">
            <h3 className="mb-3 text-lg font-semibold">Patient snapshot</h3>
            <div className="grid gap-2 text-sm text-slate-600 md:grid-cols-3">
              <Metric title="Reports" value={String(dashboardData?.stats?.reports_uploaded ?? 0)} />
              <Metric title="Summaries" value={String(dashboardData?.stats?.summaries_generated ?? 0)} />
              <Metric title="Appointments" value={String(dashboardData?.stats?.upcoming_appointments ?? 0)} />
            </div>
            <div className="mt-3 space-y-2">
              {recentReports.map((report) => (
                <div key={report.id} className="rounded-xl bg-slate-50 px-3 py-2">
                  {report.title} • {report.status}
                </div>
              ))}
              {recentSummaries.map((summary) => (
                <div key={summary.id} className="rounded-xl bg-slate-50 px-3 py-2">
                  Summary ready: {summary.report__title}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Card({ title, icon, text }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-3 inline-flex rounded-xl bg-cyan-50 p-2 text-cyan-700">{icon}</div>
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mt-2 text-sm text-slate-600">{text}</p>
    </div>
  );
}

function Metric({ title, value }) {
  return (
    <div className="rounded-2xl bg-slate-50 p-4">
      <div className="text-sm text-slate-500">{title}</div>
      <div className="mt-2 text-2xl font-bold">{value}</div>
    </div>
  );
}

export default App;
