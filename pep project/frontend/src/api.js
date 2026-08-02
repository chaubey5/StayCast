import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export function setAuthToken(token) {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common.Authorization;
  }
}

export const loginUser = (payload) => api.post('/api/auth/login/', payload);
export const registerUser = (payload) => api.post('/api/auth/register/', payload);
export const fetchDoctors = () => api.get('/api/doctors/');
export const fetchDashboard = () => api.get('/api/analytics/dashboard/');
export const fetchNotifications = () => api.get('/api/notifications/');
export const createAppointment = (payload) => api.post('/api/appointments/', payload);
