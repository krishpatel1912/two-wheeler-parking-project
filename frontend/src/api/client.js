import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api/v1` 
  : '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120000,  // 2 minutes — AI detection can take 5-15s per image
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('parking_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const api = {
  // Auth
  login: async (username, password) => {
    const res = await apiClient.post('/auth/login/json', { username, password });
    if (res.data.access_token) {
      localStorage.setItem('parking_token', res.data.access_token);
      localStorage.setItem('parking_user', JSON.stringify(res.data));
    }
    return res.data;
  },
  logout: () => {
    localStorage.removeItem('parking_token');
    localStorage.removeItem('parking_user');
  },
  getMe: () => apiClient.get('/auth/me'),

  // Dashboard
  getStats: () => apiClient.get('/dashboard/stats'),
  getTrend: (days = 30) => apiClient.get(`/dashboard/trend?days=${days}`),
  getHeatmap: () => apiClient.get('/dashboard/heatmap'),

  // Violations
  getViolations: (params = {}) => apiClient.get('/violations', { params }),
  getViolationDetail: (id) => apiClient.get(`/violations/${id}`),
  reviewViolation: (id, action, notes) => 
    apiClient.patch(`/violations/${id}/review`, { action, notes }),

  // Zones
  getZones: () => apiClient.get('/zones'),
  createOrUpdateZone: (data) => apiClient.post('/zones', data),
  deleteZone: (zoneId) => apiClient.delete(`/zones/${zoneId}`),
  syncZones: () => apiClient.post('/zones/sync'),

  // Cameras
  getCameras: () => apiClient.get('/cameras'),
  createCamera: (data) => apiClient.post('/cameras', data),

  // Processing
  uploadFootage: (formData) => 
    apiClient.post('/processing/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  detectPhoto: (formData) =>
    apiClient.post('/processing/detect-photo', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  getJobs: () => apiClient.get('/processing/jobs'),
  getJobStatus: (id) => apiClient.get(`/processing/jobs/${id}`),

  // Health
  getHealth: () => apiClient.get('/health'),

  // Helper for evidence asset URL
  getEvidenceUrl: (filenameOrPath) => {
    if (!filenameOrPath) return '';
    if (filenameOrPath.startsWith('http://') || filenameOrPath.startsWith('https://')) {
      return filenameOrPath;
    }
    const cleanPath = filenameOrPath.replace(/^\/+/, '');
    // If it points to data/evidence/filename.jpg, strip to filename
    const parts = cleanPath.split('/');
    const fname = parts[parts.length - 1];
    return `${API_BASE}/evidence/${fname}`;
  },
};

export default apiClient;
