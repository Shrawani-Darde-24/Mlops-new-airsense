const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`);
  }
  return body;
}

export const getDatasetVersionStats = () =>
  request('/api/v1/dataset-versions/stats');

export const getDatasetVersionHistory = () =>
  request('/api/v1/dataset-versions/history');

export const getDatasetChangeHistory = () =>
  request('/api/v1/dataset-versions/changes');

export const syncDatasetVersions = () =>
  request('/api/v1/dataset-versions/sync', { method: 'POST' });
