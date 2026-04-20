import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Config endpoints
export const getConfig = () => api.get('/config');
export const getConfigParam = (name) => api.get(`/config/${name}`);
export const updateConfigParam = (name, data) => api.put(`/config/${name}`, data);
export const saveConfig = (name) => api.post('/config/save', { name });
export const loadConfig = (name) => api.post('/config/load', { name });
export const listPresets = () => api.get('/config/presets');
export const deletePreset = (name) => api.delete(`/config/presets/${name}`);

// Training endpoints
export const startTraining = () => api.post('/training/start');
export const stopTraining = () => api.post('/training/stop');
export const getTrainingStatus = () => api.get('/training/status');
export const getTrainingProgress = () => api.get('/training/progress');

// Metrics endpoints
export const getMetricsHistory = (params) => api.get('/metrics/history', { params });
export const getMetricsSummary = () => api.get('/metrics/summary');

// System endpoints
export const getSystemInfo = () => api.get('/system/info');

// Training history endpoints
export const listTrainingSessions = () => api.get('/history/sessions');
export const getTrainingSession = (sessionId) => api.get(`/history/sessions/${sessionId}`);
export const deleteTrainingSession = (sessionId) => api.delete(`/history/sessions/${sessionId}`);
export const compareSessions = (data) => api.post('/history/compare', data);

// Backtest endpoints
export const startBacktest = (data) => api.post('/backtest/start', data);
export const getBacktestStatus = () => api.get('/backtest/status');
export const getBacktestResult = () => api.get('/backtest/result');

// Model checkpoint endpoints
export const listModelCheckpoints = () => api.get('/models/list');
export const deleteModelCheckpoint = (name) => api.delete(`/models/${name}`);

// Inference endpoints
export const loadInferenceModel = (data) => api.post('/inference/load', data);
export const unloadInferenceModel = (data) => api.post('/inference/unload', data);
export const listLoadedModels = () => api.get('/inference/models');
export const setActiveInferenceModel = (data) => api.post('/inference/set-active', data);
export const runInference = (data) => api.post('/inference/predict', data);
export const runBatchInference = (data) => api.post('/inference/predict-batch', data);
export const getInferenceModelInfo = (modelName) => api.get('/inference/model-info', { params: { model_name: modelName } });

export default api;

