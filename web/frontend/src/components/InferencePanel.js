import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Alert,
  Chip,
  TextField,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import {
  loadInferenceModel,
  unloadInferenceModel,
  listLoadedModels,
  setActiveInferenceModel,
  runInference,
  getInferenceModelInfo,
  listModelCheckpoints,
} from '../api';

function InferencePanel() {
  const [availableModels, setAvailableModels] = useState([]);
  const [loadedModels, setLoadedModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Sample input for testing
  const [sampleInput, setSampleInput] = useState('');
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    loadAvailableModels();
    loadLoadedModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const response = await listModelCheckpoints();
      setAvailableModels(response.data.checkpoints);
      if (response.data.checkpoints.length > 0) {
        setSelectedCheckpoint(response.data.checkpoints[0].path);
      }
    } catch (err) {
      setError('Failed to load available models');
    }
  };

  const loadLoadedModels = async () => {
    try {
      const response = await listLoadedModels();
      setLoadedModels(response.data.models);
      setActiveModel(response.data.active_model);
    } catch (err) {
      setError('Failed to load loaded models');
    }
  };

  const handleLoadModel = async () => {
    if (!selectedCheckpoint) {
      setError('Please select a model checkpoint');
      return;
    }

    try {
      setError(null);
      const response = await loadInferenceModel({
        model_path: selectedCheckpoint,
      });
      setSuccess(`Model loaded: ${response.data.model_name}`);
      loadLoadedModels();

      // Load model info
      const infoResponse = await getInferenceModelInfo(response.data.model_name);
      setModelInfo(infoResponse.data);
    } catch (err) {
      setError('Failed to load model: ' + err.message);
    }
  };

  const handleUnloadModel = async (modelName) => {
    try {
      await unloadInferenceModel({ model_name: modelName });
      setSuccess(`Model unloaded: ${modelName}`);
      loadLoadedModels();
      if (activeModel === modelName) {
        setModelInfo(null);
      }
    } catch (err) {
      setError('Failed to unload model: ' + err.message);
    }
  };

  const handleSetActive = async (modelName) => {
    try {
      await setActiveInferenceModel({ model_name: modelName });
      setSuccess(`Active model set to: ${modelName}`);
      loadLoadedModels();

      // Load model info
      const infoResponse = await getInferenceModelInfo(modelName);
      setModelInfo(infoResponse.data);
    } catch (err) {
      setError('Failed to set active model: ' + err.message);
    }
  };

  const handleRunInference = async () => {
    if (!activeModel) {
      setError('No active model selected');
      return;
    }

    if (!sampleInput) {
      setError('Please provide input data');
      return;
    }

    try {
      setError(null);
      // Parse input as JSON array
      const inputData = JSON.parse(sampleInput);

      const response = await runInference({
        input_data: inputData,
      });

      setPrediction(response.data.predictions);
      setSuccess('Inference completed successfully');
    } catch (err) {
      setError('Failed to run inference: ' + err.message);
    }
  };

  const formatNumber = (num, decimals = 4) => {
    return num?.toFixed(decimals) || '0.0000';
  };

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Load Model
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Select Checkpoint</InputLabel>
              <Select
                value={selectedCheckpoint}
                label="Select Checkpoint"
                onChange={(e) => setSelectedCheckpoint(e.target.value)}
              >
                {availableModels.map((model) => (
                  <MenuItem key={model.path} value={model.path}>
                    {model.name} ({formatNumber(model.size_mb, 2)} MB)
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button
              variant="contained"
              startIcon={<CloudUploadIcon />}
              onClick={handleLoadModel}
              fullWidth
            >
              Load Model
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Loaded Models
            </Typography>

            {loadedModels.length === 0 ? (
              <Typography color="text.secondary">No models loaded</Typography>
            ) : (
              <Box>
                {loadedModels.map((modelName) => (
                  <Box
                    key={modelName}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      mb: 1,
                      p: 1,
                      border: '1px solid',
                      borderColor: activeModel === modelName ? 'primary.main' : 'divider',
                      borderRadius: 1,
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography>{modelName}</Typography>
                      {activeModel === modelName && (
                        <Chip label="Active" color="primary" size="small" />
                      )}
                    </Box>
                    <Box>
                      {activeModel !== modelName && (
                        <Button
                          size="small"
                          onClick={() => handleSetActive(modelName)}
                        >
                          Set Active
                        </Button>
                      )}
                      <Button
                        size="small"
                        color="error"
                        startIcon={<DeleteIcon />}
                        onClick={() => handleUnloadModel(modelName)}
                      >
                        Unload
                      </Button>
                    </Box>
                  </Box>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {modelInfo && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Model Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Encoder Parameters
                      </Typography>
                      <Typography variant="h6">
                        {(modelInfo.encoder_params / 1e6).toFixed(2)}M
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Task Head Parameters
                      </Typography>
                      <Typography variant="h6">
                        {(modelInfo.taskheads_params / 1e6).toFixed(2)}M
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Device
                      </Typography>
                      <Typography variant="h6">{modelInfo.device}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        d_model
                      </Typography>
                      <Typography variant="h6">
                        {modelInfo.hyperparams?.d_model || 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        )}

        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Run Inference
            </Typography>

            <TextField
              fullWidth
              multiline
              rows={6}
              label="Input Data (JSON array)"
              placeholder='[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]'
              value={sampleInput}
              onChange={(e) => setSampleInput(e.target.value)}
              sx={{ mb: 2 }}
            />

            <Button
              variant="contained"
              startIcon={<PlayArrowIcon />}
              onClick={handleRunInference}
              disabled={!activeModel}
              fullWidth
            >
              Run Inference
            </Button>

            {prediction && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Prediction Results
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Reversal Confidence
                        </Typography>
                        <Typography
                          variant="h4"
                          color={
                            prediction.reversal_confidence > 0
                              ? 'success.main'
                              : prediction.reversal_confidence < 0
                              ? 'error.main'
                              : 'text.primary'
                          }
                        >
                          {formatNumber(prediction.reversal_confidence, 3)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {prediction.reversal_confidence > 0 ? 'Bullish' : 'Bearish'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Support Level
                        </Typography>
                        <Typography variant="h4">
                          {formatNumber(prediction.support_level, 2)} bps
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Resistance Level
                        </Typography>
                        <Typography variant="h4">
                          {formatNumber(prediction.resistance_level, 2)} bps
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default InferencePanel;
