import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  LinearProgress,
  Card,
  CardContent,
  Alert,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { startTraining, stopTraining, getTrainingStatus, getTrainingProgress, getMetricsHistory } from '../api';

function TrainingDashboard() {
  const [trainingActive, setTrainingActive] = useState(false);
  const [status, setStatus] = useState({
    status: 'idle',
    epoch: 0,
    step: 0,
    loss: 0,
    progress: 0,
    log: 'Ready to start training',
  });
  const [progress, setProgress] = useState({
    status: 'idle',
    progress: 0,
    epoch: 0,
    total_epochs: 0,
    current_loss: 0,
  });
  const [lossHistory, setLossHistory] = useState([]);
  const [error, setError] = useState(null);
  const eventSourceRef = useRef(null);
  const progressIntervalRef = useRef(null);

  useEffect(() => {
    checkTrainingStatus();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  const checkTrainingStatus = async () => {
    try {
      const response = await getTrainingStatus();
      const currentStatus = response.data;
      setStatus(currentStatus);
      setTrainingActive(currentStatus.status === 'training' || currentStatus.status === 'loading');

      if (currentStatus.status === 'training') {
        connectToStream();
        startProgressPolling();
      }
    } catch (err) {
      console.error('Failed to check training status:', err);
    }
  };

  const startProgressPolling = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }

    progressIntervalRef.current = setInterval(async () => {
      try {
        const response = await getTrainingProgress();
        const progressData = response.data;
        setProgress(progressData);

        // Stop polling if training is finished
        if (progressData.status === 'finished' || progressData.status === 'error' || progressData.status === 'idle') {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
          setTrainingActive(false);
        }
      } catch (err) {
        console.error('Failed to fetch training progress:', err);
      }
    }, 1000); // Poll every second
  };

  const stopProgressPolling = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  const connectToStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource('http://localhost:5000/api/training/stream');
    eventSourceRef.current = eventSource;

    eventSource.addEventListener('training', (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);

      if (data.loss > 0) {
        setLossHistory((prev) => [
          ...prev,
          { step: data.step, loss: data.loss, epoch: data.epoch },
        ]);
      }
    });

    eventSource.addEventListener('completed', (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);
      setTrainingActive(false);
      eventSource.close();
    });

    eventSource.addEventListener('error', (event) => {
      console.error('SSE error:', event);
      eventSource.close();
      setTrainingActive(false);
    });

    eventSource.onerror = () => {
      eventSource.close();
      setTrainingActive(false);
    };
  };

  const handleStartTraining = async () => {
    try {
      setError(null);
      setLossHistory([]);
      await startTraining();
      setTrainingActive(true);
      connectToStream();
      startProgressPolling();
    } catch (err) {
      setError('Failed to start training: ' + err.message);
    }
  };

  const handleStopTraining = async () => {
    try {
      await stopTraining();
      setTrainingActive(false);
      stopProgressPolling();
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    } catch (err) {
      setError('Failed to stop training: ' + err.message);
    }
  };

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <Typography variant="h6" gutterBottom>
              Training Control
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {status.log}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<PlayArrowIcon />}
                onClick={handleStartTraining}
                disabled={trainingActive}
                fullWidth
              >
                Start Training
              </Button>
              <Button
                variant="contained"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleStopTraining}
                disabled={!trainingActive}
                fullWidth
              >
                Stop
              </Button>
            </Box>
          </Grid>
        </Grid>

        {trainingActive && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress variant="determinate" value={progress.progress * 100} />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Progress: {(progress.progress * 100).toFixed(1)}% - Epoch {progress.epoch}/{progress.total_epochs}
            </Typography>
          </Box>
        )}
      </Paper>

      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Epoch
              </Typography>
              <Typography variant="h4">{status.epoch}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Step
              </Typography>
              <Typography variant="h4">{status.step}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Loss
              </Typography>
              <Typography variant="h4">{status.loss.toFixed(6)}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Status
              </Typography>
              <Typography variant="h5" sx={{ textTransform: 'capitalize' }}>
                {status.status}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Loss Curve
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={lossHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
            />
            <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#8884d8"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
}

export default TrainingDashboard;
