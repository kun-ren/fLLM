import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Card,
  CardContent,
  Checkbox,
  Alert,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
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
import {
  listTrainingSessions,
  getTrainingSession,
  deleteTrainingSession,
  compareSessions,
} from '../api';

function TrainHistoryPanel() {
  const [sessions, setSessions] = useState([]);
  const [selectedSessions, setSelectedSessions] = useState([]);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  const [compareDialogOpen, setCompareDialogOpen] = useState(false);
  const [currentSession, setCurrentSession] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const response = await listTrainingSessions();
      setSessions(response.data.sessions);
    } catch (err) {
      setError('Failed to load training sessions');
    }
  };

  const handleViewDetails = async (sessionId) => {
    try {
      const response = await getTrainingSession(sessionId);
      setCurrentSession(response.data);
      setDetailDialogOpen(true);
    } catch (err) {
      setError('Failed to load session details');
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!window.confirm('Are you sure you want to delete this training session?')) {
      return;
    }

    try {
      await deleteTrainingSession(sessionId);
      loadSessions();
    } catch (err) {
      setError('Failed to delete session');
    }
  };

  const handleToggleSelect = (sessionId) => {
    setSelectedSessions((prev) =>
      prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : [...prev, sessionId]
    );
  };

  const handleCompare = async () => {
    if (selectedSessions.length < 2) {
      setError('Please select at least 2 sessions to compare');
      return;
    }

    try {
      const response = await compareSessions({ session_ids: selectedSessions });
      setComparisonData(response.data.sessions);
      setCompareDialogOpen(true);
    } catch (err) {
      setError('Failed to compare sessions');
    }
  };

  const formatDate = (isoString) => {
    return new Date(isoString).toLocaleString();
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

      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Training History</Typography>
          <Button
            variant="contained"
            startIcon={<CompareArrowsIcon />}
            onClick={handleCompare}
            disabled={selectedSessions.length < 2}
          >
            Compare Selected ({selectedSessions.length})
          </Button>
        </Box>

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={
                      selectedSessions.length > 0 && selectedSessions.length < sessions.length
                    }
                    checked={sessions.length > 0 && selectedSessions.length === sessions.length}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedSessions(sessions.map((s) => s.session_id));
                      } else {
                        setSelectedSessions([]);
                      }
                    }}
                  />
                </TableCell>
                <TableCell>Session ID</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell align="right">Epochs</TableCell>
                <TableCell align="right">Final Loss</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sessions.map((session) => (
                <TableRow key={session.session_id} hover>
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedSessions.includes(session.session_id)}
                      onChange={() => handleToggleSelect(session.session_id)}
                    />
                  </TableCell>
                  <TableCell>
                    <Chip label={session.session_id} size="small" />
                  </TableCell>
                  <TableCell>{formatDate(session.timestamp)}</TableCell>
                  <TableCell align="right">{session.epochs}</TableCell>
                  <TableCell align="right">
                    <Chip
                      label={formatNumber(session.final_loss)}
                      color={session.final_loss < 0.01 ? 'success' : 'default'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="center">
                    <IconButton
                      size="small"
                      onClick={() => handleViewDetails(session.session_id)}
                      color="primary"
                    >
                      <VisibilityIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDeleteSession(session.session_id)}
                      color="error"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {sessions.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography color="text.secondary">No training sessions found</Typography>
          </Box>
        )}
      </Paper>

      {/* Session Detail Dialog */}
      <Dialog
        open={detailDialogOpen}
        onClose={() => setDetailDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>Training Session Details</DialogTitle>
        <DialogContent>
          {currentSession && (
            <Box>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={4}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Session ID
                      </Typography>
                      <Typography variant="h6">{currentSession.session_id}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Epochs
                      </Typography>
                      <Typography variant="h6">{currentSession.epochs}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Final Loss
                      </Typography>
                      <Typography variant="h6">{formatNumber(currentSession.final_loss)}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              <Paper sx={{ p: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Loss Curve
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart
                    data={currentSession.history
                      .filter((h) => h.loss > 0)
                      .map((h) => ({ step: h.step, loss: h.loss, epoch: h.epoch }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#8884d8" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Paper>

              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Configuration
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(currentSession.config || {})
                    .slice(0, 12)
                    .map(([key, param]) => (
                      <Grid item xs={6} md={4} key={key}>
                        <Typography variant="caption" color="text.secondary">
                          {key}
                        </Typography>
                        <Typography variant="body2">{String(param.value)}</Typography>
                      </Grid>
                    ))}
                </Grid>
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Comparison Dialog */}
      <Dialog
        open={compareDialogOpen}
        onClose={() => setCompareDialogOpen(false)}
        maxWidth="xl"
        fullWidth
      >
        <DialogTitle>Session Comparison</DialogTitle>
        <DialogContent>
          {comparisonData && (
            <Box>
              <Paper sx={{ p: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Loss Comparison
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" type="number" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {comparisonData.map((session, idx) => {
                      const data = session.history
                        .filter((h) => h.loss > 0)
                        .map((h) => ({ step: h.step, loss: h.loss }));
                      const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];
                      return (
                        <Line
                          key={session.session_id}
                          data={data}
                          type="monotone"
                          dataKey="loss"
                          stroke={colors[idx % colors.length]}
                          name={session.session_id}
                          dot={false}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </Paper>

              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Session ID</TableCell>
                      <TableCell align="right">Epochs</TableCell>
                      <TableCell align="right">Final Loss</TableCell>
                      <TableCell align="right">Batch Size</TableCell>
                      <TableCell align="right">Learning Rate</TableCell>
                      <TableCell align="right">d_model</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {comparisonData.map((session) => (
                      <TableRow key={session.session_id}>
                        <TableCell>
                          <Chip label={session.session_id} size="small" />
                        </TableCell>
                        <TableCell align="right">{session.epochs}</TableCell>
                        <TableCell align="right">{formatNumber(session.final_loss)}</TableCell>
                        <TableCell align="right">{session.config?.batch_size?.value}</TableCell>
                        <TableCell align="right">{session.config?.lr?.value}</TableCell>
                        <TableCell align="right">{session.config?.d_model?.value}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCompareDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default TrainHistoryPanel;
