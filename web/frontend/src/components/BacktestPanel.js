import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  CircularProgress,
  Slider,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { startBacktest, getBacktestStatus, getBacktestResult, listModelCheckpoints } from '../api';

function BacktestPanel() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [backtestRunning, setBacktestRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Backtest parameters
  const [params, setParams] = useState({
    confidence_threshold: 0.6,
    take_profit_bps: 5.0,
    stop_loss_bps: 10.0,
    max_hold_periods: 20,
    commission_rate: 0.0004,
  });

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    let interval;
    if (backtestRunning) {
      interval = setInterval(checkBacktestStatus, 2000);
    }
    return () => clearInterval(interval);
  }, [backtestRunning]);

  const loadModels = async () => {
    try {
      const response = await listModelCheckpoints();
      setModels(response.data.checkpoints);
      if (response.data.checkpoints.length > 0) {
        setSelectedModel(response.data.checkpoints[0].path);
      }
    } catch (err) {
      setError('Failed to load model checkpoints');
    }
  };

  const checkBacktestStatus = async () => {
    try {
      const response = await getBacktestStatus();
      const status = response.data;

      if (status.status === 'completed') {
        setBacktestRunning(false);
        const resultResponse = await getBacktestResult();
        setResult(resultResponse.data);
      } else if (status.status === 'idle') {
        setBacktestRunning(false);
      }
    } catch (err) {
      console.error('Failed to check backtest status:', err);
    }
  };

  const handleStartBacktest = async () => {
    if (!selectedModel) {
      setError('Please select a model checkpoint');
      return;
    }

    try {
      setError(null);
      setResult(null);
      await startBacktest({
        model_path: selectedModel,
        ...params,
      });
      setBacktestRunning(true);
    } catch (err) {
      setError('Failed to start backtest: ' + err.message);
    }
  };

  const handleParamChange = (name, value) => {
    setParams((prev) => ({ ...prev, [name]: value }));
  };

  const formatNumber = (num, decimals = 2) => {
    return num?.toFixed(decimals) || '0.00';
  };

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Backtest Configuration
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Model Checkpoint</InputLabel>
              <Select
                value={selectedModel}
                label="Model Checkpoint"
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {models.map((model) => (
                  <MenuItem key={model.path} value={model.path}>
                    {model.name} ({formatNumber(model.size_mb)} MB)
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', height: '100%' }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={backtestRunning ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                onClick={handleStartBacktest}
                disabled={backtestRunning || !selectedModel}
                fullWidth
              >
                {backtestRunning ? 'Running...' : 'Run Backtest'}
              </Button>
            </Box>
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography gutterBottom>Confidence Threshold: {params.confidence_threshold}</Typography>
            <Slider
              value={params.confidence_threshold}
              min={0.0}
              max={1.0}
              step={0.05}
              onChange={(e, value) => handleParamChange('confidence_threshold', value)}
              valueLabelDisplay="auto"
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography gutterBottom>Take Profit (bps): {params.take_profit_bps}</Typography>
            <Slider
              value={params.take_profit_bps}
              min={1.0}
              max={50.0}
              step={0.5}
              onChange={(e, value) => handleParamChange('take_profit_bps', value)}
              valueLabelDisplay="auto"
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography gutterBottom>Stop Loss (bps): {params.stop_loss_bps}</Typography>
            <Slider
              value={params.stop_loss_bps}
              min={1.0}
              max={50.0}
              step={1.0}
              onChange={(e, value) => handleParamChange('stop_loss_bps', value)}
              valueLabelDisplay="auto"
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Max Hold Periods"
              type="number"
              value={params.max_hold_periods}
              onChange={(e) => handleParamChange('max_hold_periods', parseInt(e.target.value))}
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Commission Rate"
              type="number"
              value={params.commission_rate}
              onChange={(e) => handleParamChange('commission_rate', parseFloat(e.target.value))}
              inputProps={{ step: 0.0001 }}
            />
          </Grid>
        </Grid>
      </Paper>

      {result && (
        <>
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Total Trades
                  </Typography>
                  <Typography variant="h4">{result.total_trades}</Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Win Rate
                  </Typography>
                  <Typography variant="h4" color={result.win_rate > 0.5 ? 'success.main' : 'error.main'}>
                    {formatNumber(result.win_rate * 100, 1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Total PnL (bps)
                  </Typography>
                  <Typography variant="h4" color={result.total_pnl_bps > 0 ? 'success.main' : 'error.main'}>
                    {formatNumber(result.total_pnl_bps)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Sharpe Ratio
                  </Typography>
                  <Typography variant="h4">{formatNumber(result.sharpe_ratio)}</Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Max Drawdown
                  </Typography>
                  <Typography variant="h5" color="error.main">
                    {formatNumber(result.max_drawdown_pct)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Profit Factor
                  </Typography>
                  <Typography variant="h5">{formatNumber(result.profit_factor)}</Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Avg Win / Loss
                  </Typography>
                  <Typography variant="h6">
                    {formatNumber(result.avg_win_bps)} / {formatNumber(result.avg_loss_bps)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Avg Hold Periods
                  </Typography>
                  <Typography variant="h5">{formatNumber(result.avg_hold_periods, 1)}</Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Equity Curve
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={result.equity_curve.map((equity, idx) => ({ idx, equity }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="idx" label={{ value: 'Trade #', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Cumulative PnL (bps)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="equity" stroke="#82ca9d" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Paper>

          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Exit Reasons
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={[
                  { reason: 'Take Profit', count: result.tp_exits },
                  { reason: 'Stop Loss', count: result.sl_exits },
                  { reason: 'Timeout', count: result.timeout_exits },
                ]}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="reason" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>

          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Trade Statistics
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell align="right">Value</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>Winning Trades</TableCell>
                    <TableCell align="right">{result.winning_trades}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Losing Trades</TableCell>
                    <TableCell align="right">{result.losing_trades}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Max Consecutive Wins</TableCell>
                    <TableCell align="right">{result.max_consecutive_wins}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Max Consecutive Losses</TableCell>
                    <TableCell align="right">{result.max_consecutive_losses}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Sortino Ratio</TableCell>
                    <TableCell align="right">{formatNumber(result.sortino_ratio)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Calmar Ratio</TableCell>
                    <TableCell align="right">{formatNumber(result.calmar_ratio)}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </>
      )}
    </Box>
  );
}

export default BacktestPanel;
