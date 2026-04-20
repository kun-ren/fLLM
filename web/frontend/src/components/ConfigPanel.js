import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Snackbar,
  Alert,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SaveIcon from '@mui/icons-material/Save';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import { getConfig, updateConfigParam, saveConfig, loadConfig, listPresets } from '../api';

const GROUPS = ['Data', 'Model', 'Optimizer', 'Training', 'Loss', 'Backtest'];

function ConfigPanel() {
  const [config, setConfig] = useState({});
  const [presets, setPresets] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState('');
  const [savePresetName, setSavePresetName] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  useEffect(() => {
    loadConfigData();
    loadPresets();
  }, []);

  const loadConfigData = async () => {
    try {
      const response = await getConfig();
      setConfig(response.data);
    } catch (error) {
      showSnackbar('Failed to load configuration', 'error');
    }
  };

  const loadPresets = async () => {
    try {
      const response = await listPresets();
      setPresets(response.data.presets);
    } catch (error) {
      showSnackbar('Failed to load presets', 'error');
    }
  };

  const handleParamChange = async (name, value) => {
    try {
      await updateConfigParam(name, { value, mode: 'single' });
      setConfig((prev) => ({
        ...prev,
        [name]: { ...prev[name], value },
      }));
    } catch (error) {
      showSnackbar(`Failed to update ${name}`, 'error');
    }
  };

  const handleSaveConfig = async () => {
    if (!savePresetName) {
      showSnackbar('Please enter a preset name', 'warning');
      return;
    }
    try {
      await saveConfig(savePresetName);
      showSnackbar(`Configuration saved as "${savePresetName}"`, 'success');
      setSavePresetName('');
      loadPresets();
    } catch (error) {
      showSnackbar('Failed to save configuration', 'error');
    }
  };

  const handleLoadConfig = async () => {
    if (!selectedPreset) {
      showSnackbar('Please select a preset', 'warning');
      return;
    }
    try {
      const response = await loadConfig(selectedPreset);
      setConfig(response.data.config);
      showSnackbar(`Configuration "${selectedPreset}" loaded`, 'success');
    } catch (error) {
      showSnackbar('Failed to load configuration', 'error');
    }
  };

  const showSnackbar = (message, severity) => {
    setSnackbar({ open: true, message, severity });
  };

  const renderParamControl = (name, param) => {
    if (typeof param.value === 'boolean') {
      return (
        <FormControlLabel
          control={
            <Switch
              checked={param.value}
              onChange={(e) => handleParamChange(name, e.target.checked)}
            />
          }
          label={param.description}
        />
      );
    }

    if (param.slider && param.min_val !== null && param.max_val !== null) {
      return (
        <Box>
          <Typography gutterBottom>{param.description}</Typography>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs>
              <Slider
                value={param.value}
                min={param.min_val}
                max={param.max_val}
                step={param.step || 1}
                onChange={(e, value) => handleParamChange(name, value)}
                valueLabelDisplay="auto"
              />
            </Grid>
            <Grid item>
              <TextField
                value={param.value}
                size="small"
                onChange={(e) => handleParamChange(name, parseFloat(e.target.value) || 0)}
                sx={{ width: 100 }}
                type="number"
              />
            </Grid>
          </Grid>
        </Box>
      );
    }

    if (typeof param.value === 'string' && param.value.includes(',')) {
      return (
        <TextField
          fullWidth
          label={param.description}
          value={param.value}
          onChange={(e) => handleParamChange(name, e.target.value)}
          size="small"
        />
      );
    }

    return (
      <TextField
        fullWidth
        label={param.description}
        value={param.value}
        onChange={(e) => {
          const val = typeof param.value === 'number'
            ? parseFloat(e.target.value) || 0
            : e.target.value;
          handleParamChange(name, val);
        }}
        size="small"
        type={typeof param.value === 'number' ? 'number' : 'text'}
      />
    );
  };

  const groupedParams = GROUPS.reduce((acc, group) => {
    acc[group] = Object.entries(config).filter(([_, param]) => param.group === group);
    return acc;
  }, {});

  return (
    <Box>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Configuration Presets
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Load Preset</InputLabel>
              <Select
                value={selectedPreset}
                label="Load Preset"
                onChange={(e) => setSelectedPreset(e.target.value)}
              >
                {presets.map((preset) => (
                  <MenuItem key={preset} value={preset}>
                    {preset}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<FolderOpenIcon />}
              onClick={handleLoadConfig}
            >
              Load
            </Button>
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              size="small"
              label="Save As"
              value={savePresetName}
              onChange={(e) => setSavePresetName(e.target.value)}
              placeholder="Enter preset name"
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={handleSaveConfig}
            >
              Save
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {GROUPS.map((group) => (
        <Accordion key={group} defaultExpanded={group === 'Model'}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">{group} Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              {groupedParams[group]?.map(([name, param]) => (
                <Grid item xs={12} md={6} key={name}>
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      {name}
                    </Typography>
                    {renderParamControl(name, param)}
                  </Box>
                </Grid>
              ))}
            </Grid>
          </AccordionDetails>
        </Accordion>
      ))}

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default ConfigPanel;
