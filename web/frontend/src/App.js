import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Container, AppBar, Toolbar, Typography, Tabs, Tab } from '@mui/material';
import ConfigPanel from './components/ConfigPanel';
import TrainingDashboard from './components/TrainingDashboard';
import BacktestPanel from './components/BacktestPanel';
import TrainHistoryPanel from './components/TrainHistoryPanel';
import InferencePanel from './components/InferencePanel';
import SystemInfo from './components/SystemInfo';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
  },
});

function App() {
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              fLLM Dashboard
            </Typography>
            <SystemInfo />
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
          <Tabs value={currentTab} onChange={handleTabChange} sx={{ mb: 3 }}>
            <Tab label="Training" />
            <Tab label="Backtest" />
            <Tab label="Inference" />
            <Tab label="History" />
            <Tab label="Configuration" />
          </Tabs>

          {currentTab === 0 && <TrainingDashboard />}
          {currentTab === 1 && <BacktestPanel />}
          {currentTab === 2 && <InferencePanel />}
          {currentTab === 3 && <TrainHistoryPanel />}
          {currentTab === 4 && <ConfigPanel />}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
