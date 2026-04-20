import React, { useState, useEffect } from 'react';
import { Chip, Tooltip } from '@mui/material';
import MemoryIcon from '@mui/icons-material/Memory';
import { getSystemInfo } from '../api';

function SystemInfo() {
  const [info, setInfo] = useState(null);

  useEffect(() => {
    loadSystemInfo();
    const interval = setInterval(loadSystemInfo, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadSystemInfo = async () => {
    try {
      const response = await getSystemInfo();
      setInfo(response.data);
    } catch (error) {
      console.error('Failed to load system info:', error);
    }
  };

  if (!info) return null;

  const label = info.cuda_available
    ? `${info.gpu_name} | ${info.memory_allocated}`
    : 'CPU';

  return (
    <Tooltip title={info.cuda_available ? `Reserved: ${info.memory_reserved}` : 'No GPU available'}>
      <Chip
        icon={<MemoryIcon />}
        label={label}
        color={info.cuda_available ? 'success' : 'default'}
        variant="outlined"
      />
    </Tooltip>
  );
}

export default SystemInfo;
