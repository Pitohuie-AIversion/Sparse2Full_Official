import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'

// JS Error Tracking (Mock for custom logging)
window.addEventListener('error', (event) => {
  console.error('[Tracking] Caught Exception:', event.message, event.filename, event.lineno);
  // fetch('https://api.yourdomain.com/logs', { method: 'POST', body: JSON.stringify({ error: event.message }) });
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('[Tracking] Unhandled Promise Rejection:', event.reason);
});

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)