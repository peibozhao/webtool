
export function backendServer() {
  const protocol = window.location.protocol;
  const hostname = import.meta.env.VITE_BACKEND_HOSTNAME || window.location.hostname
  const port = import.meta.env.VITE_BACKEND_PORT || '';
  return `${protocol}//${hostname}:${port}`;
}

