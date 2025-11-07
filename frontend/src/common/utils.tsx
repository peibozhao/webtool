
export function backendServer() {
  const protocol = window.location.protocol;
  const hostname = import.meta.env.VITE_BACKEND_HOSTNAME || window.location.hostname
  const port = import.meta.env.VITE_BACKEND_PORT || window.location.port;
  return `${protocol}//${hostname}:${port}`;
}

export async function processFetchResponse(response: Response, notifyApi: any, actionName: string) {
  if (!response.ok) {
    const text = await response.text();
    notifyApi.error({ message: `${actionName}失败`, description: text, duration: 3 });
    console.error(response);
    return false;
  }
  notifyApi.success({ message: `${actionName}成功`, duration: 1 });
  return true;
}
