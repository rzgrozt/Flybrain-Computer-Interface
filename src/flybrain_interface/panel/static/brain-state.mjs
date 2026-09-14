export function activityIsStale(lastTelemetryAtMs, nowMs, thresholdMs=2000){
  return lastTelemetryAtMs>0&&nowMs-lastTelemetryAtMs>thresholdMs;
}

export function shouldClearActivity(status){
  return ['idle','offline','failed','stopped','disconnected'].includes(status);
}
