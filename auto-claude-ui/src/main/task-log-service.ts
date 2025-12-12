import path from 'path';
import { existsSync, readFileSync, watchFile, unwatchFile, FSWatcher } from 'fs';
import { EventEmitter } from 'events';
import type { TaskLogs, TaskLogPhase, TaskLogStreamChunk, TaskPhaseLog } from '../shared/types';

/**
 * Service for loading and watching phase-based task logs (task_logs.json)
 *
 * This service provides:
 * - Loading logs from the spec directory
 * - Watching for log file changes
 * - Emitting streaming updates when logs change
 * - Determining which phase is currently active
 */
export class TaskLogService extends EventEmitter {
  private watchers: Map<string, { watcher: ReturnType<typeof watchFile>; specDir: string }> = new Map();
  private logCache: Map<string, TaskLogs> = new Map();
  private pollIntervals: Map<string, NodeJS.Timeout> = new Map();

  // Poll interval for watching log changes (more reliable than fs.watch on some systems)
  private readonly POLL_INTERVAL_MS = 1000;

  constructor() {
    super();
  }

  /**
   * Load task logs from a spec directory
   */
  loadLogs(specDir: string): TaskLogs | null {
    const logFile = path.join(specDir, 'task_logs.json');

    if (!existsSync(logFile)) {
      return null;
    }

    try {
      const content = readFileSync(logFile, 'utf-8');
      const logs = JSON.parse(content) as TaskLogs;
      this.logCache.set(specDir, logs);
      return logs;
    } catch (error) {
      console.error(`[TaskLogService] Failed to load logs from ${logFile}:`, error);
      return null;
    }
  }

  /**
   * Get the currently active phase from logs
   */
  getActivePhase(specDir: string): TaskLogPhase | null {
    const logs = this.loadLogs(specDir);
    if (!logs) return null;

    const phases: TaskLogPhase[] = ['planning', 'coding', 'validation'];
    for (const phase of phases) {
      if (logs.phases[phase]?.status === 'active') {
        return phase;
      }
    }
    return null;
  }

  /**
   * Get logs for a specific phase
   */
  getPhaseLog(specDir: string, phase: TaskLogPhase): TaskPhaseLog | null {
    const logs = this.loadLogs(specDir);
    if (!logs) return null;
    return logs.phases[phase] || null;
  }

  /**
   * Start watching a spec directory for log changes
   */
  startWatching(specId: string, specDir: string): void {
    // Stop any existing watch
    this.stopWatching(specId);

    const logFile = path.join(specDir, 'task_logs.json');
    let lastContent = '';

    // Initial load
    if (existsSync(logFile)) {
      try {
        lastContent = readFileSync(logFile, 'utf-8');
        this.logCache.set(specDir, JSON.parse(lastContent));
      } catch (e) {
        // Ignore parse errors on initial load
      }
    }

    // Poll for changes (more reliable than fs.watch across platforms)
    const pollInterval = setInterval(() => {
      if (!existsSync(logFile)) {
        return;
      }

      try {
        const currentContent = readFileSync(logFile, 'utf-8');

        if (currentContent !== lastContent) {
          lastContent = currentContent;
          const logs = JSON.parse(currentContent) as TaskLogs;
          const previousLogs = this.logCache.get(specDir);
          this.logCache.set(specDir, logs);

          // Emit change event with the new logs
          this.emit('logs-changed', specId, logs);

          // Calculate and emit streaming chunks for new entries
          this.emitNewEntries(specId, previousLogs, logs);
        }
      } catch (error) {
        // Ignore read/parse errors (file might be mid-write)
      }
    }, this.POLL_INTERVAL_MS);

    this.pollIntervals.set(specId, pollInterval);
    console.log(`[TaskLogService] Started watching ${specId}`);
  }

  /**
   * Stop watching a spec directory
   */
  stopWatching(specId: string): void {
    const interval = this.pollIntervals.get(specId);
    if (interval) {
      clearInterval(interval);
      this.pollIntervals.delete(specId);
      console.log(`[TaskLogService] Stopped watching ${specId}`);
    }
  }

  /**
   * Stop all watches
   */
  stopAllWatching(): void {
    for (const specId of this.pollIntervals.keys()) {
      this.stopWatching(specId);
    }
  }

  /**
   * Emit streaming chunks for new log entries
   */
  private emitNewEntries(specId: string, previousLogs: TaskLogs | undefined, currentLogs: TaskLogs): void {
    const phases: TaskLogPhase[] = ['planning', 'coding', 'validation'];

    for (const phase of phases) {
      const prevPhase = previousLogs?.phases[phase];
      const currPhase = currentLogs.phases[phase];

      if (!currPhase) continue;

      // Check for phase status changes
      if (prevPhase?.status !== currPhase.status) {
        if (currPhase.status === 'active') {
          this.emit('stream-chunk', specId, {
            type: 'phase_start',
            phase,
            timestamp: currPhase.started_at || new Date().toISOString()
          } as TaskLogStreamChunk);
        } else if (currPhase.status === 'completed' || currPhase.status === 'failed') {
          this.emit('stream-chunk', specId, {
            type: 'phase_end',
            phase,
            timestamp: currPhase.completed_at || new Date().toISOString()
          } as TaskLogStreamChunk);
        }
      }

      // Check for new entries
      const prevEntryCount = prevPhase?.entries.length || 0;
      const currEntryCount = currPhase.entries.length;

      if (currEntryCount > prevEntryCount) {
        // Emit new entries
        for (let i = prevEntryCount; i < currEntryCount; i++) {
          const entry = currPhase.entries[i];

          const chunk: TaskLogStreamChunk = {
            type: entry.type as TaskLogStreamChunk['type'],
            content: entry.content,
            phase: entry.phase,
            timestamp: entry.timestamp,
            chunk_id: entry.chunk_id
          };

          if (entry.tool_name) {
            chunk.tool = {
              name: entry.tool_name,
              input: entry.tool_input
            };
          }

          this.emit('stream-chunk', specId, chunk);
        }
      }
    }
  }

  /**
   * Get cached logs without re-reading from disk
   */
  getCachedLogs(specDir: string): TaskLogs | null {
    return this.logCache.get(specDir) || null;
  }

  /**
   * Clear the log cache for a spec
   */
  clearCache(specDir: string): void {
    this.logCache.delete(specDir);
  }

  /**
   * Check if logs exist for a spec
   */
  hasLogs(specDir: string): boolean {
    const logFile = path.join(specDir, 'task_logs.json');
    return existsSync(logFile);
  }
}

// Singleton instance
export const taskLogService = new TaskLogService();
