import { useCallback, useEffect, useState } from 'react';
import {
  getDatasetChangeHistory,
  getDatasetVersionHistory,
  getDatasetVersionStats,
  syncDatasetVersions,
} from '../services/api.js';

const number = (value) => Number(value ?? 0).toLocaleString('en-IN');
const date = (value) => {
  if (!value) return '—';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString('en-IN', {
        day: '2-digit', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
};

export default function DatasetVersions() {
  const [stats, setStats] = useState(null);
  const [changes, setChanges] = useState([]);
  const [versions, setVersions] = useState([]);
  const [syncing, setSyncing] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [lastChecked, setLastChecked] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [nextStats, nextChanges, nextVersions] = await Promise.all([
        getDatasetVersionStats(),
        getDatasetChangeHistory(),
        getDatasetVersionHistory(),
      ]);
      setStats(nextStats);
      setChanges(Array.isArray(nextChanges) ? nextChanges : []);
      setVersions(nextVersions?.versions ?? []);
      setError('');
      setLastChecked(new Date());
    } catch (err) {
      setError(err.message || 'Could not load dataset version data.');
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const sync = async () => {
    setSyncing(true);
    setNotice('');
    setError('');
    try {
      const result = await syncDatasetVersions();
      setNotice(result.message || 'Sync complete.');
      await refresh();
    } catch (err) {
      setError(err.message || 'Sync failed.');
    } finally {
      setSyncing(false);
    }
  };

  const healthy = String(stats?.validation_status || '').toLowerCase() === 'healthy';

  return (
    <main className="page">
      <header className="hero">
        <div>
          <div className="eyebrow">AirSense AI / MLOps</div>
          <h1>Dataset Versions</h1>
          <p>Sync DVC snapshots and review dataset changes over time.</p>
        </div>
        <div className="actions">
          <span className="live"><i /> Live monitoring · 5 sec</span>
          <button onClick={sync} disabled={syncing}>
            {syncing ? 'Syncing…' : '↻ Sync to GitHub + DVC'}
          </button>
        </div>
      </header>

      {error && <div className="notice error">{error}</div>}
      {notice && <div className="notice">{notice}</div>}

      <section className="stats">
        <Stat label="Dataset records" value={number(stats?.rows)} foot="Rows in data/test_dataset.csv" />
        <Stat label="Feature columns" value={number(stats?.columns)} foot="Available data fields" />
        <Stat label="Duplicate pairs" value={number(stats?.duplicate_station_timestamp_pairs)} foot="Station and timestamp matches" />
        <Stat label="Change events" value={number(changes.length)} foot="Recorded dataset updates" />
      </section>

      <section className="panel">
        <div className="panel-head">
          <div><h2>Dataset health</h2><p>Current source and data-quality checks</p></div>
          <span className={`pill ${healthy ? 'healthy' : ''}`}>{stats?.validation_status || 'Loading'}</span>
        </div>
        <div className="details">
          <div className="detail"><small>DATASET</small><b>{stats?.dataset || 'test_dataset.csv'}</b><small>SOURCE PATH</small><code>{stats?.path || 'data/test_dataset.csv'}</code></div>
          <div className="detail"><small>TRACKING</small><b>DVC snapshots + Git commits</b><small>AUTO REFRESH</small><b>Every 5 seconds</b></div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head"><div><h2>Dataset change history</h2><p>Added, removed, and modified records in test_dataset.csv</p></div><span className="pill">{number(changes.length)} events</span></div>
        {changes.length === 0 ? <Empty>No dataset changes recorded yet.</Empty> : (
          <div className="table-wrap"><table><thead><tr><th>Detected at</th><th>Rows before → after</th><th>Added</th><th>Removed</th><th>Modified</th><th>Changed columns</th></tr></thead>
            <tbody>{changes.map((item, index) => <tr key={`${item.detected_at}-${index}`}>
              <td>{date(item.detected_at)}</td><td><b>{number(item.old_rows)} → {number(item.new_rows)}</b></td>
              <td>{number(item.added_rows)}</td><td>{number(item.removed_rows)}</td><td>{number(item.changed_records)}</td>
              <td>{[...(item.added_columns || []), ...(item.removed_columns || []), ...(item.changed_columns || [])].join(', ') || '—'}</td>
            </tr>)}</tbody></table></div>
        )}
      </section>

      <section className="panel">
        <div className="panel-head"><div><h2>DVC version history</h2><p>Git commits that updated the DVC pointer files</p></div><span className="pill">{number(versions.length)} versions</span></div>
        {versions.length === 0 ? <Empty>No DVC snapshots yet. Click Sync to create one.</Empty> : (
          <div className="table-wrap"><table><thead><tr><th>Dataset</th><th>Commit</th><th>Date</th><th>Commit message</th></tr></thead>
            <tbody>{versions.map((item, index) => <tr key={`${item.commit}-${index}`}>
              <td><b>{item.dataset || item.pointer_file}</b></td><td><code>{item.short_commit || item.commit}</code></td>
              <td>{date(item.date)}</td><td>{item.message}</td>
            </tr>)}</tbody></table></div>
        )}
      </section>

      <footer>AirSense AI · Dataset observability <span>Last checked: {lastChecked?.toLocaleTimeString() || '—'}</span></footer>
    </main>
  );
}

function Stat({ label, value, foot }) {
  return <article className="stat"><span>{label}</span><strong>{value}</strong><small>{foot}</small></article>;
}

function Empty({ children }) {
  return <div className="empty">{children}</div>;
}
