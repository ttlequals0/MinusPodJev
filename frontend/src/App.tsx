import { useCallback, useEffect, useRef, useState } from 'react'

type Health = { status: string; environment: string; version: string }
type Upstream = { configured: boolean; reachable: boolean; authenticated?: boolean; url: string; reason?: string }
type Status = { status: string; jev: Upstream; minuspod: Upstream }
type Latency = { count: number; sum: number; min: number; max: number; average: number }
type Stats = {
  scope: { kind: 'process'; pid: number; configured_workers: number | null }
  reset_on_restart: boolean
  uptime_seconds: number
  proxy_requests: { count: number; success: number; failure: number; latency_ms: Latency }
  jev_http: { attempts: number; success: number; failure: number; unknown_usage: number; latency_ms: Latency }
  cache: { hits: number; misses: number }
  cost: { estimated_input_usd: number }
}
type ApiState = { health: Health | null; status: Status | null; stats: Stats | null; error: string | null; statsError: string | null }

const initialState: ApiState = { health: null, status: null, stats: null, error: null, statsError: null }

function displayState(upstream: Upstream) {
  if (!upstream.configured) return 'Not configured'
  return upstream.reachable ? 'Reachable' : 'Unavailable'
}

function formatDuration(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds))
  const days = Math.floor(whole / 86400)
  const hours = Math.floor((whole % 86400) / 3600)
  const minutes = Math.floor((whole % 3600) / 60)
  if (days) return `${days}d ${hours}h`
  if (hours) return `${hours}h ${minutes}m`
  return whole < 60 ? `${whole}s` : `${minutes}m`
}

function formatLatency(value: number, count: number) {
  return count ? `${Math.round(value).toLocaleString()} ms` : 'No samples'
}

function formatUsd(value: number) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 6, maximumFractionDigits: 6 }).format(value)
}

function App() {
  const [state, setState] = useState<ApiState>(initialState)
  const [loading, setLoading] = useState(true)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const [statsUpdatedAt, setStatsUpdatedAt] = useState<Date | null>(null)
  const statsRefreshInFlight = useRef(false)

  const refreshStats = useCallback(async () => {
    if (statsRefreshInFlight.current) return
    statsRefreshInFlight.current = true
    try {
      const response = await fetch('/api/stats')
      if (!response.ok) throw new Error(`Runtime stats request failed (${response.status})`)
      const stats = await (response.json() as Promise<Stats>)
      setState((current) => ({ ...current, stats, statsError: null }))
      setStatsUpdatedAt(new Date())
    } catch (error) {
      setState((current) => ({
        ...current,
        statsError: error instanceof Error ? error.message : 'Runtime stats request failed',
      }))
    }
    finally {
      statsRefreshInFlight.current = false
    }
  }, [])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [healthResponse, statusResponse] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/status'),
      ])
      if (!healthResponse.ok || !statusResponse.ok) {
        throw new Error(`Service status request failed (${healthResponse.status}/${statusResponse.status})`)
      }
      const [health, status] = await Promise.all([
        healthResponse.json() as Promise<Health>,
        statusResponse.json() as Promise<Status>,
      ])
      setState((current) => ({ ...current, health, status, error: null }))
      setUpdatedAt(new Date())
      await refreshStats()
    } catch (error) {
      setState((current) => ({
        ...current,
        error: error instanceof Error ? error.message : 'Unable to reach the proxy',
      }))
    } finally {
      setLoading(false)
    }
  }, [refreshStats])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => { void refreshStats() }, 5000)
    return () => window.clearInterval(timer)
  }, [refresh, refreshStats])

  const serviceHealthy = state.health?.status === 'healthy'
  const statusOk = state.status?.status === 'ok'
  const checking = loading && !state.health && !state.status

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand"><strong>MinusPod</strong><span>Jev Proxy</span></div>
          <nav aria-label="Page sections"><a href="#overview">Overview</a><a href="#runtime">Runtime stats</a></nav>
          <button className="refresh" type="button" onClick={() => void refresh()} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </header>
      <main>
        <section className="page-header" id="overview" aria-labelledby="page-title">
          <div><h1 id="page-title">Jev Proxy</h1><p>Operational status and process-local runtime metrics.</p></div>
          {updatedAt && <time dateTime={updatedAt.toISOString()}>Updated {updatedAt.toLocaleTimeString()}</time>}
        </section>

        {state.error ? (
          <div className="notice error" role="alert"><strong>Proxy unavailable.</strong> {state.error}</div>
        ) : (
          <div className="service-line" aria-live="polite">
            <span className={`status-dot ${serviceHealthy && statusOk ? 'healthy' : 'degraded'}`} />
            <span>{checking ? 'Checking service status...' : serviceHealthy && statusOk ? 'Service healthy' : 'Service needs attention'}</span>
          </div>
        )}

        <section className="section" aria-labelledby="connections-title">
          <h2 id="connections-title">Connections</h2>
          <div className="connection-grid">
            <article className="card"><h3>Proxy</h3><dl><Definition label="Health" value={state.health?.status ?? 'Unknown'} tone={serviceHealthy ? 'success' : 'warning'} /><Definition label="Environment" value={state.health?.environment ?? 'Unknown'} /><Definition label="Version" value={state.health?.version ?? 'Unknown'} /></dl></article>
            <UpstreamCard name="TypeSafe Jev" upstream={state.status?.jev} />
            <UpstreamCard name="MinusPod" upstream={state.status?.minuspod} />
          </div>
        </section>

        <section className="section" id="runtime" aria-labelledby="runtime-title">
          <div className="section-heading"><div><h2 id="runtime-title">Runtime stats</h2><p>Current proxy process only. Counters reset when the process restarts.</p></div><span className="badge">Process scoped</span></div>
          {statsUpdatedAt && <p className="stats-updated">Stats updated {statsUpdatedAt.toLocaleTimeString()}</p>}
          {state.statsError && <div className="notice error" role="status">{state.stats ? `Stats refresh failed. Showing last received values: ${state.statsError}` : `Runtime stats unavailable: ${state.statsError}`}</div>}
          <RuntimeStats stats={state.stats} />
        </section>
      </main>
    </div>
  )
}

function Definition({ label, value, tone }: { label: string; value: string; tone?: 'success' | 'warning' }) {
  return <div className="definition"><dt>{label}</dt><dd className={tone ? `tone-${tone}` : undefined}>{value}</dd></div>
}

function UpstreamCard({ name, upstream }: { name: string; upstream: Upstream | undefined }) {
  const available = upstream?.configured && upstream.reachable
  return <article className="card"><h3>{name}</h3><dl>
    <Definition label="Connection" value={upstream ? displayState(upstream) : 'Unknown'} tone={available ? 'success' : 'warning'} />
    <Definition label="Host" value={upstream?.url || 'Not set'} />
    {name === 'MinusPod' && <Definition label="Session" value={upstream?.authenticated ? 'Active' : 'Inactive'} />}
    {upstream?.reason && <Definition label="Reason" value={upstream.reason} />}
  </dl></article>
}

function RuntimeStats({ stats }: { stats: Stats | null }) {
  if (!stats) return <div className="notice" role="status">Loading runtime statistics...</div>
  const cacheTotal = stats.cache.hits + stats.cache.misses
  const cacheRate = cacheTotal ? `${Math.round((stats.cache.hits / cacheTotal) * 100)}%` : 'No samples'
  return <>
    <div className="metric-grid">
      <Metric label="Proxy calls" value={stats.proxy_requests.count.toLocaleString()} detail={`${stats.proxy_requests.success} successful, ${stats.proxy_requests.failure} failed`} />
      <Metric label="Average proxy handling time" value={formatLatency(stats.proxy_requests.latency_ms.average, stats.proxy_requests.latency_ms.count)} detail="Request receipt to response headers; not full MinusPod RTT" />
      <Metric label="Jev HTTP attempts" value={stats.jev_http.attempts.toLocaleString()} detail={`${stats.jev_http.success} successful, ${stats.jev_http.failure} failed`} />
      <Metric label="Average Jev round-trip" value={formatLatency(stats.jev_http.latency_ms.average, stats.jev_http.latency_ms.count)} detail="Upstream POST attempts, including retries" />
      <Metric label="Cache hit rate" value={cacheRate} detail={`${stats.cache.hits} hits, ${stats.cache.misses} misses`} />
      <Metric label="Estimated input cost" value={formatUsd(stats.cost.estimated_input_usd)} detail={`Uncached successful calls; ${stats.jev_http.unknown_usage} unknown usage`} />
    </div>
    <div className="runtime-details">
      <div><span>Process uptime</span><strong>{formatDuration(stats.uptime_seconds)}</strong></div>
      <div><span>Configured workers</span><strong>{stats.scope.configured_workers ?? 'Unknown'}</strong></div>
      <p>Stats are local to process {stats.scope.pid}. Configured workers is an optional environment hint, not a discovered worker count. With multiple workers, requests may alternate between per-worker counters rather than forming a container-wide total. Full MinusPod round-trip timing requires client timing. Cache hits do not call Jev; failed cache fetches count as misses. Failed or usage-unknown attempts do not add to the cost estimate.</p>
    </div>
  </>
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <article className="metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>
}

export default App
