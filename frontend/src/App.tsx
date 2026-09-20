import { useCallback, useEffect, useState } from 'react'

type Health = { status: string; environment: string; version: string }
type Upstream = {
  configured: boolean
  reachable: boolean
  authenticated?: boolean
  url: string
  reason?: string
}
type Status = { status: string; jev: Upstream; minuspod: Upstream }
type ApiState = { health: Health | null; status: Status | null; error: string | null }

const initialState: ApiState = { health: null, status: null, error: null }

function displayState(upstream: Upstream) {
  if (!upstream.configured) return 'Not configured'
  return upstream.reachable ? 'Reachable' : 'Unavailable'
}

function App() {
  const [state, setState] = useState<ApiState>(initialState)
  const [loading, setLoading] = useState(true)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [healthResponse, statusResponse] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/status'),
      ])
      if (!healthResponse.ok || !statusResponse.ok) {
        throw new Error(`API request failed (${healthResponse.status}/${statusResponse.status})`)
      }
      const [health, status] = await Promise.all([
        healthResponse.json() as Promise<Health>,
        statusResponse.json() as Promise<Status>,
      ])
      setState({ health, status, error: null })
      setUpdatedAt(new Date())
    } catch (error) {
      setState({
        health: null,
        status: null,
        error: error instanceof Error ? error.message : 'Unable to reach the proxy',
      })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const serviceHealthy = state.health?.status === 'healthy'
  const statusOk = state.status?.status === 'ok'
  const checking = loading && !state.health && !state.status

  return (
    <main className="page">
      <section className="panel" aria-labelledby="page-title">
        <header className="header">
          <div>
            <p className="eyebrow">MinusPod integration</p>
            <h1 id="page-title">Jev Proxy</h1>
            <p className="subtitle">Live service and upstream connectivity status.</p>
          </div>
          <button type="button" onClick={() => void refresh()} disabled={loading}>
            {loading ? 'Checking...' : 'Refresh status'}
          </button>
        </header>

        {state.error ? (
          <div className="notice error" role="alert">
            <strong>Proxy unavailable.</strong> {state.error}
          </div>
        ) : (
          <div className="summary" aria-live="polite">
            <span className={`dot ${serviceHealthy && statusOk ? 'good' : 'warn'}`} />
            <span>
              {checking ? 'Checking status...' : serviceHealthy && statusOk ? 'Service is healthy' : 'Service needs attention'}
            </span>
            {updatedAt && (
              <time dateTime={updatedAt.toISOString()}>Updated {updatedAt.toLocaleTimeString()}</time>
            )}
          </div>
        )}

        <div className="cards">
          <article className="card">
            <h2>Proxy</h2>
            <dl>
              <div><dt>Health</dt><dd className={serviceHealthy ? 'good-text' : 'warn-text'}>{state.health?.status ?? 'Unknown'}</dd></div>
              <div><dt>Environment</dt><dd>{state.health?.environment ?? 'Unknown'}</dd></div>
              <div><dt>Version</dt><dd>{state.health?.version ?? 'Unknown'}</dd></div>
            </dl>
          </article>
          <UpstreamCard name="TypeSafe Jev" upstream={state.status?.jev} />
          <UpstreamCard name="MinusPod" upstream={state.status?.minuspod} />
        </div>

        <footer>
          Polls only when you refresh. Status probes do not make Jev inference calls or log in to MinusPod.
        </footer>
      </section>
    </main>
  )
}

function UpstreamCard({ name, upstream }: { name: string; upstream: Upstream | undefined }) {
  const available = upstream?.configured && upstream.reachable
  return (
    <article className="card">
      <h2>{name}</h2>
      <dl>
        <div><dt>Connection</dt><dd className={available ? 'good-text' : 'warn-text'}>{upstream ? displayState(upstream) : 'Unknown'}</dd></div>
        <div><dt>Host</dt><dd>{upstream?.url || 'Not set'}</dd></div>
        {name === 'MinusPod' && <div><dt>Session</dt><dd>{upstream?.authenticated ? 'Active' : 'Inactive'}</dd></div>}
        {upstream?.reason && <div><dt>Reason</dt><dd>{upstream.reason}</dd></div>}
      </dl>
    </article>
  )
}

export default App
