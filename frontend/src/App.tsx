import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'

type Health = { status: string; environment: string; version: string }
type Upstream = { configured: boolean; reachable: boolean; authenticated?: boolean; url: string; reason?: string }
type ReviewSettings = {
  refine_boundaries?: boolean
  model?: string
  evidence_threshold?: number
  choice_threshold?: number
}
type Status = { status: string; jev: Upstream; minuspod: Upstream; review?: ReviewSettings }
type Thresholds = {
  detection_enter: number
  detection_stay: number
  review_evidence: number
  review_choice: number
}
type RuntimeSettings = {
  thresholds: Thresholds
  defaults: Thresholds
  persisted: boolean
  editable: boolean
}
type SettingsDraft = {
  detection_enter: string
  detection_stay: string
  review_evidence: string
  review_choice: string
}
type Latency = { count: number; sum: number; min: number; max: number; average: number }
type RefinementStats = {
  attempted?: number
  completed?: number
  changed?: number
  unchanged?: number
  inconclusive?: number
  upstream_error?: number
  skipped?: Partial<Record<'disabled' | 'missing_word_timings' | 'insufficient_evidence' | 'ambiguous_spans' | 'no_overlapping_span' | 'no_valid_pairs', number>>
}
type Review = {
  count: number
  outcomes: {
    confirmed: number
    adjusted: number
    rejected: number
    inconclusive: number
    upstream_error: number
    invalid_request: number
    internal_error: number
  }
  reasons: Record<string, number>
  latency_ms: Latency
  refinement?: RefinementStats
}
type Stats = {
  scope: { kind: 'process'; pid: number; configured_workers: number | null }
  reset_on_restart: boolean
  uptime_seconds: number
  proxy_requests: { count: number; success: number; failure: number; latency_ms: Latency }
  jev_http: { attempts: number; success: number; failure: number; unknown_usage: number; latency_ms: Latency }
  review?: Review
  cache: { hits: number; misses: number }
  cost: { estimated_input_usd: number }
}
type ApiState = { health: Health | null; status: Status | null; stats: Stats | null; settings: RuntimeSettings | null; error: string | null; statsError: string | null }

const initialState: ApiState = { health: null, status: null, stats: null, settings: null, error: null, statsError: null }

function settingsToDraft(settings: RuntimeSettings): SettingsDraft {
  return {
    detection_enter: String(settings.thresholds.detection_enter),
    detection_stay: String(settings.thresholds.detection_stay),
    review_evidence: String(settings.thresholds.review_evidence),
    review_choice: String(settings.thresholds.review_choice),
  }
}

function settingsErrorMessage(status: number) {
  if (status === 401 || status === 403) return 'Settings credential rejected.'
  if (status === 422) return 'Settings values are invalid.'
  return `Settings request failed (${status}).`
}

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
  const settingsOperation = useRef(0)
  const settingsSavingRef = useRef(false)
  const settingsDirtyRef = useRef(false)
  const [settingsDraft, setSettingsDraft] = useState<SettingsDraft | null>(null)
  const [settingsDirty, setSettingsDirty] = useState(false)
  const [settingsCredential, setSettingsCredential] = useState('')
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsError, setSettingsError] = useState<string | null>(null)
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null)

  const refreshSettings = useCallback(async () => {
    if (settingsSavingRef.current) return
    const operation = ++settingsOperation.current
    try {
      const response = await fetch('/api/settings')
      if (!response.ok) throw new Error(settingsErrorMessage(response.status))
      const settings = await (response.json() as Promise<RuntimeSettings>)
      if (operation !== settingsOperation.current) return
      setState((current) => ({ ...current, settings }))
      if (!settingsDirtyRef.current) setSettingsDraft(settingsToDraft(settings))
      setSettingsError(null)
    } catch (error) {
      if (operation !== settingsOperation.current) return
      setSettingsError(error instanceof Error ? error.message : 'Settings are unavailable.')
    }
  }, [])

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
    const settingsRefresh = refreshSettings()
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
      await settingsRefresh
    } catch (error) {
      await settingsRefresh
      setState((current) => ({
        ...current,
        error: error instanceof Error ? error.message : 'Unable to reach the proxy',
      }))
    } finally {
      setLoading(false)
    }
  }, [refreshSettings, refreshStats])

  const updateSettingsDraft = useCallback((field: keyof SettingsDraft, value: string) => {
    settingsDirtyRef.current = true
    setSettingsDirty(true)
    setSettingsMessage(null)
    setSettingsError(null)
    setSettingsDraft((current) => current ? { ...current, [field]: value } : current)
  }, [])

  const saveSettings = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (settingsSavingRef.current) return
    const runtimeSettings = state.settings
    const draft = settingsDraft
    const credential = settingsCredential
    setSettingsCredential('')
    setSettingsMessage(null)
    setSettingsError(null)
    if (!runtimeSettings || !draft) return
    if (!credential.trim()) {
      setSettingsError('Enter the MinusPod password to save settings.')
      return
    }
    const values = {
      detection_enter: Number(draft.detection_enter),
      detection_stay: Number(draft.detection_stay),
      review_evidence: Number(draft.review_evidence),
      review_choice: Number(draft.review_choice),
    }
    if (Object.values(draft).some((value) => !value.trim())) {
      setSettingsError('Enter all four thresholds before saving.')
      return
    }
    if (Object.values(values).some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
      setSettingsError('Thresholds must be finite probabilities from 0 to 1.')
      return
    }
    if (values.detection_enter < values.detection_stay) {
      setSettingsError('Detection enter must be at least detection stay.')
      return
    }
    const operation = ++settingsOperation.current
    settingsSavingRef.current = true
    setSettingsSaving(true)
    try {
      const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${credential}` },
        body: JSON.stringify(values),
      })
      if (operation !== settingsOperation.current) return
      if (!response.ok) throw new Error(settingsErrorMessage(response.status))
      const saved = await (response.json() as Promise<RuntimeSettings>)
      if (operation !== settingsOperation.current) return
      setState((current) => ({ ...current, settings: saved }))
      setSettingsDraft(settingsToDraft(saved))
      settingsDirtyRef.current = false
      setSettingsDirty(false)
      setSettingsMessage(saved.persisted ? 'Settings saved across restarts.' : 'Settings use environment defaults.')
    } catch (error) {
      if (operation !== settingsOperation.current) return
      setSettingsError(error instanceof Error ? error.message : 'Settings could not be saved.')
    } finally {
      settingsSavingRef.current = false
      if (operation === settingsOperation.current) setSettingsSaving(false)
    }
  }, [settingsCredential, settingsDraft, state.settings])

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
          <div className="brand"><img className="brand-mark" src="/minuspodjev-logo.png" alt="" /><strong>MinusPod</strong><span>Jev Proxy</span></div>
          <nav aria-label="Page sections"><a href="#overview">Overview</a><a href="#runtime">Runtime stats</a></nav>
          <button className="refresh" type="button" onClick={() => void refresh()} disabled={loading || settingsSaving}>
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
            <ReviewCard
              settings={state.status?.review}
              runtimeSettings={state.settings}
              draft={settingsDraft}
              dirty={settingsDirty}
              saving={settingsSaving}
              error={settingsError}
              message={settingsMessage}
              credential={settingsCredential}
              onCredentialChange={setSettingsCredential}
              onDraftChange={updateSettingsDraft}
              onSave={saveSettings}
            />
          </div>
        </section>

        <section className="section" id="runtime" aria-labelledby="runtime-title">
          <div className="section-heading"><div><h2 id="runtime-title">Runtime stats</h2><p>Current proxy process only. Counters reset when the process restarts.</p></div><span className="badge">Process scoped</span></div>
          {statsUpdatedAt && <p className="stats-updated">Stats updated {statsUpdatedAt.toLocaleTimeString()}</p>}
          {state.statsError && <div className="notice error" role="status">{state.stats ? `Stats refresh failed. Showing last received values: ${state.statsError}` : `Runtime stats unavailable: ${state.statsError}`}</div>}
          <RuntimeStats stats={state.stats} statsError={state.statsError} />
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

function ReviewCard({
  settings,
  runtimeSettings,
  draft,
  dirty,
  saving,
  error,
  message,
  credential,
  onCredentialChange,
  onDraftChange,
  onSave,
}: {
  settings: ReviewSettings | undefined
  runtimeSettings: RuntimeSettings | null
  draft: SettingsDraft | null
  dirty: boolean
  saving: boolean
  error: string | null
  message: string | null
  credential: string
  onCredentialChange: (value: string) => void
  onDraftChange: (field: keyof SettingsDraft, value: string) => void
  onSave: (event: FormEvent<HTMLFormElement>) => void
}) {
  const thresholds = runtimeSettings?.thresholds
  const evidenceThreshold = thresholds?.review_evidence ?? settings?.evidence_threshold
  const choiceThreshold = thresholds?.review_choice ?? settings?.choice_threshold
  return <article className="card review-card"><h3>Jev review</h3>
    {settings || runtimeSettings ? <>
      <dl>
        <Definition label="Boundary refinement" value={settings?.refine_boundaries == null ? 'Not reported' : settings.refine_boundaries ? 'Enabled' : 'Disabled'} tone={settings?.refine_boundaries ? 'success' : undefined} />
        <Definition label="Model" value={settings?.model ?? 'Not reported'} />
        <Definition label="Evidence threshold" value={evidenceThreshold == null ? 'Not reported' : String(evidenceThreshold)} />
        <Definition label="Choice threshold" value={choiceThreshold == null ? 'Not reported' : String(choiceThreshold)} />
      </dl>
      <p className="card-note">Enabled does not mean every review runs refinement. Both word-timing edges and sufficient evidence are required.</p>
      {runtimeSettings && draft && <form className="settings-form" onSubmit={onSave}>
        <h4>Runtime thresholds</h4>
        <p className="card-note">Thresholds are probabilities from 0 to 1. Detection enter must be at least detection stay. Review evidence and Choice are independent. Values apply to new requests; in-flight requests keep existing settings.</p>
        <dl className="settings-saved"><Definition label="Saved detection enter" value={String(runtimeSettings.thresholds.detection_enter)} /><Definition label="Saved detection stay" value={String(runtimeSettings.thresholds.detection_stay)} /><Definition label="Saved review evidence" value={String(runtimeSettings.thresholds.review_evidence)} /><Definition label="Saved review Choice" value={String(runtimeSettings.thresholds.review_choice)} /><Definition label="Persistence" value={runtimeSettings.persisted ? 'Saved override' : 'Environment defaults'} /></dl>
        <p className="card-note">Environment defaults: detection enter {String(runtimeSettings.defaults.detection_enter)}, detection stay {String(runtimeSettings.defaults.detection_stay)}, review evidence {String(runtimeSettings.defaults.review_evidence)}, review Choice {String(runtimeSettings.defaults.review_choice)}.</p>
        <div className="settings-fields">
          <SettingsField id="detection-enter" label="Draft detection enter (JEV_ENTER)" value={draft.detection_enter} min={0} onChange={(value) => onDraftChange('detection_enter', value)} disabled={!runtimeSettings.editable || saving} />
          <SettingsField id="detection-stay" label="Draft detection stay (JEV_STAY)" value={draft.detection_stay} min={0} onChange={(value) => onDraftChange('detection_stay', value)} disabled={!runtimeSettings.editable || saving} />
          <SettingsField id="review-evidence" label="Draft review evidence" value={draft.review_evidence} min={0} onChange={(value) => onDraftChange('review_evidence', value)} disabled={!runtimeSettings.editable || saving} />
          <SettingsField id="review-choice" label="Draft review Choice" value={draft.review_choice} min={0} onChange={(value) => onDraftChange('review_choice', value)} disabled={!runtimeSettings.editable || saving} />
        </div>
        {runtimeSettings.editable ? <>
          <label className="settings-credential" htmlFor="settings-credential">MinusPod password</label>
          <input id="settings-credential" type="password" autoComplete="off" value={credential} onChange={(event) => onCredentialChange(event.target.value)} disabled={saving} />
          <p className="card-note">Checked locally for this save only. Not saved in browser storage. Cleared after each attempt.</p>
          <div className="settings-actions"><button className="refresh settings-save" type="submit" disabled={saving || !dirty}>{saving ? 'Saving...' : 'Save settings'}</button>{dirty && <span className="settings-dirty">Unsaved changes</span>}</div>
        </> : <p className="card-note">Editing is disabled because the MinusPod password is not configured.</p>}
        {message && <p className="settings-result success" role="status">{message}</p>}
      </form>}
    </> : <p className="card-note">Review settings are not reported by this backend.</p>}
    {error && <p className="settings-result error" role="alert">{error}</p>}
  </article>
}

function SettingsField({ id, label, value, min, onChange, disabled }: { id: string; label: string; value: string; min: number; onChange: (value: string) => void; disabled: boolean }) {
  return <div className="settings-field"><label htmlFor={id}>{label}</label><input id={id} type="number" inputMode="decimal" min={min} max={1} step="any" required value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled} /></div>
}

function RuntimeStats({ stats, statsError }: { stats: Stats | null; statsError: string | null }) {
  if (!stats) return <div className="notice" role="status">{statsError ? 'Runtime statistics unavailable.' : 'Loading runtime statistics...'}</div>
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
      {stats.review && <>
        <Metric label="Review outcomes" value={stats.review.count.toLocaleString()} detail={`${stats.review.outcomes.confirmed} confirmed, ${stats.review.outcomes.adjusted} adjusted, ${stats.review.outcomes.rejected} rejected`} />
        <Metric label="Review inconclusive" value={stats.review.outcomes.inconclusive.toLocaleString()} detail={`${stats.review.outcomes.invalid_request} invalid requests, ${stats.review.outcomes.internal_error} internal errors`} />
        <Metric label="Review upstream failures" value={stats.review.outcomes.upstream_error.toLocaleString()} detail="Upstream failures are separate from inconclusive reviews" />
        <Metric label="Average review latency" value={formatLatency(stats.review.latency_ms.average, stats.review.latency_ms.count)} detail="Completed review attempts in this process" />
      </>}
    </div>
    {stats.review && <>
      <RefinementStats refinement={stats.review.refinement} />
      <ReviewReasons reasons={stats.review.reasons} />
    </>}
    <div className="runtime-details">
      <div><span>Process uptime</span><strong>{formatDuration(stats.uptime_seconds)}</strong></div>
      <div><span>Configured workers</span><strong>{stats.scope.configured_workers ?? 'Unknown'}</strong></div>
      <p>Stats are local to process {stats.scope.pid}. Configured workers is an optional environment hint, not a discovered worker count. With multiple workers, requests may alternate between per-worker counters rather than forming a container-wide total. Full MinusPod round-trip timing requires client timing. Cache hits do not call Jev; failed cache fetches count as misses. Failed or usage-unknown attempts do not add to the cost estimate.</p>
    </div>
  </>
}

function RefinementStats({ refinement }: { refinement: RefinementStats | undefined }) {
  if (!refinement) return <div className="notice refinement-summary" role="status">Boundary refinement counters are not reported by this backend.</div>
  const skipped = refinement.skipped
  const count = (value: number | undefined) => value == null ? 'Not reported' : value.toLocaleString()
  return <section className="refinement-summary" aria-labelledby="refinement-title">
    <div className="section-heading"><div><h3 id="refinement-title">Boundary refinement</h3><p>Candidate-pair refinement counters are separate from coarse review outcomes.</p><p>An attempt starts when boundary selection begins. The model ranks expanded start and end candidates, then chooses a valid pair jointly.</p></div></div>
    <div className="metric-grid">
      <Metric label="Refinement attempts" value={count(refinement.attempted)} detail="Reviews that started boundary selection" />
      <Metric label="Refinement completed" value={count(refinement.completed)} detail={`${count(refinement.changed)} changed, ${count(refinement.unchanged)} unchanged`} />
      <Metric label="Refinement inconclusive" value={count(refinement.inconclusive)} detail="Uncertain selection or invalid boundary pair" />
      <Metric label="Refinement upstream failures" value={count(refinement.upstream_error)} detail="Choice request failed or returned an invalid response" />
    </div>
    <article className="card refinement-skips"><h4>Refinement skipped</h4><dl>
      <Definition label="Disabled" value={count(skipped?.disabled)} />
      <Definition label="Missing word timings" value={count(skipped?.missing_word_timings)} />
      <Definition label="Insufficient evidence" value={count(skipped?.insufficient_evidence)} />
      <Definition label="Ambiguous spans" value={count(skipped?.ambiguous_spans)} />
      <Definition label="No overlapping span" value={count(skipped?.no_overlapping_span)} />
      <Definition label="No valid pairs before ranking" value={count(skipped?.no_valid_pairs)} />
    </dl></article>
    <p className="refinement-note">Changed and unchanged compare the selected boundary pair with the original candidate at the 0.1 s tolerance. They do not include coarse span adjustments counted under review outcomes.</p>
  </section>
}

function ReviewReasons({ reasons }: { reasons: Record<string, number> }) {
  const entries = Object.entries(reasons).filter(([, value]) => value > 0)
  return <article className="card refinement-skips"><h4>Review reasons</h4>
    {entries.length ? <dl>{entries.map(([reason, value]) => <Definition key={reason} label={reason.replaceAll('_', ' ')} value={value.toLocaleString()} />)}</dl> : <p className="card-note">No review reasons recorded.</p>}
  </article>
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <article className="metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>
}

export default App
