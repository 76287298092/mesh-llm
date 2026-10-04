import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Activity, Layers3, Play, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { env } from '@/lib/env'
import type { StatusPayload } from '@/lib/api/types'

type SplitReadiness = {
  model_ref: string
  verdict: string
  participant_count: number
  exclusion_count: number
  participants: Array<{
    node_id: string
    short_node_id: string
    role: string
    source: string
    model_source_state: string
  }>
  exclusions: Array<{
    node_id: string
    short_node_id: string
    reason: string
    recommendation: string
  }>
  recommendations: string[]
}

async function responseError(response: Response) {
  const body = await response.text()
  return body || `HTTP ${response.status}`
}

async function fetchSplitReadiness(modelRef: string): Promise<SplitReadiness> {
  const url = new URL(`${env.managementApiUrl}/api/diagnostics/split-readiness`)
  url.searchParams.set('model_ref', modelRef)
  const response = await fetch(url)
  if (!response.ok) throw new Error(await responseError(response))
  return response.json() as Promise<SplitReadiness>
}

async function requestRuntimeLoad(model: string): Promise<{ loaded: string; instance_id: string }> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model })
  })
  if (!response.ok) throw new Error(await responseError(response))
  return response.json() as Promise<{ loaded: string; instance_id: string }>
}

function readableVerdict(value: string) {
  return value.replaceAll('_', ' ')
}

function readableBytes(value?: number) {
  if (value == null || !Number.isFinite(value)) return 'Unknown'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  let unit = 0
  while (amount >= 1000 && unit < units.length - 1) {
    amount /= 1000
    unit += 1
  }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

export function ShardSchedulerPanel({ status }: { status: StatusPayload }) {
  const modelOptions = useMemo(() => {
    const values = [
      status.model_name,
      ...(status.available_models ?? []),
      ...status.models.flatMap((model) => [model.source_ref, model.name])
    ]
    return [...new Set(values.map((value) => value?.trim()).filter((value): value is string => Boolean(value)))]
  }, [status.available_models, status.model_name, status.models])
  const [modelRef, setModelRef] = useState('')

  useEffect(() => {
    if (modelOptions.length > 0 && !modelOptions.includes(modelRef)) setModelRef(modelOptions[0] ?? '')
  }, [modelOptions, modelRef])

  const readinessQuery = useQuery({
    queryKey: ['split-readiness', modelRef],
    queryFn: () => fetchSplitReadiness(modelRef),
    enabled: modelRef.length > 0,
    staleTime: 10_000,
    refetchInterval: 15_000
  })
  const loadMutation = useMutation({ mutationFn: requestRuntimeLoad })
  const stages = status.runtime?.stages ?? []

  return (
    <section aria-label="Shard scheduler" className="overflow-hidden rounded-[var(--radius)] border border-border-soft bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border-soft px-4 py-3">
        <div className="flex items-center gap-2">
          <Layers3 className="size-4 text-fg-muted" aria-hidden="true" />
          <h2 className="text-sm font-semibold text-fg">Shard scheduling and management</h2>
        </div>
        <div className="flex items-center gap-2">
          <label className="sr-only" htmlFor="split-model-ref">Model</label>
          <select
            id="split-model-ref"
            className="h-9 max-w-[min(64vw,360px)] rounded-md border border-border-soft bg-background px-2 text-sm text-fg"
            value={modelRef}
            onChange={(event) => {
              loadMutation.reset()
              setModelRef(event.target.value)
            }}
            disabled={modelOptions.length === 0}
          >
            {modelOptions.length === 0 ? <option value="">No model references</option> : null}
            {modelOptions.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
          <Button
            variant="outline"
            size="sm"
            disabled={!modelRef || readinessQuery.isFetching}
            onClick={() => void readinessQuery.refetch()}
            aria-label="Refresh split readiness"
          >
            <RefreshCw className="size-3.5" aria-hidden="true" />
          </Button>
        </div>
      </div>

      <div className="grid gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge tone={readinessQuery.data?.verdict === 'ready' ? 'good' : 'warn'}>
              {readinessQuery.data ? readableVerdict(readinessQuery.data.verdict) : readinessQuery.isLoading ? 'Checking' : 'Unavailable'}
            </StatusBadge>
            {readinessQuery.data ? (
              <span className="text-xs text-fg-faint">
                {readinessQuery.data.participant_count} eligible · {readinessQuery.data.exclusion_count} excluded
              </span>
            ) : null}
          </div>

          {readinessQuery.error ? (
            <p role="alert" className="rounded-md border border-border-soft bg-background px-3 py-2 text-xs text-fg-muted">
              Readiness report unavailable: {readinessQuery.error.message}
            </p>
          ) : null}

          {readinessQuery.data?.participants.length ? (
            <div className="overflow-x-auto rounded-md border border-border-soft">
              <table className="w-full text-left text-xs">
                <thead className="bg-panel-strong text-fg-faint">
              <tr><th className="px-3 py-2 font-medium">Node</th><th className="px-3 py-2 font-medium">Role</th><th className="px-3 py-2 font-medium">Model source</th></tr>
                </thead>
                <tbody>
                  {readinessQuery.data.participants.map((participant) => (
                    <tr key={participant.node_id} className="border-t border-border-soft">
                      <td className="px-3 py-2 font-mono">{participant.short_node_id}</td>
                      <td className="px-3 py-2">{participant.role}</td>
                      <td className="px-3 py-2">{participant.model_source_state.replaceAll('_', ' ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : readinessQuery.data ? (
            <p className="text-xs text-fg-faint">No node currently meets split readiness for this model.</p>
          ) : null}

          {readinessQuery.data?.exclusions.length ? (
            <details className="text-xs text-fg-muted">
              <summary className="cursor-pointer">Show {readinessQuery.data.exclusions.length} excluded nodes</summary>
              <ul className="mt-2 space-y-1">
                {readinessQuery.data.exclusions.map((entry) => (
                  <li key={entry.node_id}>{entry.short_node_id}: {entry.reason.replaceAll('_', ' ')}</li>
                ))}
              </ul>
            </details>
          ) : null}

          <div className="flex flex-wrap items-center gap-2 border-t border-border-soft pt-3">
            <Button
              size="sm"
              disabled={!modelRef || loadMutation.isPending}
              onClick={() => loadMutation.mutate(modelRef)}
            >
              <Play className="mr-1.5 size-3.5" aria-hidden="true" />
              {loadMutation.isPending ? 'Submitting…' : 'Request runtime load'}
            </Button>
            {loadMutation.data ? (
              <span role="status" className="text-xs text-fg-muted">
                Load accepted: {loadMutation.data.loaded} · {loadMutation.data.instance_id}
              </span>
            ) : null}
            {loadMutation.error ? (
              <span role="alert" className="text-xs text-destructive">{loadMutation.error.message}</span>
            ) : null}
          </div>
          <p className="text-xs leading-5 text-fg-faint">
            This request loads the model into this node’s runtime. Layer assignments are created by the split runtime coordinator; this control does not create a new split assignment. The readiness view reports which peers are currently eligible.
          </p>
        </div>

        <div className="min-w-0 space-y-3">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-fg-faint">
            <Activity className="size-3.5" aria-hidden="true" /> Active shard stages
          </div>
          {stages.length > 0 ? (
            <div className="max-h-64 overflow-auto rounded-md border border-border-soft">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-panel-strong text-fg-faint">
                  <tr><th className="px-3 py-2 font-medium">Stage</th><th className="px-3 py-2 font-medium">Layers</th><th className="px-3 py-2 font-medium">State</th><th className="px-3 py-2 font-medium">Node</th><th className="px-3 py-2 font-medium">Local package</th></tr>
                </thead>
                <tbody>
                  {stages.map((stage) => (
                    <tr key={`${stage.model_id}:${stage.stage_id}`} className="border-t border-border-soft">
                      <td className="px-3 py-2 font-mono">{stage.stage_id}{stage.stage_index == null ? '' : ` · ${stage.stage_index}`}</td>
                      <td className="px-3 py-2">{stage.layer_start}–{stage.layer_end}</td>
                      <td className="px-3 py-2" title={stage.error}>{stage.state}</td>
                      <td className="px-3 py-2 font-mono">{stage.node_id?.slice(0, 8) ?? 'local'}</td>
                      <td className="px-3 py-2">{stage.materialized_pinned ? `Pinned · ${readableBytes(stage.materialized_bytes)}` : readableBytes(stage.materialized_bytes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="rounded-md border border-border-soft bg-background px-3 py-4 text-xs text-fg-faint">
              No active split stages have been reported by this node.
            </p>
          )}
          <p className="text-xs text-fg-faint">Stage readiness and assigned layer ranges come from the runtime status report.</p>
        </div>
      </div>
    </section>
  )
}
