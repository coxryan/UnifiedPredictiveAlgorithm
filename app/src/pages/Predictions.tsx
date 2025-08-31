import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import Table from '../components/Table'
import { loadCsv } from '../lib/csv'
import { useStore } from '../state/useStore'

type PredRow = {
  week: string
  home_team: string
  away_team: string
  UPA_pred_spread_home_minus_away: number
  baseline_power_diff: number
  HFA_points: number
}

export default function Predictions() {
  const [rows, setRows] = useState<PredRow[]>([])
  const query = useStore(s => s.query)

  useEffect(() => {
    loadCsv('./data/upa_predictions_week1_datadriven_v0.csv').then(d => setRows(d as PredRow[]))
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return rows
    return rows.filter(r =>
      r.home_team.toLowerCase().includes(q) ||
      r.away_team.toLowerCase().includes(q) ||
      String(r.week).toLowerCase().includes(q)
    )
  }, [rows, query])

  const cols = [
    { key: 'week', header: 'Week' },
    { key: 'home_team', header: 'Home' },
    { key: 'away_team', header: 'Away' },
    { key: 'UPA_pred_spread_home_minus_away', header: 'UPA Spread', render: (r: PredRow) => (r.UPA_pred_spread_home_minus_away >= 0 ? '+' : '') + r.UPA_pred_spread_home_minus_away.toFixed(2) },
    { key: 'baseline_power_diff', header: 'Power Δ', render: (r: PredRow) => (r.baseline_power_diff >= 0 ? '+' : '') + r.baseline_power_diff.toFixed(2) },
    { key: 'HFA_points', header: 'HFA', render: (r: PredRow) => (r.HFA_points >= 0 ? '+' : '') + r.HFA_points.toFixed(1) },
  ] as const

  return (
    <div className="grid gap-6">
      <Card title="Predictions — Week 1">
        <Table columns={cols as any} rows={filtered} />
      </Card>
    </div>
  )
}
