import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import Table from '../components/Table'
import { loadCsv } from '../lib/csv'
import { useStore } from '../state/useStore'
import { TrendingUp } from 'lucide-react'

type EdgeRow = {
  week: string
  home_team: string
  away_team: string
  UPA_pred_spread_home_minus_away: number
  market_spread_home_minus_away: number | null
  value_edge_points: number | null
  sportsbook?: string
  as_of?: string
}

export default function LiveEdge() {
  const [rows, setRows] = useState<EdgeRow[]>([])
  const query = useStore(s => s.query)
  const setQuery = useStore(s => s.setQuery)

  useEffect(() => {
    loadCsv('./data/live_edge_report_week1.csv').then(d => setRows(d as EdgeRow[]))
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
    { key: 'UPA_pred_spread_home_minus_away', header: 'UPA Spread', render: (r: EdgeRow) => (r.UPA_pred_spread_home_minus_away >= 0 ? '+' : '') + r.UPA_pred_spread_home_minus_away.toFixed(2) },
    { key: 'market_spread_home_minus_away', header: 'Market', render: (r: EdgeRow) => r.market_spread_home_minus_away == null ? '' : (r.market_spread_home_minus_away >= 0 ? '+' : '') + r.market_spread_home_minus_away.toFixed(1) },
    { key: 'value_edge_points', header: 'Edge', render: (r: EdgeRow) => r.value_edge_points == null ? '' : (r.value_edge_points >= 0 ? '+' : '') + r.value_edge_points.toFixed(2) },
    { key: 'sportsbook', header: 'Book' },
    { key: 'as_of', header: 'As of' },
  ] as const

  return (
    <div className="grid gap-6">
      <Card title="Filter">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm text-slate-700">Search</label>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Team, week..." className="bg-white border border-slate-300 rounded-xl px-3 py-2 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-slate-400" />
          <div className="ml-auto flex items-center gap-2 text-sm text-slate-600">
            <TrendingUp className="size-4" />
            <span>Higher “Edge” = bigger model vs market difference</span>
          </div>
        </div>
      </Card>

      <Card title="Live Edge — Week 1">
        <Table columns={cols as any} rows={filtered} />
      </Card>
    </div>
  )
}
