import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import Table from '../components/Table'
import { loadCsv } from '../lib/csv'
import { useStore } from '../state/useStore'

type TeamRow = {
  team: string
  talent_score_0_100: number
  wrps_percent_0_100: number
  prev_season_sos_rank_1_133: number
  transfer_net_points: number
  momentum_points: number
}

export default function Teams() {
  const [rows, setRows] = useState<TeamRow[]>([])
  const query = useStore(s => s.query)
  const setQuery = useStore(s => s.setQuery)

  useEffect(() => {
    loadCsv('./data/upa_team_inputs_datadriven_v0.csv').then(d => setRows(d as TeamRow[]))
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return rows
    return rows.filter(r => r.team.toLowerCase().includes(q))
  }, [rows, query])

  const cols = [
    { key: 'team', header: 'Team' },
    { key: 'talent_score_0_100', header: 'Talent', render: (r: TeamRow) => r.talent_score_0_100?.toFixed(1) },
    { key: 'wrps_percent_0_100', header: 'WRPS %', render: (r: TeamRow) => r.wrps_percent_0_100?.toFixed(1) },
    { key: 'prev_season_sos_rank_1_133', header: 'SoS Rank' },
    { key: 'transfer_net_points', header: 'Portal pts', render: (r: TeamRow) => (r.transfer_net_points >= 0 ? '+' : '') + r.transfer_net_points.toFixed(1) },
    { key: 'momentum_points', header: 'Momentum', render: (r: TeamRow) => (r.momentum_points >= 0 ? '+' : '') + r.momentum_points.toFixed(1) },
  ] as const

  return (
    <div className="grid gap-6">
      <Card title="Search">
        <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search team..." className="bg-white border border-slate-300 rounded-xl px-3 py-2 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-slate-400" />
      </Card>
      <Card title="Team Inputs">
        <Table columns={cols as any} rows={filtered} />
      </Card>
    </div>
  )
}
