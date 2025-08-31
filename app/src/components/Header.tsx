import React from 'react'
import { Link, NavLink } from 'react-router-dom'
import { BarChart3, Table2 } from 'lucide-react'

export default function Header() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-xl text-sm font-medium ${isActive ? 'bg-slate-900 text-white' : 'text-slate-700 hover:bg-slate-200'}`

  return (
    <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="text-lg font-semibold tracking-tight">UPA-F Dashboard</Link>
        <nav className="flex items-center gap-2">
          <NavLink to="/" className={linkClass} end><BarChart3 className="inline size-4 mr-1" /> Live Edge</NavLink>
          <NavLink to="/predictions" className={linkClass}><Table2 className="inline size-4 mr-1" /> Predictions</NavLink>
          <NavLink to="/teams" className={linkClass}><Table2 className="inline size-4 mr-1" /> Teams</NavLink>
        </nav>
      </div>
    </header>
  )
}
