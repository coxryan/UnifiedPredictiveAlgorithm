import React from 'react'

type Column<T> = {
  key: keyof T
  header: string
  width?: string | number
  render?: (row: T) => React.ReactNode
}

export default function Table<T extends Record<string, any>>({ columns, rows }: { columns: Column<T>[], rows: T[] }) {
  return (
    <div className="overflow-auto rounded-xl border border-slate-200">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-100 sticky top-0">
          <tr>
            {columns.map(col => (
              <th key={String(col.key)} className="text-left font-semibold text-slate-700 px-3 py-2" style={{ width: col.width }}>{col.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="odd:bg-white even:bg-slate-50">
              {columns.map(col => (
                <td key={String(col.key)} className="px-3 py-2 whitespace-nowrap">{col.render ? col.render(row) : String(row[col.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
