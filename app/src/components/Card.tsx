import React from 'react'
import clsx from 'clsx'

export default function Card({ title, children, className }: { title?: string, children: React.ReactNode, className?: string }) {
  return (
    <section className={clsx('bg-white rounded-2xl shadow-sm border border-slate-200', className)}>
      {title && <div className="px-4 py-3 border-b border-slate-200 font-semibold">{title}</div>}
      <div className="p-4">{children}</div>
    </section>
  )
}
