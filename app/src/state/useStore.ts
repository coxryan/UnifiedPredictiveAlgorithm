import { create } from 'zustand'
type State = { week: string | null; setWeek: (w: string | null) => void; query: string; setQuery: (q: string) => void }
export const useStore = create<State>((set) => ({
  week: null, setWeek: (w) => set({ week: w }),
  query: '', setQuery: (q) => set({ query: q }),
}))
