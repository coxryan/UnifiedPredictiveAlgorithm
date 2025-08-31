import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'
import App from './pages/App'
import LiveEdge from './pages/LiveEdge'
import Predictions from './pages/Predictions'
import Teams from './pages/Teams'

const router = createBrowserRouter([

  { path: '/', element: <App />, children: [
    { index: true, element: <LiveEdge /> },
    { path: 'predictions', element: <Predictions /> },
    { path: 'teams', element: <Teams /> },
  ]}
], {
  basename: '/UnifiedPredictiveAlgorithm/',
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)
