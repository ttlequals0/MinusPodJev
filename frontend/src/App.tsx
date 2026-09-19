import { useState, useEffect } from 'react'
import axios from 'axios'
import toast from 'react-hot-toast'

interface HealthCheck {
  status: string
  environment: string
  version: string
}

function App() {
  const [health, setHealth] = useState<HealthCheck | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get<HealthCheck>('/api/health')
        setHealth(response.data)
        toast.success('Connected to backend!')
      } catch (error) {
        console.error('Health check failed:', error)
        toast.error('Failed to connect to backend')
      } finally {
        setLoading(false)
      }
    }

    checkHealth()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="mx-auto max-w-4xl">
          <div className="rounded-xl bg-white p-8 shadow-lg">
            <h1 className="mb-6 text-center text-4xl font-bold text-gray-800">
              Jev Proxy
            </h1>

            <div className="mb-8">
              <h2 className="mb-4 text-2xl font-semibold text-gray-700">
                Welcome to Your New Application!
              </h2>
              <p className="leading-relaxed text-gray-600">
                This is a full-stack web application template with a Python FastAPI backend and
                React frontend. Everything is containerized and ready for development.
              </p>
            </div>

            <div className="mb-8 rounded-lg bg-gray-50 p-6">
              <h3 className="mb-3 text-lg font-semibold text-gray-700">Backend Health Status</h3>
              {loading ? (
                <p className="text-gray-600">Checking backend connection...</p>
              ) : health ? (
                <div className="space-y-2">
                  <p className="text-sm">
                    <span className="font-medium text-gray-700">Status:</span>{' '}
                    <span className="font-semibold text-green-600">{health.status}</span>
                  </p>
                  <p className="text-sm">
                    <span className="font-medium text-gray-700">Environment:</span>{' '}
                    <span className="text-blue-600">{health.environment}</span>
                  </p>
                  <p className="text-sm">
                    <span className="font-medium text-gray-700">Version:</span>{' '}
                    <span className="text-gray-600">{health.version}</span>
                  </p>
                </div>
              ) : (
                <p className="text-red-600">Failed to connect to backend</p>
              )}
            </div>

            <div className="border-t pt-6">
              <h3 className="mb-3 text-lg font-semibold text-gray-700">Get Started</h3>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-start">
                  <span className="mr-2 text-blue-500">"</span>
                  <span>
                    Edit backend code in{' '}
                    <code className="rounded bg-gray-100 px-2 py-1 text-sm">backend/app</code>
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-blue-500">"</span>
                  <span>
                    Edit frontend code in{' '}
                    <code className="rounded bg-gray-100 px-2 py-1 text-sm">frontend/src</code>
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-blue-500">"</span>
                  <span>
                    API documentation at{' '}
                    <a href="/api/docs" className="text-blue-600 hover:underline">
                      /api/docs
                    </a>
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-blue-500">"</span>
                  <span>
                    Run tests with{' '}
                    <code className="rounded bg-gray-100 px-2 py-1 text-sm">
                      ./scripts/run_ci_checks.sh
                    </code>
                  </span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
