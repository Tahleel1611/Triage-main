"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/AuthContext";
import {
  Activity,
  Lock,
  Mail,
  User,
  AlertCircle,
  Loader2,
  Database,
  Shield,
  Cpu,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SystemHealth {
  status: string;
  database: string;
  models: string;
  dev_mode: boolean;
}

export default function LoginPage() {
  const router = useRouter();
  const { login, register, token, loading: authLoading, error, clearError } = useAuth();
  
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"PATIENT" | "STAFF">("PATIENT");
  const [submitting, setSubmitting] = useState(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [healthError, setHealthError] = useState(false);

  // Redirect if already logged in
  useEffect(() => {
    if (token && !authLoading) {
      router.replace("/");
    }
  }, [token, authLoading, router]);

  // Check system health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        if (res.ok) {
          const data = await res.json();
          setSystemHealth(data);
          setHealthError(false);
        } else {
          setHealthError(true);
        }
      } catch {
        setHealthError(true);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Clear error when switching modes
  useEffect(() => {
    clearError();
  }, [isLogin, clearError]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      let success: boolean;
      if (isLogin) {
        success = await login(email, password);
      } else {
        success = await register(email, password, role);
      }

      if (success) {
        router.replace("/");
      }
    } finally {
      setSubmitting(false);
    }
  };

  // Show loading while checking auth state
  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
      </div>
    );
  }

  // Don't render if already authenticated
  if (token) {
    return null;
  }

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-800 via-slate-900 to-slate-950" />
      
      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0wIDBoNDB2NDBIMHoiLz48cGF0aCBkPSJNNDAgMEgwdjQwaDQwVjB6TTEgMXYzOGgzOFYxSDF6IiBmaWxsPSIjMWUyOTNiIiBmaWxsLW9wYWNpdHk9Ii4zIi8+PC9nPjwvc3ZnPg==')] opacity-50" />

      <div className="relative w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl shadow-lg shadow-blue-500/25 mb-4">
            <Activity className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">TriageAI</h1>
          <p className="text-slate-400 mt-1">Hospital Command Center</p>
        </div>

        {/* Login Card */}
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 shadow-2xl">
          {/* Tab Toggle */}
          <div className="flex bg-slate-900/50 rounded-lg p-1 mb-6">
            <button
              type="button"
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                isLogin
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                !isLogin
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Register
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 bg-slate-900/50 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                  placeholder="doctor@hospital.com"
                  required
                  autoComplete="email"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 bg-slate-900/50 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                  placeholder="••••••••"
                  required
                  minLength={8}
                  autoComplete={isLogin ? "current-password" : "new-password"}
                />
              </div>
              {!isLogin && (
                <p className="text-xs text-slate-500 mt-1">
                  Minimum 8 characters
                </p>
              )}
            </div>

            {/* Role Selection (Register only) */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">
                  Role
                </label>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => setRole("PATIENT")}
                    className={`flex-1 py-2.5 px-4 rounded-lg border transition-all flex items-center justify-center gap-2 ${
                      role === "PATIENT"
                        ? "bg-blue-600/20 border-blue-500 text-blue-400"
                        : "bg-slate-900/50 border-slate-600 text-slate-400 hover:border-slate-500"
                    }`}
                  >
                    <User className="w-4 h-4" />
                    Patient
                  </button>
                  <button
                    type="button"
                    onClick={() => setRole("STAFF")}
                    className={`flex-1 py-2.5 px-4 rounded-lg border transition-all flex items-center justify-center gap-2 ${
                      role === "STAFF"
                        ? "bg-purple-600/20 border-purple-500 text-purple-400"
                        : "bg-slate-900/50 border-slate-600 text-slate-400 hover:border-slate-500"
                    }`}
                  >
                    <Shield className="w-4 h-4" />
                    Staff
                  </button>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-medium rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-blue-500/25"
            >
              {submitting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  {isLogin ? "Signing In..." : "Creating Account..."}
                </>
              ) : (
                <>{isLogin ? "Sign In" : "Create Account"}</>
              )}
            </button>
          </form>

          {/* Demo Credentials */}
          {isLogin && (
            <div className="mt-4 p-3 bg-slate-900/50 border border-slate-700 rounded-lg">
              <p className="text-xs text-slate-400 mb-2">Demo Credentials:</p>
              <div className="space-y-1 text-xs">
                <p className="text-slate-500">
                  <span className="text-slate-400">Patient:</span> patient@test.com / testpassword123
                </p>
                <p className="text-slate-500">
                  <span className="text-slate-400">Staff:</span> staff@hospital.com / staffpassword123
                </p>
              </div>
            </div>
          )}

          {/* Divider */}
          <div className="my-6 flex items-center gap-4">
            <div className="flex-1 h-px bg-slate-700" />
            <span className="text-xs text-slate-500 uppercase tracking-wider">
              System Status
            </span>
            <div className="flex-1 h-px bg-slate-700" />
          </div>

          {/* System Status Footer */}
          <div className="flex flex-wrap items-center justify-center gap-4">
            {/* API Status */}
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  systemHealth && !healthError
                    ? "bg-green-500 animate-pulse"
                    : "bg-red-500"
                }`}
              />
              <span className="text-xs text-slate-400">
                {systemHealth && !healthError ? "API Online" : "API Offline"}
              </span>
            </div>

            {/* Database Status */}
            <div className="flex items-center gap-2">
              <Database className={`w-3.5 h-3.5 ${
                systemHealth?.database === "ok" ? "text-green-400" : "text-slate-500"
              }`} />
              <span
                className={`text-xs ${
                  systemHealth?.database === "ok"
                    ? "text-green-400"
                    : "text-red-400"
                }`}
              >
                {systemHealth?.database === "ok" ? "DB OK" : "DB Down"}
              </span>
            </div>

            {/* Models Status */}
            <div className="flex items-center gap-2">
              <Cpu className={`w-3.5 h-3.5 ${
                systemHealth?.models === "loaded" || systemHealth?.models === "mock"
                  ? "text-green-400"
                  : "text-slate-500"
              }`} />
              <span className={`text-xs ${
                systemHealth?.models === "loaded" || systemHealth?.models === "mock"
                  ? "text-green-400"
                  : "text-yellow-400"
              }`}>
                {systemHealth?.models || "Models"}
              </span>
            </div>

            {/* Dev Mode Indicator */}
            {systemHealth?.dev_mode && (
              <div className="flex items-center gap-1.5 px-2 py-0.5 bg-yellow-500/10 border border-yellow-500/20 rounded-full">
                <span className="text-xs text-yellow-400">Dev Mode</span>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-slate-500 text-xs mt-6">
          TriageAI Hospital Operating System v2.0
        </p>
      </div>
    </div>
  );
}
