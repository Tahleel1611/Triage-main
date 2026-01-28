"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  BarChart3,
  Monitor,
  Database,
  Cpu,
  Wifi,
  ChevronLeft,
  ChevronRight,
  Stethoscope,
  ExternalLink,
  Users,
  Clock,
  Moon,
  Sun,
  Volume2,
  VolumeX,
  Bed,
  LogOut,
  Loader2,
} from "lucide-react";
import { useDashboard } from "@/lib/DashboardContext";
import { useAuth } from "@/lib/AuthContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SystemStatus {
  api_connected: boolean;
  api_latency: number | null;
  db_status: "connected" | "disconnected" | "unknown";
  model_status: "loaded" | "loading" | "error" | "unknown";
}

const NAV_ITEMS = [
  {
    label: "Command Center",
    href: "/",
    icon: Monitor,
    description: "Live triage queue",
  },
  {
    label: "Patient Kiosk",
    href: "/patient/triage",
    icon: Stethoscope,
    description: "Submit symptoms",
  },
  {
    label: "Analytics",
    href: "http://localhost:8501",
    icon: BarChart3,
    description: "Streamlit dashboard",
    external: true,
  },
];

export default function ShellLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const { token, loading: authLoading, logout, user } = useAuth();
  const [collapsed, setCollapsed] = useState(false);
  const { darkMode, setDarkMode, audioEnabled, setAudioEnabled, bedCapacity } = useDashboard();
  const [status, setStatus] = useState<SystemStatus>({
    api_connected: false,
    api_latency: null,
    db_status: "unknown",
    model_status: "unknown",
  });
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [mounted, setMounted] = useState(false);

  // Check if we're on the login page - skip auth guard and shell
  const isLoginPage = pathname === "/login";

  // Auth guard: redirect to login if not authenticated (skip for login page)
  useEffect(() => {
    if (!isLoginPage && !authLoading && !token) {
      router.replace("/login");
    }
  }, [authLoading, token, router, isLoginPage]);

  // Mark as mounted (client-side only) to avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
    setCurrentTime(new Date());
  }, []);

  // Update clock only after mounted
  useEffect(() => {
    if (!mounted) return;
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, [mounted]);

  // Check system status on mount and periodically (with latency measurement)
  useEffect(() => {
    const checkStatus = async () => {
      const startTime = performance.now();
      try {
        const res = await fetch(`${API_BASE}/health`, {
          method: "GET",
          signal: AbortSignal.timeout(3000),
        });
        const latency = Math.round(performance.now() - startTime);
        if (res.ok) {
          const data = await res.json();
          setStatus({
            api_connected: true,
            api_latency: latency,
            db_status: data.database || "connected",
            model_status: data.models || "loaded",
          });
        } else {
          setStatus((prev) => ({ ...prev, api_connected: true, api_latency: latency }));
        }
      } catch {
        setStatus((prev) => ({ ...prev, api_connected: false, api_latency: null }));
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  // For login page, render children without the shell
  if (isLoginPage) {
    return <>{children}</>;
  }

  // Show loading spinner while checking auth (only for protected pages)
  if (authLoading || !token) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-slate-400 text-sm">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex h-screen transition-colors duration-300 ${darkMode ? "bg-slate-950" : "bg-gray-100"}`}>
      {/* Sidebar */}
      <aside
        className={`${
          collapsed ? "w-16" : "w-64"
        } bg-slate-900 text-white flex flex-col transition-all duration-300 ease-in-out`}
      >
        {/* Logo */}
        <div className="p-4 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shrink-0 shadow-lg">
              <Activity className="w-6 h-6" />
            </div>
            {!collapsed && (
              <div>
                <h1 className="font-bold text-lg leading-tight">TriageAI</h1>
                <p className="text-xs text-slate-400">Hospital OS v2.0</p>
              </div>
            )}
          </div>
        </div>

        {/* Clock & Controls */}
        {!collapsed && (
          <div className="px-4 py-3 border-b border-slate-700/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-400">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-mono" suppressHydrationWarning>
                  {currentTime ? currentTime.toLocaleTimeString("en-US", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  }) : "--:--:--"}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className="p-1.5 rounded-md hover:bg-slate-700 transition-colors text-slate-400 hover:text-white"
                  title={darkMode ? "Light mode" : "Dark mode"}
                >
                  {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => setAudioEnabled(!audioEnabled)}
                  className={`p-1.5 rounded-md hover:bg-slate-700 transition-colors ${audioEnabled ? "text-green-400" : "text-slate-500"}`}
                  title={audioEnabled ? "Mute alerts" : "Enable alerts"}
                >
                  {audioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-1" suppressHydrationWarning>
              {currentTime ? currentTime.toLocaleDateString("en-US", {
                weekday: "long",
                month: "short",
                day: "numeric",
              }) : "Loading..."}
            </p>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 p-2 space-y-1">
          <p
            className={`text-xs text-slate-500 uppercase tracking-wider font-medium px-3 py-2 ${
              collapsed ? "hidden" : ""
            }`}
          >
            Navigation
          </p>
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;

            if (item.external) {
              return (
                <a
                  key={item.href}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-slate-300 hover:bg-slate-800 hover:text-white"
                  title={collapsed ? item.label : undefined}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {!collapsed && (
                    <>
                      <span className="flex-1">{item.label}</span>
                      <ExternalLink className="w-3 h-3 opacity-50" />
                    </>
                  )}
                </a>
              );
            }

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                  isActive
                    ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                }`}
                title={collapsed ? item.label : undefined}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Bed Capacity Widget */}
        {!collapsed && (
          <div className="px-3 py-2 border-t border-slate-700">
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Bed className="w-3.5 h-3.5" />
                <span>Bed Capacity</span>
              </div>
              <span className="text-xs font-medium text-slate-300">
                {bedCapacity.occupied}/{bedCapacity.total}
              </span>
            </div>
            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 rounded-full ${
                  bedCapacity.occupied / bedCapacity.total > 0.9
                    ? "bg-red-500"
                    : bedCapacity.occupied / bedCapacity.total > 0.7
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
                style={{ width: `${(bedCapacity.occupied / bedCapacity.total) * 100}%` }}
              />
            </div>
            <p className="text-xs text-slate-500 mt-1">
              {Math.round((bedCapacity.occupied / bedCapacity.total) * 100)}% occupied
            </p>
          </div>
        )}

        {/* System Status */}
        <div className="p-3 border-t border-slate-700">
          {!collapsed ? (
            <div className="space-y-2">
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">
                System Status
              </p>
              <div className="space-y-1.5">
                <StatusIndicator
                  icon={Wifi}
                  label="API Server"
                  status={status.api_connected ? "ok" : "error"}
                  latency={status.api_latency}
                />
                <StatusIndicator
                  icon={Database}
                  label="Database"
                  status={
                    status.db_status === "connected"
                      ? "ok"
                      : status.db_status === "unknown"
                      ? "loading"
                      : "error"
                  }
                />
                <StatusIndicator
                  icon={Cpu}
                  label="AI Models"
                  status={
                    status.model_status === "loaded"
                      ? "ok"
                      : status.model_status === "loading"
                      ? "loading"
                      : status.model_status === "unknown"
                      ? "loading"
                      : "error"
                  }
                />
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  status.api_connected ? "bg-green-500 shadow-green-500/50 shadow-lg" : "bg-red-500"
                }`}
                title={status.api_connected ? "System Online" : "System Offline"}
              />
            </div>
          )}
        </div>

        {/* User Info & Logout */}
        <div className="p-3 border-t border-slate-700">
          {!collapsed ? (
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {user?.email?.split("@")[0] || "User"}
                </p>
                <p className="text-xs text-slate-400">
                  {user?.role === "STAFF" ? "Staff" : "Patient"}
                </p>
              </div>
              <button
                onClick={logout}
                className="p-2 rounded-lg text-slate-400 hover:text-red-400 hover:bg-slate-800 transition-colors"
                title="Sign out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={logout}
              className="w-full flex justify-center p-2 rounded-lg text-slate-400 hover:text-red-400 hover:bg-slate-800 transition-colors"
              title="Sign out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Collapse Toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-3 border-t border-slate-700 hover:bg-slate-800 transition-colors flex items-center justify-center"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <ChevronLeft className="w-5 h-5" />
              <span>Collapse</span>
            </div>
          )}
        </button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}

function StatusIndicator({
  icon: Icon,
  label,
  status,
  latency,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  status: "ok" | "loading" | "error";
  latency?: number | null;
}) {
  const colors = {
    ok: "text-green-400",
    loading: "text-yellow-400 animate-pulse",
    error: "text-red-400",
  };

  const dots = {
    ok: "bg-green-500 shadow-green-500/50",
    loading: "bg-yellow-500 shadow-yellow-500/50",
    error: "bg-red-500 shadow-red-500/50",
  };

  return (
    <div className="flex items-center gap-2 text-sm">
      <Icon className={`w-4 h-4 ${colors[status]}`} />
      <span className="text-slate-300 flex-1">
        {label}
        {latency !== null && latency !== undefined && (
          <span className={`ml-1 text-xs ${latency < 100 ? "text-green-400" : latency < 300 ? "text-yellow-400" : "text-red-400"}`}>
            ({latency}ms)
          </span>
        )}
      </span>
      <span className={`w-2 h-2 rounded-full shadow-lg ${dots[status]}`} />
    </div>
  );
}
