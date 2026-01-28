"use client";

import { useEffect, useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Heart,
  Wind,
  Droplets,
  AlertCircle,
  X,
  RefreshCw,
  TrendingUp,
  ClipboardList,
  AlertTriangle,
  Clock,
  Users,
  Cpu,
  Stethoscope,
  CheckCircle2,
} from "lucide-react";
import ShapChart from "./ShapChart";
import Sparkline from "./Sparkline";
import {
  fetchPatientVitalsHistory,
  transformVitalsForChart,
  fetchHandoffReport,
  VitalHistoryPoint,
  HandoffReport,
} from "../lib/api";
import { useDashboard } from "../lib/DashboardContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Vitals {
  SBP?: number;
  HR?: number;
  RR?: number;
  O2Sat?: number;
}

interface TopFeature {
  feature: string;
  impact: number;
}

interface Patient {
  appointment_id: number;
  patient_id: number;
  token: string;
  triage_level: number;
  priority_score: number;
  action: string | null;
  vitals: Vitals;
  chief_complaint: string;
  shap_ready: boolean;
  top_features?: TopFeature[];
}

interface TriageResult {
  esi_level: number;
  supervised_confidence: number;
  rl_action: string | null;
  shap_values: number[][] | null;
  top_features?: TopFeature[];
  shock_index?: number;
}

const TRIAGE_COLORS: Record<number, string> = {
  1: "bg-red-500",
  2: "bg-orange-500",
  3: "bg-yellow-500",
  4: "bg-green-500",
  5: "bg-blue-500",
};

const TRIAGE_LABELS: Record<number, string> = {
  1: "Resuscitation",
  2: "Emergent",
  3: "Urgent",
  4: "Less Urgent",
  5: "Non-Urgent",
};

export default function StaffDashboard() {
  const { darkMode, playAlert, setBedCapacity } = useDashboard();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [triageResult, setTriageResult] = useState<TriageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [vitalsHistory, setVitalsHistory] = useState<VitalHistoryPoint[]>([]);
  const [vitalsLoading, setVitalsLoading] = useState(false);
  
  // Handoff Mode
  const [handoffMode, setHandoffMode] = useState(false);
  const [handoffReport, setHandoffReport] = useState<HandoffReport | null>(null);
  const [handoffLoading, setHandoffLoading] = useState(false);

  // SSE Connection
  useEffect(() => {
    const eventSource = new EventSource(`${API_BASE}/dashboard/stream`);

    eventSource.onopen = () => setConnected(true);
    eventSource.onerror = () => setConnected(false);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.event_type === "new_patient") {
        setPatients((prev) => {
          const exists = prev.find(
            (p) => p.appointment_id === data.appointment_id
          );
          if (exists) return prev;
          const updated = [...prev, data].sort(
            (a, b) => b.priority_score - a.priority_score
          );
          // Update bed capacity based on patient count
          setBedCapacity({ occupied: Math.min(updated.length + 8, 20), total: 20 });
          return updated;
        });
        // Play audio alert for critical patients (ESI 1-2)
        if (data.triage_level <= 2) {
          playAlert("critical");
        } else if (data.triage_level === 3) {
          playAlert("warning");
        }
      }

      if (data.event_type === "shap_ready") {
        setPatients((prev) =>
          prev.map((p) =>
            p.appointment_id === data.appointment_id
              ? { ...p, shap_ready: true, top_features: data.top_features }
              : p
          )
        );
      }
    };

    return () => eventSource.close();
  }, []);

  const fetchTriageResult = useCallback(async (appointmentId: number) => {
    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/triage/result/${appointmentId}`,
        { credentials: "include" }
      );
      if (res.ok) {
        const data = await res.json();
        setTriageResult(data);
      }
    } catch (err) {
      console.error("Failed to fetch triage result", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchVitalsHistory = useCallback(async (patientId: number) => {
    setVitalsLoading(true);
    try {
      const data = await fetchPatientVitalsHistory(patientId);
      setVitalsHistory(data.history);
    } catch (err) {
      console.error("Failed to fetch vitals history", err);
      setVitalsHistory([]);
    } finally {
      setVitalsLoading(false);
    }
  }, []);

  const openModal = (patient: Patient) => {
    setSelectedPatient(patient);
    if (patient.shap_ready) {
      fetchTriageResult(patient.appointment_id);
    }
    // Fetch vitals history for sparklines
    if (patient.patient_id) {
      fetchVitalsHistory(patient.patient_id);
    }
  };

  const closeModal = () => {
    setSelectedPatient(null);
    setTriageResult(null);
    setVitalsHistory([]);
  };

  // Handoff Mode functions
  const loadHandoffReport = useCallback(async () => {
    setHandoffLoading(true);
    try {
      const report = await fetchHandoffReport();
      setHandoffReport(report);
    } catch (err) {
      console.error("Failed to fetch handoff report", err);
    } finally {
      setHandoffLoading(false);
    }
  }, []);

  const toggleHandoffMode = () => {
    const newMode = !handoffMode;
    setHandoffMode(newMode);
    if (newMode) {
      loadHandoffReport();
    }
  };

  return (
    <div className={`p-6 min-h-screen transition-colors duration-300 ${darkMode ? "bg-slate-900" : "bg-transparent"}`}>
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <div>
          <h1 className={`text-3xl font-bold ${darkMode ? "text-white" : "text-gray-900"}`}>
            🏥 Triage Command Center
          </h1>
          <p className={darkMode ? "text-slate-400" : "text-gray-500"}>Real-time Emergency Department Monitor</p>
        </div>
        <div className="flex items-center gap-4">
          {/* Handoff Mode Toggle */}
          <button
            onClick={toggleHandoffMode}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              handoffMode
                ? "bg-purple-600 text-white shadow-lg"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            <ClipboardList className="w-5 h-5" />
            Handoff Mode
          </button>
          <div className="flex items-center gap-2">
            <span
              className={`w-3 h-3 rounded-full ${
                connected ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-sm text-gray-600">
              {connected ? "Live" : "Disconnected"}
            </span>
          </div>
        </div>
      </header>

      {/* Handoff Summary Banner */}
      <AnimatePresence>
        {handoffMode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6"
          >
            {handoffLoading ? (
              <div className="bg-purple-50 border border-purple-200 rounded-xl p-6 flex items-center justify-center">
                <RefreshCw className="w-5 h-5 animate-spin text-purple-600 mr-2" />
                <span className="text-purple-700">Generating handoff report...</span>
              </div>
            ) : handoffReport ? (
              <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-xl overflow-hidden">
                {/* Stats Row */}
                <div className="grid grid-cols-4 gap-4 p-4 border-b border-purple-100">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-red-100 rounded-lg">
                      <AlertTriangle className="w-5 h-5 text-red-600" />
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-red-600">{handoffReport.critical_count}</div>
                      <div className="text-xs text-gray-500">Critical (ESI 1-2)</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-yellow-100 rounded-lg">
                      <AlertCircle className="w-5 h-5 text-yellow-600" />
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-yellow-600">{handoffReport.urgent_count}</div>
                      <div className="text-xs text-gray-500">Urgent (ESI 3)</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Users className="w-5 h-5 text-green-600" />
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-green-600">{handoffReport.stable_count}</div>
                      <div className="text-xs text-gray-500">Stable (ESI 4-5)</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Clock className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-blue-600">
                        {handoffReport.avg_wait_minutes?.toFixed(0) || "—"}
                      </div>
                      <div className="text-xs text-gray-500">Avg Wait (min)</div>
                    </div>
                  </div>
                </div>

                {/* AI Summary */}
                <div className="p-4 bg-white/50">
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-purple-100 rounded-lg shrink-0">
                      <ClipboardList className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-purple-900 mb-1">AI Handoff Summary</h3>
                      <p className="text-gray-700">{handoffReport.summary}</p>
                    </div>
                  </div>
                </div>

                {/* Alerts */}
                {handoffReport.alerts.length > 0 && (
                  <div className="p-4 bg-red-50/50 border-t border-purple-100">
                    <h4 className="text-sm font-medium text-red-800 mb-2 flex items-center gap-1">
                      <AlertTriangle className="w-4 h-4" />
                      Critical Alerts
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {handoffReport.alerts.map((alert, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-1 bg-red-100 text-red-800 text-sm rounded-lg"
                        >
                          {alert}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Refresh Button */}
                <div className="p-3 bg-purple-100/50 flex justify-between items-center">
                  <span className="text-xs text-purple-600">
                    Generated: {new Date(handoffReport.generated_at).toLocaleTimeString()}
                  </span>
                  <button
                    onClick={loadHandoffReport}
                    className="flex items-center gap-1 text-sm text-purple-700 hover:text-purple-900"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Refresh
                  </button>
                </div>
              </div>
            ) : null}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Patient Queue */}
      <div className={`rounded-xl shadow-lg overflow-hidden transition-colors duration-300 ${darkMode ? "bg-slate-800" : "bg-white"}`}>
        <div className={`px-6 py-4 border-b ${darkMode ? "border-slate-700 bg-slate-800/50" : "border-gray-200 bg-gray-50"}`}>
          <h2 className={`text-lg font-semibold ${darkMode ? "text-slate-200" : "text-gray-700"}`}>
            {handoffMode ? "Watch List" : "Waiting Patients"} ({handoffMode && handoffReport ? handoffReport.watch_list.length : patients.length})
          </h2>
        </div>

        <div className={`divide-y ${darkMode ? "divide-slate-700" : "divide-gray-100"}`}>
          <AnimatePresence>
            {patients.map((patient) => (
              <motion.div
                key={patient.appointment_id}
                layout
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.3 }}
                className={`px-6 py-4 cursor-pointer flex items-center gap-4 transition-colors ${darkMode ? "hover:bg-slate-700/50" : "hover:bg-gray-50"}`}
                onClick={() => openModal(patient)}
              >
                {/* Token Badge */}
                <div
                  className={`${
                    TRIAGE_COLORS[patient.triage_level]
                  } text-white px-3 py-2 rounded-lg font-mono font-bold min-w-[100px] text-center`}
                >
                  {patient.token}
                </div>

                {/* Triage Level */}
                <div className="flex-1">
                  <div className={`font-medium ${darkMode ? "text-slate-100" : "text-gray-900"}`}>
                    {TRIAGE_LABELS[patient.triage_level]}
                  </div>
                  <div className={`text-sm truncate max-w-xs ${darkMode ? "text-slate-400" : "text-gray-500"}`}>
                    {patient.chief_complaint}
                  </div>
                </div>

                {/* Vitals */}
                <div className={`flex items-center gap-4 text-sm ${darkMode ? "text-slate-300" : "text-gray-600"}`}>
                  {patient.vitals.HR && (
                    <div className="flex items-center gap-1">
                      <Heart className="w-4 h-4 text-red-400" />
                      {patient.vitals.HR}
                    </div>
                  )}
                  {patient.vitals.SBP && (
                    <div className="flex items-center gap-1">
                      <Activity className="w-4 h-4 text-blue-400" />
                      {patient.vitals.SBP}
                    </div>
                  )}
                  {patient.vitals.RR && (
                    <div className="flex items-center gap-1">
                      <Wind className="w-4 h-4 text-green-400" />
                      {patient.vitals.RR}
                    </div>
                  )}
                  {patient.vitals.O2Sat && (
                    <div className="flex items-center gap-1">
                      <Droplets className="w-4 h-4 text-cyan-400" />
                      {patient.vitals.O2Sat}%
                    </div>
                  )}
                </div>

                {/* Action & SHAP Indicator */}
                <div className="flex items-center gap-2">
                  {patient.action && (
                    <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                      {patient.action}
                    </span>
                  )}
                  {patient.shap_ready && (
                    <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                      AI Explained
                    </span>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {patients.length === 0 && (
            <div className="p-6">
              {/* Status Cards */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-4 border border-green-100">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <CheckCircle2 className="w-5 h-5 text-green-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-green-700">Ready</p>
                      <p className="text-xs text-green-600">System Online</p>
                    </div>
                  </div>
                </div>
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Users className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-blue-700">0</p>
                      <p className="text-xs text-blue-600">Patients In Queue</p>
                    </div>
                  </div>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-violet-50 rounded-xl p-4 border border-purple-100">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-100 rounded-lg">
                      <Cpu className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-purple-700">Active</p>
                      <p className="text-xs text-purple-600">AI Models Loaded</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Call to Action */}
              <div className="text-center py-10 bg-gradient-to-br from-gray-50 to-slate-50 rounded-xl border-2 border-dashed border-gray-200">
                <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center shadow-lg">
                  <Stethoscope className="w-8 h-8 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-700 mb-2">
                  Emergency Department Queue is Clear
                </h3>
                <p className="text-gray-500 mb-6 max-w-md mx-auto">
                  The system is ready and monitoring. New patients will appear here automatically via real-time stream.
                </p>
                <a
                  href="/patient/triage"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl"
                >
                  <Stethoscope className="w-5 h-5" />
                  Open Patient Kiosk
                </a>
                <p className="text-xs text-gray-400 mt-4">
                  Use the Patient Kiosk to simulate a patient check-in
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Patient Detail Modal */}
      <AnimatePresence>
        {selectedPatient && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={closeModal}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className={`rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto ${darkMode ? "bg-slate-800" : "bg-white"}`}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div
                className={`${
                  TRIAGE_COLORS[selectedPatient.triage_level]
                } px-6 py-4 flex items-center justify-between`}
              >
                <div className="text-white">
                  <div className="text-2xl font-bold">
                    {selectedPatient.token}
                  </div>
                  <div className="opacity-80">
                    {TRIAGE_LABELS[selectedPatient.triage_level]}
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-white/80 hover:text-white"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Modal Body */}
              <div className="p-6 space-y-6">
                {/* Chief Complaint */}
                <div>
                  <h3 className="text-sm font-medium text-gray-500 mb-1">
                    Chief Complaint (ClinicalBERT Summary)
                  </h3>
                  <p className="text-gray-900">
                    {selectedPatient.chief_complaint}
                  </p>
                </div>

                {/* Vitals Grid */}
                <div>
                  <h3 className="text-sm font-medium text-gray-500 mb-2">
                    Vital Signs
                  </h3>
                  <div className="grid grid-cols-4 gap-3">
                    <VitalCard
                      icon={<Heart className="text-red-500" />}
                      label="Heart Rate"
                      value={selectedPatient.vitals.HR}
                      unit="bpm"
                    />
                    <VitalCard
                      icon={<Activity className="text-blue-500" />}
                      label="Blood Pressure"
                      value={selectedPatient.vitals.SBP}
                      unit="mmHg"
                    />
                    <VitalCard
                      icon={<Wind className="text-green-500" />}
                      label="Resp Rate"
                      value={selectedPatient.vitals.RR}
                      unit="/min"
                    />
                    <VitalCard
                      icon={<Droplets className="text-cyan-500" />}
                      label="O2 Sat"
                      value={selectedPatient.vitals.O2Sat}
                      unit="%"
                    />
                  </div>
                </div>

                {/* Vitals Trends (Sparklines) */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className={`w-4 h-4 ${darkMode ? "text-slate-400" : "text-gray-500"}`} />
                    <h3 className={`text-sm font-medium ${darkMode ? "text-slate-400" : "text-gray-500"}`}>
                      Vitals Trends (Last 4 Hours)
                    </h3>
                  </div>
                  {vitalsLoading ? (
                    <div className={`flex items-center justify-center py-8 rounded-lg ${darkMode ? "bg-slate-700" : "bg-gray-50"}`}>
                      <RefreshCw className="w-5 h-5 animate-spin text-gray-400 mr-2" />
                      <span className={`text-sm ${darkMode ? "text-slate-400" : "text-gray-400"}`}>Loading trends...</span>
                    </div>
                  ) : vitalsHistory.length > 0 ? (
                    <div className={`grid grid-cols-3 gap-4 p-4 rounded-lg ${darkMode ? "bg-slate-700" : "bg-gray-50"}`}>
                      <Sparkline
                        data={transformVitalsForChart(vitalsHistory, "hr")}
                        color="#ef4444"
                        threshold={100}
                        thresholdDirection="above"
                        label="Heart Rate"
                        unit="bpm"
                      />
                      <Sparkline
                        data={transformVitalsForChart(vitalsHistory, "sbp")}
                        color="#3b82f6"
                        threshold={90}
                        thresholdDirection="below"
                        label="Blood Pressure"
                        unit="mmHg"
                      />
                      <Sparkline
                        data={transformVitalsForChart(vitalsHistory, "o2_sat")}
                        color="#10b981"
                        threshold={95}
                        thresholdDirection="below"
                        label="O2 Saturation"
                        unit="%"
                      />
                    </div>
                  ) : (
                    <div className={`flex items-center justify-center py-6 rounded-lg ${darkMode ? "bg-slate-700" : "bg-gray-50"}`}>
                      <span className={`text-sm ${darkMode ? "text-slate-400" : "text-gray-400"}`}>
                        No historical data available yet
                      </span>
                    </div>
                  )}
                </div>

                {/* RL Recommended Action */}
                {selectedPatient.action && (
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-1">
                      <AlertCircle className="w-5 h-5 text-purple-600" />
                      <h3 className="font-medium text-purple-900">
                        System Recommendation
                      </h3>
                    </div>
                    <p className="text-purple-700 text-lg font-semibold">
                      {selectedPatient.action}
                    </p>
                  </div>
                )}

                {/* Clinical Insights */}
                {triageResult?.shock_index && triageResult.shock_index > 0.7 && (
                  <div className={`border rounded-lg p-4 ${
                    triageResult.shock_index > 1.0 
                      ? 'bg-red-50 border-red-300' 
                      : 'bg-orange-50 border-orange-200'
                  }`}>
                    <div className="flex items-center gap-2 mb-1">
                      <AlertCircle className={`w-5 h-5 ${
                        triageResult.shock_index > 1.0 
                          ? 'text-red-600' 
                          : 'text-orange-600'
                      }`} />
                      <h3 className={`font-medium ${
                        triageResult.shock_index > 1.0 
                          ? 'text-red-900' 
                          : 'text-orange-900'
                      }`}>
                        Clinical Alert: Elevated Shock Index
                      </h3>
                    </div>
                    <p className={`text-sm ${
                      triageResult.shock_index > 1.0 
                        ? 'text-red-700' 
                        : 'text-orange-700'
                    }`}>
                      SI = {triageResult.shock_index.toFixed(2)} (HR/SBP) — 
                      {triageResult.shock_index > 1.0 
                        ? ' Critical: Consider immediate intervention' 
                        : ' Borderline: Monitor closely'}
                    </p>
                  </div>
                )}

                {/* SHAP Explanation */}
                <div>
                  <h3 className="text-sm font-medium text-gray-500 mb-2">
                    AI Reasoning (Top SHAP Features)
                  </h3>
                  {loading ? (
                    <div className="text-center py-4 text-gray-400">
                      <RefreshCw className="w-5 h-5 mx-auto animate-spin mb-2" />
                      Loading AI explanation...
                    </div>
                  ) : selectedPatient.top_features?.length ? (
                    <ShapChart features={selectedPatient.top_features} />
                  ) : triageResult?.top_features?.length ? (
                    <ShapChart features={triageResult.top_features} />
                  ) : selectedPatient.shap_ready ? (
                    <div className="bg-yellow-50 text-yellow-700 p-3 rounded-lg text-sm">
                      SHAP computation complete. Feature analysis being processed.
                    </div>
                  ) : (
                    <div className="bg-gray-50 text-gray-400 p-3 rounded-lg text-sm">
                      AI explanation will appear once computed (~2-5 seconds)
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function VitalCard({
  icon,
  label,
  value,
  unit,
}: {
  icon: React.ReactNode;
  label: string;
  value?: number;
  unit: string;
}) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      <div className="flex justify-center mb-1">{icon}</div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-lg font-semibold text-gray-900">
        {value ?? "—"} <span className="text-xs font-normal">{unit}</span>
      </div>
    </div>
  );
}
