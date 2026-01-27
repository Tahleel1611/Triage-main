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
} from "lucide-react";
import ShapChart from "./ShapChart";

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
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [triageResult, setTriageResult] = useState<TriageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);

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
          return updated;
        });
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

  const openModal = (patient: Patient) => {
    setSelectedPatient(patient);
    if (patient.shap_ready) {
      fetchTriageResult(patient.appointment_id);
    }
  };

  const closeModal = () => {
    setSelectedPatient(null);
    setTriageResult(null);
  };

  return (
    <div className="p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            🏥 Triage Command Center
          </h1>
          <p className="text-gray-500">Real-time Emergency Department Monitor</p>
        </div>
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
      </header>

      {/* Patient Queue */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <h2 className="text-lg font-semibold text-gray-700">
            Waiting Patients ({patients.length})
          </h2>
        </div>

        <div className="divide-y divide-gray-100">
          <AnimatePresence>
            {patients.map((patient) => (
              <motion.div
                key={patient.appointment_id}
                layout
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.3 }}
                className="px-6 py-4 hover:bg-gray-50 cursor-pointer flex items-center gap-4"
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
                  <div className="font-medium text-gray-900">
                    {TRIAGE_LABELS[patient.triage_level]}
                  </div>
                  <div className="text-sm text-gray-500 truncate max-w-xs">
                    {patient.chief_complaint}
                  </div>
                </div>

                {/* Vitals */}
                <div className="flex items-center gap-4 text-sm text-gray-600">
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
            <div className="px-6 py-12 text-center text-gray-400">
              <RefreshCw className="w-8 h-8 mx-auto mb-2 animate-spin-slow" />
              Waiting for patients...
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
              className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
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
