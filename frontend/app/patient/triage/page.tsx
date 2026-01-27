"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Heart, Thermometer, Clock, HelpCircle, Loader2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TRIAGE_COLORS: Record<number, { bg: string; text: string; label: string }> = {
  1: { bg: "bg-red-500", text: "text-white", label: "Immediate" },
  2: { bg: "bg-orange-500", text: "text-white", label: "Urgent" },
  3: { bg: "bg-yellow-400", text: "text-gray-900", label: "Moderate" },
  4: { bg: "bg-green-500", text: "text-white", label: "Less Urgent" },
  5: { bg: "bg-blue-500", text: "text-white", label: "Non-Urgent" },
};

const WAIT_ESTIMATES: Record<number, string> = {
  1: "You will be seen immediately",
  2: "Estimated wait: 10-15 minutes",
  3: "Estimated wait: 30-60 minutes",
  4: "Estimated wait: 1-2 hours",
  5: "Estimated wait: 2-4 hours",
};

interface TriageResult {
  token_number: string;
  triage_level: number;
  estimated_wait_minutes: number | null;
  action: string | null;
}

export default function PatientTriagePage() {
  const [step, setStep] = useState<"form" | "loading" | "result">("form");
  const [result, setResult] = useState<TriageResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [symptoms, setSymptoms] = useState("");
  const [age, setAge] = useState<number | "">("");
  const [sex, setSex] = useState<"male" | "female" | "other">("male");
  const [heartRate, setHeartRate] = useState<number | "">("");
  const [temperature, setTemperature] = useState<number | "">(98.6);
  const [painLevel, setPainLevel] = useState(5);

  const [showHRHelp, setShowHRHelp] = useState(false);
  const [showTempHelp, setShowTempHelp] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStep("loading");
    setError(null);

    try {
      // For demo purposes, we'll use a mock token
      // In production, you'd get this from the auth flow
      const response = await fetch(`${API_BASE}/triage/assess`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // In production: "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({
          chief_complaint: symptoms,
          arrival_mode: "Walk-in",
          static_vitals: {
            Age: age || 30,
            Pulse: heartRate || 80,
            HR: heartRate || 80,
            Temp: temperature || 98.6,
            Resp: 16,
            SBP: 120,
            DBP: 80,
            O2Sat: 98,
            PainScale: painLevel,
            patient_id: 1, // Demo patient
          },
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to submit triage");
      }

      const data = await response.json();
      setResult(data);
      setStep("result");
    } catch (err: any) {
      setError(err.message || "Something went wrong");
      setStep("form");
    }
  };

  const resetForm = () => {
    setStep("form");
    setResult(null);
    setSymptoms("");
    setAge("");
    setHeartRate("");
    setTemperature(98.6);
    setPainLevel(5);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-2xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-blue-900">🏥 Emergency Symptom Checker</h1>
          <p className="text-gray-500 text-sm">Tell us how you're feeling</p>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-8">
        <AnimatePresence mode="wait">
          {step === "form" && (
            <motion.form
              key="form"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              onSubmit={handleSubmit}
              className="space-y-6"
            >
              {/* Main Symptom Input */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <label className="block text-lg font-medium text-gray-900 mb-3">
                  How are you feeling today?
                </label>
                <textarea
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  placeholder="Describe your symptoms in your own words... (e.g., 'I have a bad headache and feel dizzy')"
                  className="w-full h-32 p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all text-lg resize-none"
                  required
                />
              </div>

              {/* Basic Info */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Basic Information</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Age</label>
                    <input
                      type="number"
                      value={age}
                      onChange={(e) => setAge(e.target.value ? parseInt(e.target.value) : "")}
                      placeholder="Enter age"
                      className="w-full p-3 border rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-200"
                      min={0}
                      max={120}
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Sex</label>
                    <select
                      value={sex}
                      onChange={(e) => setSex(e.target.value as any)}
                      className="w-full p-3 border rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-200"
                    >
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Vitals */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Vital Signs (Optional)</h2>

                {/* Heart Rate */}
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-1">
                    <Heart className="w-4 h-4 text-red-500" />
                    <label className="text-sm text-gray-600">Heart Rate (bpm)</label>
                    <button
                      type="button"
                      onClick={() => setShowHRHelp(!showHRHelp)}
                      className="text-blue-500 hover:text-blue-700"
                    >
                      <HelpCircle className="w-4 h-4" />
                    </button>
                  </div>
                  {showHRHelp && (
                    <div className="bg-blue-50 text-blue-800 text-sm p-3 rounded-lg mb-2">
                      <strong>How to measure:</strong> Place two fingers on your wrist or neck.
                      Count the beats for 30 seconds and multiply by 2. Normal range: 60-100 bpm.
                    </div>
                  )}
                  <input
                    type="number"
                    value={heartRate}
                    onChange={(e) => setHeartRate(e.target.value ? parseInt(e.target.value) : "")}
                    placeholder="e.g., 80"
                    className="w-full p-3 border rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-200"
                    min={30}
                    max={250}
                  />
                </div>

                {/* Temperature */}
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-1">
                    <Thermometer className="w-4 h-4 text-orange-500" />
                    <label className="text-sm text-gray-600">Temperature (°F)</label>
                    <button
                      type="button"
                      onClick={() => setShowTempHelp(!showTempHelp)}
                      className="text-blue-500 hover:text-blue-700"
                    >
                      <HelpCircle className="w-4 h-4" />
                    </button>
                  </div>
                  {showTempHelp && (
                    <div className="bg-blue-50 text-blue-800 text-sm p-3 rounded-lg mb-2">
                      <strong>How to measure:</strong> Use a digital thermometer under your tongue
                      for 30-60 seconds. Normal: 97.8-99.1°F. Fever: above 100.4°F.
                    </div>
                  )}
                  <input
                    type="number"
                    value={temperature}
                    onChange={(e) => setTemperature(e.target.value ? parseFloat(e.target.value) : "")}
                    placeholder="e.g., 98.6"
                    className="w-full p-3 border rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-200"
                    min={90}
                    max={110}
                    step={0.1}
                  />
                </div>

                {/* Pain Level */}
                <div>
                  <label className="block text-sm text-gray-600 mb-2">
                    Pain Level: <strong>{painLevel}</strong> / 10
                  </label>
                  <input
                    type="range"
                    value={painLevel}
                    onChange={(e) => setPainLevel(parseInt(e.target.value))}
                    min={0}
                    max={10}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>No Pain</span>
                    <span>Worst Pain</span>
                  </div>
                </div>
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg">
                  {error}
                </div>
              )}

              <button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 px-6 rounded-xl shadow-lg transition-all text-lg"
              >
                Check My Symptoms
              </button>

              <p className="text-center text-sm text-gray-500">
                ⚠️ This is not a substitute for professional medical advice.
                If you're experiencing a life-threatening emergency, call 911 immediately.
              </p>
            </motion.form>
          )}

          {step === "loading" && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="bg-white rounded-2xl shadow-lg p-12 text-center"
            >
              <Loader2 className="w-16 h-16 text-blue-500 animate-spin mx-auto mb-4" />
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                Analyzing your symptoms...
              </h2>
              <p className="text-gray-500">Our AI is reviewing your information</p>
            </motion.div>
          )}

          {step === "result" && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              {/* Token Card */}
              <div
                className={`${TRIAGE_COLORS[result.triage_level]?.bg || "bg-gray-500"} rounded-2xl shadow-2xl p-8 text-center`}
              >
                <p className={`text-sm uppercase tracking-wider mb-2 ${TRIAGE_COLORS[result.triage_level]?.text || "text-white"} opacity-80`}>
                  Your Priority Token
                </p>
                <h2 className={`text-5xl font-bold font-mono mb-4 ${TRIAGE_COLORS[result.triage_level]?.text || "text-white"}`}>
                  {result.token_number}
                </h2>
                <p className={`text-lg ${TRIAGE_COLORS[result.triage_level]?.text || "text-white"}`}>
                  Priority: {TRIAGE_COLORS[result.triage_level]?.label || "Unknown"}
                </p>
              </div>

              {/* Wait Time */}
              <div className="bg-white rounded-2xl shadow-lg p-6 text-center">
                <Clock className="w-8 h-8 text-gray-400 mx-auto mb-3" />
                <p className="text-lg text-gray-900 font-medium">
                  {result.estimated_wait_minutes
                    ? `Estimated wait: ${result.estimated_wait_minutes} minutes`
                    : WAIT_ESTIMATES[result.triage_level] || "Please wait for your turn"}
                </p>
                <p className="text-gray-500 mt-2">
                  Please proceed to the waiting area and listen for your token number.
                </p>
              </div>

              {/* Instructions */}
              <div className="bg-blue-50 rounded-2xl p-6">
                <h3 className="font-semibold text-blue-900 mb-3">What happens next?</h3>
                <ul className="space-y-2 text-blue-800 text-sm">
                  <li>✓ Your symptoms have been recorded</li>
                  <li>✓ A nurse will call your token when ready</li>
                  <li>✓ Keep this screen open to see your token</li>
                  <li>✓ If your condition worsens, notify staff immediately</li>
                </ul>
              </div>

              <button
                onClick={resetForm}
                className="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-3 px-6 rounded-xl transition-all"
              >
                Submit Another Entry
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
