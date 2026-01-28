"use client";

import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
  YAxis,
} from "recharts";

interface DataPoint {
  timestamp: string;
  value: number | null;
}

interface SparklineProps {
  data: DataPoint[];
  color: string;
  threshold?: number;
  thresholdDirection?: "above" | "below"; // "above" = bad if > threshold, "below" = bad if < threshold
  label: string;
  unit: string;
}

export default function Sparkline({
  data,
  color,
  threshold,
  thresholdDirection = "above",
  label,
  unit,
}: SparklineProps) {
  // Filter out null values for rendering
  const validData = data.filter((d) => d.value !== null);

  if (validData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-20 bg-gray-50 rounded-lg">
        <span className="text-xs text-gray-400">No {label} data</span>
      </div>
    );
  }

  // Get current (latest) value
  const currentValue = validData[validData.length - 1]?.value;
  const isWarning =
    threshold !== undefined &&
    currentValue !== null &&
    (thresholdDirection === "above"
      ? currentValue > threshold
      : currentValue < threshold);

  // Calculate domain with padding
  const values = validData.map((d) => d.value as number);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const padding = (maxVal - minVal) * 0.1 || 5;

  return (
    <div className="flex flex-col">
      {/* Header with label and current value */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-gray-500">{label}</span>
        <span
          className={`text-sm font-semibold ${
            isWarning ? "text-red-600" : "text-gray-900"
          }`}
        >
          {currentValue?.toFixed(0)} {unit}
        </span>
      </div>

      {/* Sparkline Chart */}
      <div className="h-16 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={validData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <YAxis domain={[minVal - padding, maxVal + padding]} hide />
            <Tooltip
              contentStyle={{
                backgroundColor: "#fff",
                border: "1px solid #e5e7eb",
                borderRadius: "6px",
                fontSize: "12px",
                padding: "4px 8px",
              }}
              formatter={(value: number) => [`${value.toFixed(1)} ${unit}`, label]}
              labelFormatter={(label) => {
                const date = new Date(label);
                return date.toLocaleString("en-US", {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                });
              }}
            />
            {threshold !== undefined && (
              <ReferenceLine
                y={threshold}
                stroke="#ef4444"
                strokeDasharray="3 3"
                strokeOpacity={0.6}
              />
            )}
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3, fill: color }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Threshold indicator */}
      {threshold !== undefined && (
        <div className="flex items-center gap-1 mt-1">
          <div
            className="w-3 h-0 border-t border-dashed border-red-400"
            style={{ borderTopWidth: "2px" }}
          />
          <span className="text-[10px] text-gray-400">
            {thresholdDirection === "above" ? ">" : "<"} {threshold} warning
          </span>
        </div>
      )}
    </div>
  );
}
