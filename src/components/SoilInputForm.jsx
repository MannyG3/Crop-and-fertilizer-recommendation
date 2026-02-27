import { useState } from "react";

const FIELD_CONFIG = [
  {
    key: "nitrogen",
    label: "Nitrogen (N)",
    step: "0.1",
    min: "0",
    tooltip:
      "Nitrogen helps leaves and stems grow strong. Low nitrogen can cause pale leaves and weak growth.",
  },
  {
    key: "phosphorus",
    label: "Phosphorus (P)",
    step: "0.1",
    min: "0",
    tooltip:
      "Phosphorus supports root development and flowering. Good phosphorus helps plants establish early.",
  },
  {
    key: "potassium",
    label: "Potassium (K)",
    step: "0.1",
    min: "0",
    tooltip:
      "Potassium improves crop strength and disease resistance. It also supports water balance in plants.",
  },
  {
    key: "temperature",
    label: "Temperature (°C)",
    step: "0.1",
    tooltip:
      "Temperature affects germination and crop growth speed. Extreme heat or cold can reduce yield.",
  },
  {
    key: "humidity",
    label: "Humidity (%)",
    step: "0.1",
    min: "0",
    max: "100",
    tooltip:
      "Humidity is moisture in the air. Very high or very low humidity can increase crop stress and disease risk.",
  },
  {
    key: "ph",
    label: "Soil pH",
    step: "0.1",
    min: "0",
    max: "14",
    tooltip:
      "pH shows how acidic or alkaline soil is. Most crops grow best when pH is in the suitable range for nutrient uptake.",
  },
  {
    key: "rainfall",
    label: "Rainfall (mm)",
    step: "0.1",
    min: "0",
    tooltip:
      "Rainfall indicates available water for crops. Too little causes drought stress; too much can cause root problems.",
  },
];

const INITIAL_VALUES = FIELD_CONFIG.reduce((acc, field) => {
  acc[field.key] = "";
  return acc;
}, {});

function LeafSpinner() {
  return (
    <svg
      className="h-5 w-5 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M19.5 4.5C14 4.5 9.8 6.2 7.2 8.8C4.5 11.5 3.5 15 4.5 19.5C9 20.5 12.5 19.5 15.2 16.8C17.8 14.2 19.5 10 19.5 4.5Z"
        className="fill-[#E0AB00]"
      />
      <path
        d="M7 17C9.5 14.5 12 12.5 16 9"
        stroke="#1E4D35"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function FieldTooltip({ text }) {
  return (
    <div className="group relative inline-flex">
      <button
        type="button"
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-[#8C6A4F] text-xs font-semibold text-[#4A3424] hover:bg-[#F9E7A8] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2F6B4A]"
        aria-label="Field information"
      >
        i
      </button>
      <div className="pointer-events-none absolute left-7 top-1/2 z-20 w-64 -translate-y-1/2 rounded-md bg-[#2A2E2B] p-2 text-xs leading-relaxed text-[#F7F8F5] opacity-0 shadow-md transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100">
        {text}
      </div>
    </div>
  );
}

export default function SoilInputForm({ onSubmit }) {
  const [values, setValues] = useState(INITIAL_VALUES);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (fieldKey, fieldValue) => {
    setValues((prev) => ({ ...prev, [fieldKey]: fieldValue }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const payload = Object.entries(values).reduce((acc, [key, value]) => {
      acc[key] = Number(value);
      return acc;
    }, {});

    setIsSubmitting(true);
    try {
      await onSubmit?.(payload);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="mx-auto w-full max-w-4xl rounded-xl border border-[#C9B59F] bg-[#F7F8F5] p-6 shadow-sm md:p-8">
      <header className="mb-6">
        <h2 className="text-2xl font-semibold tracking-tight text-[#123524]">
          Soil & Climate Input
        </h2>
        <p className="mt-2 text-sm text-[#5C665F]">
          Enter your current field values to get crop and fertilizer guidance.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {FIELD_CONFIG.map((field) => (
            <label
              key={field.key}
              className="rounded-lg border border-[#C9B59F] bg-white p-4"
            >
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-[#2A2E2B]">
                  {field.label}
                </span>
                <FieldTooltip text={field.tooltip} />
              </div>
              <input
                type="number"
                inputMode="decimal"
                required
                step={field.step}
                min={field.min}
                max={field.max}
                value={values[field.key]}
                onChange={(event) => handleChange(field.key, event.target.value)}
                className="w-full rounded-md border border-[#8C6A4F] bg-[#FFFFFF] px-3 py-2 text-sm text-[#2A2E2B] outline-none transition focus:border-[#2F6B4A] focus:ring-2 focus:ring-[#6FA67E]"
                placeholder={`Enter ${field.label}`}
              />
            </label>
          ))}
        </div>

        <div className="flex items-center justify-end">
          <button
            type="submit"
            disabled={isSubmitting}
            className="inline-flex min-w-44 items-center justify-center gap-2 rounded-md bg-[#1E4D35] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[#2F6B4A] disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isSubmitting ? (
              <>
                <LeafSpinner />
                Submitting...
              </>
            ) : (
              "Submit"
            )}
          </button>
        </div>
      </form>
    </section>
  );
}
