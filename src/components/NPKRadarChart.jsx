import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Radar } from "react-chartjs-2";

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

export default function NPKRadarChart({
  recommendedCrop,
  currentLevels,
  idealLevels,
}) {
  const chartData = {
    labels: ["Nitrogen", "Phosphorus", "Potassium"],
    datasets: [
      {
        label: "Current Soil Levels",
        data: [currentLevels.nitrogen, currentLevels.phosphorus, currentLevels.potassium],
        backgroundColor: "rgba(47, 107, 74, 0.35)",
        borderColor: "#1E4D35",
        borderWidth: 2,
        pointBackgroundColor: "#1E4D35",
        pointBorderColor: "#FFFFFF",
        pointHoverBackgroundColor: "#FFFFFF",
        pointHoverBorderColor: "#1E4D35",
        pointRadius: 4,
      },
      {
        label: `Ideal Levels (${recommendedCrop})`,
        data: [idealLevels.nitrogen, idealLevels.phosphorus, idealLevels.potassium],
        backgroundColor: "rgba(0, 0, 0, 0)",
        borderColor: "#8C6A4F",
        borderDash: [6, 4],
        borderWidth: 2,
        pointBackgroundColor: "#8C6A4F",
        pointRadius: 3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#2A2E2B",
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => `${context.dataset.label}: ${context.formattedValue}`,
        },
      },
    },
    scales: {
      r: {
        suggestedMin: 0,
        angleLines: {
          color: "rgba(74, 52, 36, 0.25)",
        },
        grid: {
          color: "rgba(74, 52, 36, 0.2)",
        },
        pointLabels: {
          color: "#2A2E2B",
          font: {
            size: 13,
            weight: "600",
          },
        },
        ticks: {
          backdropColor: "transparent",
          color: "#5C665F",
          stepSize: 20,
        },
      },
    },
  };

  return (
    <section className="rounded-xl border border-[#C9B59F] bg-[#F7F8F5] p-5 shadow-sm">
      <h3 className="mb-1 text-lg font-semibold text-[#123524]">N-P-K Comparison</h3>
      <p className="mb-4 text-sm text-[#5C665F]">
        Comparing your soil values with the ideal nutrient profile for {recommendedCrop}.
      </p>

      <div className="h-80">
        <Radar data={chartData} options={chartOptions} />
      </div>
    </section>
  );
}
