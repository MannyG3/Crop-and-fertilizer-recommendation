require("dotenv").config();

const express = require("express");
const mongoose = require("mongoose");

const { SoilHistory } = require("./models/farmerSchemas");

const app = express();

app.use(express.json());

const PORT = process.env.PORT || 4000;
const MONGODB_URI = process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/crop_app";

app.get("/", (req, res) => {
  res.status(200).json({
    message: "Crop & Fertilizer backend is running",
    routes: {
      health: "/health",
      recentActivity: "/farmers/:farmerId/recent-activity",
    },
  });
});

async function connectToDatabase() {
  try {
    await mongoose.connect(MONGODB_URI);
    console.log("MongoDB connected");
  } catch (error) {
    console.error("MongoDB connection failed:", error.message);
    process.exit(1);
  }
}

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

app.get("/farmers/:farmerId/recent-activity", async (req, res) => {
  const { farmerId } = req.params;

  if (!mongoose.Types.ObjectId.isValid(farmerId)) {
    return res.status(400).json({ message: "Invalid farmerId" });
  }

  try {
    const history = await SoilHistory.getRecentActivity(farmerId);

    const chartData = history.map((record) => ({
      timestamp: record.recommendedAt,
      recommendedCrop: record.aiRecommendation?.recommendedCrop,
      confidence: record.aiRecommendation?.confidence,
      nitrogen: record.inputMetrics?.nitrogen,
      phosphorus: record.inputMetrics?.phosphorus,
      potassium: record.inputMetrics?.potassium,
      ph: record.inputMetrics?.ph,
      rainfall: record.inputMetrics?.rainfall,
    }));

    return res.status(200).json({
      farmerId,
      total: chartData.length,
      recentActivity: chartData,
    });
  } catch (error) {
    return res.status(500).json({
      message: "Failed to fetch recent activity",
      error: error.message,
    });
  }
});

connectToDatabase().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});
