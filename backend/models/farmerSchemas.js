const mongoose = require("mongoose");

const { Schema, model } = mongoose;

const farmerProfileSchema = new Schema(
  {
    fullName: {
      type: String,
      required: true,
      trim: true,
    },
    phone: {
      type: String,
      required: true,
      trim: true,
    },
    location: {
      village: { type: String, trim: true },
      district: { type: String, trim: true },
      state: { type: String, trim: true },
      country: { type: String, default: "India", trim: true },
      coordinates: {
        lat: Number,
        lng: Number,
      },
    },
    farmSizeAcres: {
      type: Number,
      min: 0,
    },
    preferredLanguage: {
      type: String,
      default: "en",
      trim: true,
    },
    isActive: {
      type: Boolean,
      default: true,
    },
  },
  {
    timestamps: true,
    collection: "farmer_profiles",
  }
);

const soilHistorySchema = new Schema(
  {
    farmerId: {
      type: Schema.Types.ObjectId,
      ref: "FarmerProfile",
      required: true,
      index: true,
    },
    inputMetrics: {
      nitrogen: { type: Number, required: true, min: 0 },
      phosphorus: { type: Number, required: true, min: 0 },
      potassium: { type: Number, required: true, min: 0 },
      temperature: { type: Number, required: true },
      humidity: { type: Number, required: true, min: 0, max: 100 },
      ph: { type: Number, required: true, min: 0, max: 14 },
      rainfall: { type: Number, required: true, min: 0 },
    },
    aiRecommendation: {
      recommendedCrop: {
        type: String,
        required: true,
        trim: true,
      },
      confidence: {
        type: Number,
        required: true,
        min: 0,
        max: 1,
      },
      fertilizerPlan: {
        type: String,
        trim: true,
      },
      modelVersion: {
        type: String,
        trim: true,
      },
    },
    recommendedAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
  },
  {
    timestamps: true,
    collection: "soil_history",
  }
);

soilHistorySchema.index({ farmerId: 1, recommendedAt: -1 });

soilHistorySchema.statics.getRecentActivity = function getRecentActivity(farmerId) {
  return this.find({ farmerId })
    .sort({ recommendedAt: -1 })
    .limit(5)
    .select({
      _id: 1,
      farmerId: 1,
      inputMetrics: 1,
      aiRecommendation: 1,
      recommendedAt: 1,
    })
    .lean();
};

const FarmerProfile =
  mongoose.models.FarmerProfile || model("FarmerProfile", farmerProfileSchema);
const SoilHistory =
  mongoose.models.SoilHistory || model("SoilHistory", soilHistorySchema);

module.exports = {
  FarmerProfile,
  SoilHistory,
};
