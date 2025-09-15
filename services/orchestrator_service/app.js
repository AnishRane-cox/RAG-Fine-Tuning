// app.js
import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import chatController from "./controllers/chatController.js";

dotenv.config();

const app = express();

// Middleware
app.use(cors()); // allow cross-origin requests if needed
app.use(express.json());

// Logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`, req.body);
  next();
});

// Routes
app.post("/chat", chatController);

// Health check
app.get("/health", (req, res) => res.json({ status: "ok" }));

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Orchestrator service running on port ${PORT}`);
});