// controllers/chatController.js
import { embedAndSearch } from "../services/embeddingClient.js";
import { generateAnswer } from "../services/modelClient.js";

export default async function chatController(req, res) {
  try {
    const { query, k = 3 } = req.body;
    if (!query) {
      return res.status(400).json({ error: "Missing query" });
    }

    // Step 1: Retrieve top-k documents via embedding service
    const { success: embedSuccess, results, error: embedError } = await embedAndSearch(query, k);
    if (!embedSuccess) {
      return res.status(500).json({ error: `Embedding service failed: ${embedError}` });
    }

    // Step 2: Build augmented prompt
    const contextText = results.map(d => `- ${d.text}`).join("\n");
    const prompt = `You are a helpful assistant. Use the context to answer.\n\nContext:\n${contextText}\n\nQ: ${query}\nA:`;

    // Step 3: Call LoRA model for generation
    const { success: modelSuccess, answer, error: modelError } = await generateAnswer(prompt);
    if (!modelSuccess) {
      return res.status(500).json({ error: `Model service failed: ${modelError}` });
    }

    // Step 4: Return response
    return res.json({
      answer,
      context: results,
    });
  } catch (err) {
    console.error("[ChatController] Error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
}