import fetch from "node-fetch";

const MODEL_URL = process.env.MODEL_URL || "http://localhost:5000/generate";

/**
 * Calls LoRA model inference API (Python)
 * Returns { success, answer, error }
 */
export async function generateAnswer(prompt) {
  try {
    const resp = await fetch(MODEL_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!resp.ok) {
      return { success: false, answer: "", error: `Model service error: ${resp.status}` };
    }

    const data = await resp.json();
    return { success: true, answer: data.answer || "", error: null };
  } catch (err) {
    console.error("[generateAnswer] Error:", err.message);
    return { success: false, answer: "", error: err.message };
  }
}
