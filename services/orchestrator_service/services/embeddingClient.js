import fetch from "node-fetch";

const EMBEDDING_URL = process.env.EMBEDDING_URL || "http://localhost:8080";

/**
 * Calls embedding service /search endpoint
 * Returns { success, results, error }
 */
export async function embedAndSearch(query, k = 3) {
  try {
    const resp = await fetch(`${EMBEDDING_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k }),
    });

    if (!resp.ok) {
      return { success: false, results: [], error: `Embedding service error: ${resp.status}` };
    }

    const data = await resp.json();
    return { success: true, results: data.results || [], error: null };
  } catch (err) {
    console.error("[embedAndSearch] Error:", err.message);
    return { success: false, results: [], error: err.message };
  }
}
