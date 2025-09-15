const BASE_URL = process.env.REACT_APP_ORCH_API || "http://localhost:3000";

export async function chatQuery(question) {
  try {
    const response = await fetch(`${BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, k: 3 }),
    });
    return await response.json();
  } catch (err) {
    console.error(err);
    return { answer: "Error contacting server", context: [] };
  }
}