import React, { useState } from "react";
import { chatQuery } from "./api";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await chatQuery(question);
    setAnswer(result.answer || "");
    setContext(result.context || []);
  };

  return (
    <div style={{ margin: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h1>RAG Demo Chat</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          placeholder="Ask a question..."
          onChange={(e) => setQuestion(e.target.value)}
          style={{ width: "60%", padding: "0.5rem" }}
        />
        <button type="submit" style={{ padding: "0.5rem 1rem", marginLeft: "1rem" }}>
          Ask
        </button>
      </form>

      <div style={{ marginTop: "2rem" }}>
        <h3>Answer:</h3>
        <p>{answer}</p>
        {context.length > 0 && (
          <>
            <h4>Sources:</h4>
            <ul>
              {context.map((doc, idx) => (
                <li key={idx}>{doc.text}</li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}

export default App;