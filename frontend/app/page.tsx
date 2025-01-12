"use client"

import { useState } from 'react';
import { API_ENDPOINTS } from '@/lib/config';

export default function Home() {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<null | {
    is_phishing: boolean;
    confidence: number;
    text: string;
  }>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeMessage = async () => {
    if (!message.trim()) {
      setError('Please enter a message to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(API_ENDPOINTS.predict, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: message }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze message');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Web3 Phishing Detector</h1>
      
      <div className="space-y-4">
        <textarea
          className="w-full p-4 border rounded-lg"
          rows={6}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Enter a message to analyze..."
        />
        
        <button
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          onClick={analyzeMessage}
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze Message'}
        </button>

        {error && (
          <div className="p-4 bg-red-100 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {result && (
          <div className="p-4 bg-gray-100 rounded-lg">
            <h2 className="text-xl font-semibold mb-2">Analysis Result</h2>
            <p className="mb-2">
              This message is{' '}
              <span className={result.is_phishing ? 'text-red-600' : 'text-green-600'}>
                {result.is_phishing ? 'likely a phishing attempt' : 'likely safe'}
              </span>
            </p>
            <p className="text-sm text-gray-600">
              Confidence: {(result.confidence * 100).toFixed(2)}%
            </p>
          </div>
        )}
      </div>
    </main>
  );
}