// Default to production API URL if not in development
const DEFAULT_API_URL = process.env.NODE_ENV === 'development' 
  ? 'http://localhost:3000/api'
  : 'https://web3-phishing-detector-api.vercel.app/api';

export const API_URL = process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL;

export const API_ENDPOINTS = {
  predict: `${API_URL}/predict`
};

// Add debugging
if (process.env.NODE_ENV === 'development') {
  console.log('API Configuration:', {
    NODE_ENV: process.env.NODE_ENV,
    API_URL,
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    predictEndpoint: API_ENDPOINTS.predict
  });
} 