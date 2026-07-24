import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: Number(__ENV.VUS || 4),
  duration: __ENV.DURATION || '2m',
  thresholds: {
    http_req_duration: ['p(95)<5000', 'p(99)<10000'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  if (!__ENV.API_KEY) {
    throw new Error('API_KEY environment variable is required');
  }
  const payload = JSON.stringify({
    query: 'тема урока: стили речи',
    top_k: 8,
    min_score: 0.2,
    collection: 'default',
    return_text: false,
  });

  const baseUrl = __ENV.BASE_URL || 'http://localhost:8000';
  const res = http.post(`${baseUrl}/retrieve`, payload, {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': __ENV.API_KEY,
    },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(1);
}
