import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '1m',
  thresholds: {
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const payload = JSON.stringify({
    query: 'тема урока: стили речи',
    top_k: 8,
    min_score: 0.2,
    collection: 'default',
    return_text: false,
  });

  const res = http.post('http://localhost:8000/retrieve', payload, {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'change-me',
    },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(1);
}
