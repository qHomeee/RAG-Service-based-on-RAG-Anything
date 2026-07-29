import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: Number(__ENV.VUS || 4),
  duration: __ENV.DURATION || '2m',
  thresholds: {
    http_req_duration: ['p(95)<5000', 'p(99)<10000'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

const queries = [
  'Какие цели преследовали реформы Танзимат в Османской империи?',
  'Что такое сложносочинённое предложение и как связаны его части?',
  'Какие смысловые отношения между частями БСП передают тире и двоеточие?',
  'Какие факторы определяют размещение металлургических предприятий?',
  'Почему раствор хлороводорода проводит электрический ток?',
];

export default function () {
  if (!__ENV.API_KEY) {
    throw new Error('API_KEY environment variable is required');
  }
  const returnText = (__ENV.RETURN_TEXT || 'true').toLowerCase() === 'true';
  const returnContext = (__ENV.RETURN_CONTEXT || 'false').toLowerCase() === 'true';
  const payload = JSON.stringify({
    query: queries[__ITER % queries.length],
    top_k: Number(__ENV.TOP_K || 5),
    min_score: 0.2,
    collection: 'default',
    return_text: returnText,
    return_context: returnContext,
  });

  const baseUrl = __ENV.BASE_URL || 'http://localhost:8000';
  const res = http.post(`${baseUrl}/retrieve`, payload, {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': __ENV.API_KEY,
    },
  });
  let body = null;
  try {
    body = res.json();
  } catch (_) {
    body = null;
  }

  check(res, {
    'status is 200': (r) => r.status === 200,
    'hits are returned': () => Array.isArray(body?.hits) && body.hits.length > 0,
    'full fragments are returned': () =>
      !returnText || body?.hits?.every((hit) => typeof hit.text === 'string' && hit.text.length > 0),
    'expanded context is returned when requested': () =>
      !returnContext ||
      body?.hits?.every(
        (hit) =>
          typeof hit.context_text === 'string' &&
          hit.context_text.length > 0 &&
          Array.isArray(hit.context_fragments) &&
          hit.context_fragments.length > 0,
      ),
  });

  sleep(1);
}
