# Ingestion Progress & Cost Estimation

## Overview

The system provides:
1. **Cost Estimation** - accurate pricing in user's preferred currency with time estimates
2. **Real-time Progress Tracking** - Server-Sent Events (SSE) for live ingestion progress bar
3. **Time Estimates** - min/avg/max predictions for indexing duration

## Cost Estimation

### Response Schema

```typescript
{
  total_chunks: number;
  chunk_token_size: number;
  estimated_llm_calls: number;
  estimated_embedding_calls: number;
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_cost_min: number;      // in specified currency
  estimated_cost_max: number;       // in specified currency
  currency: "USD" | "EUR" | ...;   // ISO 4217 code
  model_name: string;
  embedding_model_name: string;
  estimated_indexing_time_min: number;  // seconds (optimistic: 50 tok/sec)
  estimated_indexing_time_max: number;  // seconds (pessimistic: 10 tok/sec)
  estimated_indexing_time_avg: number;  // seconds (average)
  warnings: string[];                   // per-role model warnings
}
```

### Example

Request:
```bash
POST /internal/apps/1/silos/5/estimate-indexing
{
  "documents": [
    {"content": "Lorem ipsum... (4800 chars)"},
    {"content": "Another doc... (3600 chars)"}
  ]
}
```

Response:
```json
{
  "total_chunks": 2,
  "chunk_token_size": 1200,
  "estimated_llm_calls": 4,
  "estimated_embedding_calls": 2,
  "estimated_input_tokens": 4800,
  "estimated_output_tokens": 500,
  "estimated_cost_min": 0.0015,
  "estimated_cost_max": 0.0045,
  "currency": "USD",
  "model_name": "gpt-4o-mini",
  "embedding_model_name": "text-embedding-3-small",
  "estimated_indexing_time_min": 15.2,
  "estimated_indexing_time_max": 65.8,
  "estimated_indexing_time_avg": 40.5,
  "warnings": []
}
```

### Time Calculation

Based on benchmark token throughput:

```
LLM throughput:
  - Optimistic: 50 tokens/second
  - Pessimistic: 10 tokens/second
  - (varies by model complexity, temperature, output size)

Embedding throughput:
  - Optimistic: 500 tokens/second
  - Pessimistic: 100 tokens/second
  - (more consistent than LLM)

Overhead: 20% for network + processing

time_min = (llm_input_tokens / 50 + emb_tokens / 500) × 1.2
time_max = (llm_input_tokens / 10 + emb_tokens / 100) × 1.2
time_avg = (time_min + time_max) / 2
```

## Real-time Progress Tracking

### Backend: Track Progress

In your indexing service:

```python
from services.ingestion_progress_tracker import IngestionProgressManager
import uuid

# Create session
session_id = str(uuid.uuid4())
await IngestionProgressManager.create_session(
    session_id=session_id,
    silo_id=5,
    total_chunks=100,
    estimated_total_time=60.5  # from cost estimation
)

# Process chunks
for i, chunk in enumerate(chunks):
    try:
        # ... index chunk ...
        await IngestionProgressManager.update_progress(
            session_id=session_id,
            processed=i + 1,
            failed=0,
            chunk_name=f"doc_{i}.txt"
        )
    except Exception as e:
        await IngestionProgressManager.update_progress(
            session_id=session_id,
            processed=i,
            failed=1,
            chunk_name=f"doc_{i}.txt (ERROR)"
        )

# Complete
await IngestionProgressManager.complete_session(session_id)
```

### Frontend: Consume Progress

#### React Hook

```typescript
import { useIngestionProgress } from '@/hooks/useIngestionProgress';

function IndexingPage() {
  const { progress, isConnected, isComplete, error } = useIngestionProgress(
    appId,
    siloId,
    sessionId
  );

  if (progress) {
    return (
      <div>
        <div className="text-lg font-bold">
          {progress.progress_percent.toFixed(1)}% complete
        </div>
        <div className="mt-2">
          {progress.processed_chunks} / {progress.total_chunks} chunks
        </div>
        <div className="mt-2">
          Elapsed: {Math.floor(progress.elapsed_seconds)}s
          {progress.estimated_remaining_seconds && (
            <>
              <br />
              ETA: {Math.floor(progress.estimated_remaining_seconds)}s
            </>
          )}
        </div>
        <div className="mt-2">
          Current: {progress.current_chunk_name}
        </div>
      </div>
    );
  }

  if (error) return <div>Error: {error}</div>;
  return <div>Loading...</div>;
}
```

#### Progress Bar Component

```typescript
import { IngestionProgressBar } from '@/components/ui/IngestionProgressBar';

export function RepositoryPage() {
  return (
    <IngestionProgressBar
      appId={appId}
      siloId={siloId}
      sessionId={sessionId}
      onComplete={() => {
        console.log('Ingestion done!');
        refetchRepository();
      }}
    />
  );
}
```

#### Raw SSE (JavaScript)

```javascript
const eventSource = new EventSource(
  `/internal/apps/${appId}/silos/${siloId}/ingestion-progress/${sessionId}`
);

eventSource.addEventListener('progress', (event) => {
  const progress = JSON.parse(event.data);
  updateProgressBar(progress.progress_percent);
  console.log(`${progress.processed_chunks}/${progress.total_chunks} chunks`);
  console.log(`ETA: ${Math.floor(progress.estimated_remaining_seconds)}s`);
});

eventSource.addEventListener('complete', (event) => {
  console.log('Ingestion complete!');
  eventSource.close();
});

eventSource.addEventListener('error', (event) => {
  console.error('Ingestion error:', event.data);
  eventSource.close();
});
```

## Currency Support

### Default: USD

All pricing catalogs default to USD. To support other currencies:

1. **Add to provider files** (e.g., `openai_pricing.py`):
```python
def fetch_llm_pricing(self, currency: str = "USD"):
    if currency == "EUR":
        # EUR prices (use conversion or region-specific pricing)
        return {"gpt-4o": (2.30, 9.50), ...}
    return self._LLM_PRICES  # USD
```

2. **Update cache_manager** to pass currency:
```python
await PricingCacheManager._upsert_pricing(
    db,
    model_name="gpt-4o",
    currency="EUR",
    input_price_per_1m=2.30,
    ...
)
```

3. **Frontend asks user for currency**:
```typescript
<select onChange={(e) => setCurrency(e.target.value)}>
  <option value="USD">USD ($)</option>
  <option value="EUR">EUR (€)</option>
</select>
```

4. **Include in cost estimation request**:
```bash
POST /internal/apps/1/silos/5/estimate-indexing?currency=EUR
```

## Database Schema

### pricing_catalog

```sql
CREATE TABLE pricing_catalog (
  model_name VARCHAR(255) PRIMARY KEY,
  provider VARCHAR(50) NOT NULL,
  currency VARCHAR(3) NOT NULL DEFAULT 'USD',
  input_price_per_1m FLOAT,
  output_price_per_1m FLOAT,
  embedding_price_per_1m FLOAT,
  last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  source VARCHAR(100) NOT NULL,
  INDEX idx_provider (provider),
  INDEX idx_currency (currency)
);
```

Example:
```sql
-- GPT-4o (USD)
INSERT INTO pricing_catalog VALUES (
  'gpt-4o', 'openai', 'USD',
  2.50, 10.00, NULL,
  NOW(), 'openai_api'
);

-- GPT-4o (EUR)
INSERT INTO pricing_catalog VALUES (
  'gpt-4o', 'openai', 'EUR',
  2.30, 9.50, NULL,
  NOW(), 'openai_api'
);
```

## Benchmarking Notes

The time estimates use conservative benchmarks. Actual performance depends on:

1. **LLM Model** - larger models are slower
   - GPT-4o: typically 15-30 tok/sec
   - GPT-4o-mini: typically 30-60 tok/sec
   - Claude: typically 20-40 tok/sec

2. **Temperature** - higher = slower (more output variety)

3. **Output size** - chunking strategy affects output tokens

4. **Network latency** - 20% overhead accounts for typical cloud latency

5. **Concurrent requests** - system handles parallel calls

### Measuring Actual Performance

Track real indexing sessions to refine benchmarks:

```sql
SELECT
  AVG(elapsed_seconds / processed_chunks) as avg_sec_per_chunk,
  MIN(elapsed_seconds / processed_chunks) as min_sec_per_chunk,
  MAX(elapsed_seconds / processed_chunks) as max_sec_per_chunk
FROM ingestion_sessions
WHERE model_name = 'gpt-4o'
  AND completed_at > DATE_SUB(NOW(), INTERVAL 7 DAY);
```

## API Reference

### Cost Estimation Endpoint

**POST** `/internal/apps/{app_id}/silos/{silo_id}/estimate-indexing`

Request:
```json
{
  "documents": [
    {"content": "...", "metadata": {...}}
  ]
}
```

Response: `CostEstimationResponseSchema`

### Progress Streaming Endpoint

**GET** `/internal/apps/{app_id}/silos/{silo_id}/ingestion-progress/{session_id}`

Content-Type: `text/event-stream`

Events:
- `message` or `progress`: `IngestionProgress` JSON
- `complete`: ingestion done
- `error`: ingestion failed

## Troubleshooting

### No Cost Estimates
- Check `pricing_catalog` table has records
- Verify model names match exactly
- Check logs for fetch errors

### Progress Not Updating
- Verify SSE connection: `curl -v <url>` (should return `200 OK` with streaming)
- Check session_id is passed correctly
- Ensure `IngestionProgressManager.update_progress()` is called in indexing loop

### Inaccurate Time Estimates
- Actual performance varies by model, temperature, output size
- Track real sessions to measure actual throughput
- Adjust benchmarks in `silo_service.py` based on data
