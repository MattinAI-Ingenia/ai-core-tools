# Dynamic Pricing System

## Overview

The dynamic pricing system automatically fetches and caches LLM and embedding model prices from official provider APIs. This enables accurate cost estimation for LightRAG indexing operations without maintaining hardcoded prices.

## Architecture

### Components

1. **PricingCatalog Model** (`backend/models/pricing_catalog.py`)
   - Database table storing cached pricing data
   - Fields: model_name, provider, input/output prices, embedding_price, last_updated, source

2. **Provider Fetchers** (`backend/tools/pricing/`)
   - `base.py`: Abstract PricingProvider base class
   - `openai_pricing.py`: OpenAI GPT models
   - `anthropic_pricing.py`: Claude models
   - `mistral_pricing.py`: Mistral models
   - `google_pricing.py`: Gemini models

3. **Cache Manager** (`backend/tools/pricing/cache_manager.py`)
   - Orchestrates fetching from all providers
   - Stores/updates prices in the database
   - Provides lookup methods with fallback to hardcoded defaults

4. **Pricing Service** (`backend/services/pricing_service.py`)
   - High-level API for pricing operations
   - Called by `estimate_indexing_cost` in silo_service.py

## Workflow

### Startup Initialization

1. FastAPI app starts → `main.py:lifespan()`
2. Calls `initialize_pricing_catalog()` from `startup_tasks.py`
3. `PricingService.update_pricing_catalog(db)` fetches from all providers
4. Prices cached in `pricing_catalog` table
5. If fetch fails, app continues with fallback hardcoded prices

### Periodic Updates

To keep prices synchronized with official sources, set up a daily scheduled task:

#### Option 1: Using APScheduler (Recommended)
```python
# In your scheduling service or main.py
from apscheduler.schedulers.background import BackgroundScheduler
from services.pricing_service import PricingService

scheduler = BackgroundScheduler()
scheduler.add_job(
    lambda: PricingService.update_pricing_catalog(SessionLocal()),
    'cron',
    hour=2,  # Run at 2 AM daily
    minute=0
)
scheduler.start()
```

#### Option 2: Using Manual Admin Endpoint
```bash
# POST /internal/silos/admin/pricing/refresh
# Requires OMNIADMIN role
curl -X POST http://localhost:8000/internal/silos/admin/pricing/refresh \
  -H "Authorization: Bearer <admin_token>"
```

#### Option 3: External Cron Job
```bash
#!/bin/bash
# refresh-pricing.sh
curl -X POST http://localhost:8000/internal/silos/admin/pricing/refresh \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Cost Estimation

When estimating LightRAG indexing costs:

1. `SiloService.estimate_indexing_cost()` called
2. Looks up prices via `PricingService.get_llm_pricing(db, model_name)`
3. If found in DB, uses cached price
4. If not found, falls back to hardcoded `_LLM_PRICING` dict
5. Calculates cost range (min: compact extraction, max: verbose extraction)

```python
# Cost formula:
cost_min = (llm_input_tokens × input_price 
          + llm_output_tokens_min × output_price 
          + embedding_tokens × embedding_price) / 1,000,000

cost_max = (llm_input_tokens × input_price 
          + llm_output_tokens_max × output_price 
          + embedding_tokens × embedding_price) / 1,000,000
```

## Provider Data Sources

| Provider | Source | Update Frequency |
|----------|--------|------------------|
| OpenAI | Public list prices (openai.com/pricing) | Manual (quarterly) |
| Anthropic | Public list prices (anthropic.com/pricing) | Manual (quarterly) |
| Mistral | Public list prices (mistral.ai/technology) | Manual (quarterly) |
| Google | GCP pricing page (cloud.google.com/pricing) | Manual (quarterly) |

**Note**: Prices are maintained as curated lists since providers don't expose pricing via consistent APIs. Updates require code changes or manual admin endpoint calls.

## Fallback Mechanism

If provider fetching fails (network issues, provider API changes), the system gracefully falls back:

1. Database query returns NULL
2. Hardcoded `_LLM_PRICING` and `_EMBEDDING_PRICING` dicts in `silo_service.py` used
3. Cost estimation continues with fallback prices
4. Warning logged for operations team

```
"[openai] Failed to fetch LLM pricing: HTTPError 503. 
 Falling back to hardcoded prices."
```

## Database Schema

```sql
CREATE TABLE pricing_catalog (
  model_name VARCHAR(255) PRIMARY KEY,
  provider VARCHAR(50) NOT NULL,
  input_price_usd_per_1m FLOAT,
  output_price_usd_per_1m FLOAT,
  embedding_price_usd_per_1m FLOAT,
  last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  source VARCHAR(100) NOT NULL,
  INDEX idx_provider (provider)
);
```

### Example Records

```sql
-- GPT-4o
INSERT INTO pricing_catalog VALUES 
('gpt-4o', 'openai', 2.50, 10.00, NULL, NOW(), 'openai_api');

-- Claude 3.5 Sonnet
INSERT INTO pricing_catalog VALUES 
('claude-3.5-sonnet', 'anthropic', 3.00, 15.00, NULL, NOW(), 'anthropic_api');

-- Text Embedding 3 Small
INSERT INTO pricing_catalog VALUES 
('text-embedding-3-small', 'openai', 0.02, NULL, 0.02, NOW(), 'openai_api');
```

## Configuration & Maintenance

### Adding a New Provider

1. Create `backend/tools/pricing/my_provider_pricing.py`:
```python
from .base import PricingProvider

class MyProviderPricingProvider(PricingProvider):
    @property
    def provider_name(self) -> str:
        return "my_provider"
    
    def fetch_llm_pricing(self):
        # Fetch and return dict of {model_name: (input_price, output_price)}
        pass
    
    def fetch_embedding_pricing(self):
        # Fetch and return dict of {model_name: input_price}
        pass
```

2. Add to `PricingCacheManager.PROVIDERS` in `cache_manager.py`

### Updating Hardcoded Prices

When official prices change, update:
- `backend/tools/pricing/openai_pricing.py` (or relevant provider file)
- `backend/services/silo_service.py` (_LLM_PRICING and _EMBEDDING_PRICING dicts)

## Monitoring

### Check Last Update
```python
from db.database import SessionLocal
from models.pricing_catalog import PricingCatalog

db = SessionLocal()
recent = db.query(PricingCatalog).order_by(PricingCatalog.last_updated.desc()).first()
print(f"Last price update: {recent.last_updated}")
```

### Verify Price Cache
```python
db.query(PricingCatalog).filter(
    PricingCatalog.provider == 'openai'
).all()
```

## Troubleshooting

### No Cost Estimates Appearing

1. Check if `pricing_catalog` table exists: `SELECT COUNT(*) FROM pricing_catalog;`
2. Check logs for fetch errors during startup
3. Verify hardcoded fallback prices in `silo_service.py`

### Stale Prices

1. Manually call admin endpoint: `POST /internal/silos/admin/pricing/refresh`
2. Or schedule daily updates via APScheduler
3. Monitor `pricing_catalog.last_updated` timestamp

### Missing Model Prices

1. Add model to provider file (e.g., `openai_pricing.py`)
2. Trigger refresh via admin endpoint
3. Or wait for next scheduled update
