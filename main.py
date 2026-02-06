"""
FreeFlow LLM Backend for Pulse Finance App
Provides unlimited AI-powered SMS parsing and spending insights.
"""

import os
import json
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import FreeFlow
try:
    from freeflow_llm import FreeFlowClient
except ImportError:
    print("Warning: freeflow-llm not installed. Run: pip install freeflow-llm")
    FreeFlowClient = None


# ============== PYDANTIC MODELS ==============

class SmsParseRequest(BaseModel):
    """Request model for SMS parsing."""
    sms: str
    sender: str


class SmsParseResponse(BaseModel):
    """Response model for parsed SMS transaction."""
    is_transaction: bool
    type: str  # CREDIT, DEBIT, or NONE
    amount: float
    merchant: Optional[str] = None
    confidence: float


class SpendingReportRequest(BaseModel):
    """Request model for spending explanation."""
    period: str
    currency: str
    total_spent: float
    total_income: float
    top_category: str
    categories: List[dict]
    peak_spending_day: int
    peak_amount: float


class SpendingReportResponse(BaseModel):
    """Response model for AI spending explanation."""
    explanation: str


class CategoryInferRequest(BaseModel):
    """Request model for category inference."""
    merchant: str
    amount: float
    upi_handle: Optional[str] = None
    time_hour: int
    day_of_week: str
    transaction_type: str
    sender: str


class CategoryInferResponse(BaseModel):
    """Response model for inferred category."""
    category: str


class TransactionForAnalysis(BaseModel):
    """Single transaction for subscription analysis."""
    merchant: str
    amount: float
    date: str  # ISO format date string
    category: Optional[str] = None


class SubscriptionDetectRequest(BaseModel):
    """Request model for subscription detection."""
    transactions: List[TransactionForAnalysis]


class DetectedSubscription(BaseModel):
    """A detected recurring subscription."""
    merchant_name: str
    average_amount: float
    frequency: str  # WEEKLY, MONTHLY, QUARTERLY, YEARLY
    last_payment_date: str
    next_predicted_date: str
    confidence: float
    category: str


class SubscriptionDetectResponse(BaseModel):
    """Response model for subscription detection."""
    subscriptions: List[DetectedSubscription]
    analysis_summary: str


# ============== PROMPTS ==============

SMS_PARSE_PROMPT = """
You are an intelligent financial SMS understanding engine for an Indian Android expense-tracking app.

Your responsibility is to read the raw SMS message exactly as it is and mentally reason like a human to determine whether the SMS represents a real financial transaction and, if so, whether it is a CREDIT or DEBIT and for what amount.

SMS to analyze:
Sender: {sender}
Message: {sms}

What you must identify:
• Is this SMS a real financial transaction or not
• If yes, decide whether money is coming in (CREDIT) or going out (DEBIT)
• Extract the correct transaction amount
• Extract merchant/recipient name if visible
• Ignore OTPs, ads, reminders, balance-only messages

How you should think:
Understand meaning, not just keywords.
Do not rely only on words like "credited" or "debited".
Infer money flow direction from the sentence meaning.

Money coming to the user = CREDIT
Examples: salary received, refund processed, interest credited, cash deposited, UPI received

Money going from the user = DEBIT
Examples: paid via UPI, purchase made, bill paid, ATM withdrawal, subscription charged

Multiple numbers appear in SMS:
Choose the number that represents the actual transaction.
Ignore balances, reference IDs, phone numbers, dates.

Ambiguous or unclear SMS:
If you are not at least reasonably sure, treat it as not a transaction.
Never guess amounts.

Indian-specific understanding:
• Recognize Indian banks, UPI apps, wallets, and card alerts
• Understand formats like: Rs. 1,250 / ₹500 / INR 999.00
• Understand UPI references, VPAs, and merchant hints
• Recognize salary, rent, EMI, recharge, electricity, food, shopping, travel patterns

Critical accuracy rules:
• Never misclassify OTP or promotional messages
• If balance is mentioned without a transaction → ignore
• If both credit and debit words exist → infer actual money flow
• If message talks about "scheduled", "will be", or "pending" → not completed, ignore

RESPOND WITH JSON ONLY in this exact format:
{{"is_transaction": true/false, "type": "CREDIT" or "DEBIT" or "NONE", "amount": 0.00, "merchant": "merchant name or null", "confidence": 0.0 to 1.0}}

If not a transaction, respond:
{{"is_transaction": false, "type": "NONE", "amount": 0, "merchant": null, "confidence": 0}}
"""

SPENDING_EXPLANATION_PROMPT = """
You are a friendly financial assistant. Analyze this spending summary:

Period: {period}
Total Spent: {currency} {total_spent}
Income: {currency} {total_income}
Top Category: {top_category}
Categories: {categories}
Peak Day: Day {peak_spending_day} ({peak_amount})

Task:
1. Summarize spending in 1 sentence.
2. Point out the biggest cost driver.
3. Give 1 short specific tip.
Keep it under 60 words. Be encouraging.
"""

CATEGORY_INFER_PROMPT = """
Categorize this transaction into ONE of these categories:
Food, Shopping, Transport, Entertainment, Bills, Health, Education, Travel, Groceries, Investments, Salary, Transfers, Other

Transaction:
- Amount: ₹{amount}
- Merchant: {merchant}
- UPI Handle: {upi_handle}
- Time: {time_hour}:00 ({day_of_week})
- Type: {transaction_type}
- Bank: {sender}

Rules:
- Respond with ONLY the category name, nothing else
- Use exact category names from the list above
- If uncertain, respond with "Other"
"""

SUBSCRIPTION_DETECT_PROMPT = """
You are a financial analyst AI. Analyze this transaction history to detect RECURRING payments (subscriptions, bills, EMIs, rent).

Transaction History:
{transactions}

Your Task:
1. Group transactions by merchant name (normalize similar names like "NETFLIX" and "Netflix India").
2. Identify merchants with REGULAR payment patterns (weekly, monthly, quarterly, yearly).
3. For each recurring payment, determine:
   - The frequency (WEEKLY, MONTHLY, QUARTERLY, YEARLY)
   - Average amount
   - When the next payment is likely due
   - Category (Entertainment, Bills, Food, Transport, Health, Education, Other)

Rules:
- Only include merchants with 2+ payments at regular intervals.
- Ignore one-time purchases.
- Common subscriptions: Netflix, Spotify, Prime, Hotstar, Gym, Insurance, EMI, Rent, Electricity, Mobile recharge.
- Bills pattern: Usually same amount ± 10% variation.
- Be conservative - only flag clear patterns.

Respond in JSON format:
{{
  "subscriptions": [
    {{
      "merchant_name": "string",
      "average_amount": number,
      "frequency": "MONTHLY" | "WEEKLY" | "QUARTERLY" | "YEARLY",
      "last_payment_date": "YYYY-MM-DD",
      "next_predicted_date": "YYYY-MM-DD",
      "confidence": 0.0 to 1.0,
      "category": "string"
    }}
  ],
  "analysis_summary": "Brief 1-2 sentence summary of findings"
}}
"""


# ============== FASTAPI APP ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 Pulse Finance AI Backend starting...")
    print(f"   FreeFlow available: {FreeFlowClient is not None}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Pulse Finance AI Backend",
    description="AI-powered SMS parsing and financial insights using FreeFlow LLM",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_ai_response(prompt: str) -> str:
    """Get AI response using FreeFlow client."""
    if FreeFlowClient is None:
        raise HTTPException(status_code=500, detail="FreeFlow LLM not available")
    
    try:
        with FreeFlowClient() as client:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content
    except Exception as e:
        print(f"AI Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")


def parse_json_response(response: str) -> dict:
    """Extract and parse JSON from AI response."""
    try:
        # Find JSON in response
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            json_str = response[start:end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Pulse Finance AI Backend",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check for monitoring."""
    return {"status": "ok", "freeflow_available": FreeFlowClient is not None}


@app.post("/api/parse-sms", response_model=SmsParseResponse)
async def parse_sms(request: SmsParseRequest):
    """
    Parse an SMS message using AI to extract transaction details.
    Used as fallback when regex-based parsing fails.
    """
    prompt = SMS_PARSE_PROMPT.format(sms=request.sms, sender=request.sender)
    
    try:
        response = get_ai_response(prompt)
        parsed = parse_json_response(response)
        
        if parsed:
            return SmsParseResponse(
                is_transaction=parsed.get("is_transaction", False),
                type=parsed.get("type", "NONE"),
                amount=float(parsed.get("amount", 0)),
                merchant=parsed.get("merchant"),
                confidence=float(parsed.get("confidence", 0))
            )
        
        # Default response if parsing failed
        return SmsParseResponse(
            is_transaction=False,
            type="NONE",
            amount=0,
            merchant=None,
            confidence=0
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Parse SMS error: {e}")
        return SmsParseResponse(
            is_transaction=False,
            type="NONE",
            amount=0,
            merchant=None,
            confidence=0
        )


@app.post("/api/explain-spending", response_model=SpendingReportResponse)
async def explain_spending(request: SpendingReportRequest):
    """
    Generate a natural language explanation of spending patterns.
    """
    categories_str = json.dumps(request.categories)
    
    prompt = SPENDING_EXPLANATION_PROMPT.format(
        period=request.period,
        currency=request.currency,
        total_spent=request.total_spent,
        total_income=request.total_income,
        top_category=request.top_category,
        categories=categories_str,
        peak_spending_day=request.peak_spending_day,
        peak_amount=request.peak_amount
    )
    
    try:
        explanation = get_ai_response(prompt)
        return SpendingReportResponse(explanation=explanation)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Explain spending error: {e}")
        return SpendingReportResponse(
            explanation="Unable to generate insight at this time."
        )


@app.post("/api/infer-category", response_model=CategoryInferResponse)
async def infer_category(request: CategoryInferRequest):
    """
    Infer transaction category using AI.
    Used when local categorization confidence is low.
    """
    prompt = CATEGORY_INFER_PROMPT.format(
        amount=int(request.amount),
        merchant=request.merchant,
        upi_handle=request.upi_handle or "N/A",
        time_hour=request.time_hour,
        day_of_week=request.day_of_week,
        transaction_type=request.transaction_type,
        sender=request.sender
    )
    
    try:
        category = get_ai_response(prompt).strip()
        
        # Validate category
        valid_categories = [
            "Food", "Shopping", "Transport", "Entertainment", "Bills",
            "Health", "Education", "Travel", "Groceries", "Investments",
            "Salary", "Transfers", "Other"
        ]
        
        # Try to find matching category
        for valid in valid_categories:
            if valid.lower() in category.lower():
                return CategoryInferResponse(category=valid)
        
        return CategoryInferResponse(category="Other")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Infer category error: {e}")
        return CategoryInferResponse(category="Other")


@app.post("/api/detect-subscriptions", response_model=SubscriptionDetectResponse)
async def detect_subscriptions(request: SubscriptionDetectRequest):
    """
    Analyze transaction history to detect recurring subscriptions and bills.
    Uses AI to identify patterns and predict next due dates.
    """
    # Format transactions for the AI prompt
    transactions_str = "\n".join([
        f"- {t.date}: {t.merchant} - ₹{t.amount}" + (f" ({t.category})" if t.category else "")
        for t in request.transactions
    ])
    
    prompt = SUBSCRIPTION_DETECT_PROMPT.format(transactions=transactions_str)
    
    try:
        response = get_ai_response(prompt)
        parsed = parse_json_response(response)
        
        if parsed and "subscriptions" in parsed:
            subscriptions = []
            for sub in parsed.get("subscriptions", []):
                try:
                    subscriptions.append(DetectedSubscription(
                        merchant_name=sub.get("merchant_name", "Unknown"),
                        average_amount=float(sub.get("average_amount", 0)),
                        frequency=sub.get("frequency", "MONTHLY"),
                        last_payment_date=sub.get("last_payment_date", ""),
                        next_predicted_date=sub.get("next_predicted_date", ""),
                        confidence=float(sub.get("confidence", 0.5)),
                        category=sub.get("category", "Other")
                    ))
                except (ValueError, KeyError) as e:
                    print(f"Error parsing subscription: {e}")
                    continue
            
            return SubscriptionDetectResponse(
                subscriptions=subscriptions,
                analysis_summary=parsed.get("analysis_summary", f"Found {len(subscriptions)} recurring payment(s).")
            )
        
        # Default empty response
        return SubscriptionDetectResponse(
            subscriptions=[],
            analysis_summary="No recurring payments detected."
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Detect subscriptions error: {e}")
        return SubscriptionDetectResponse(
            subscriptions=[],
            analysis_summary=f"Analysis failed: {str(e)}"
        )


# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

