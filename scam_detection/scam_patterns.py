"""
Scam Detection Taxonomies, Keywords, and Regex Patterns for SAFE CALL.

Defines categorized scam indicators:
1. Urgency & Coercion
2. Financial & Credential Requests (OTP, PIN, Bank details)
3. Authority Impersonation (Bank, Police, Government, Courier, IT)
4. Threats & Legal Intimidation
5. False Rewards, Lottery, & Refund Lures
"""

import re
from typing import Dict, List, Any

SCAM_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "financial_credential": {
        "weight": 0.35,
        "severity": "CRITICAL",
        "description": "Requests for confidential banking, credentials, or fund transfers",
        "patterns": [
            r"\b(one[\s\-]?time[\s\-]?password|otp)\b",
            r"\b(credit|debit)\s+card\s+(number|details|expiry|cvv)\b",
            r"\b(cvv|pin|passcode|secret\s+code)\b",
            r"\b(bank\s+account|account\s+number|routing\s+number|ifsc)\b",
            r"\b(transfer\s+(the\s+)?money|send\s+(the\s+)?amount|wire\s+transfer)\b",
            r"\b(upi\s+pin|google\s+pay|phonepe|paytm)\b",
            r"\b(gift[\s\-]?card|crypto|bitcoin|apple\s+card)\b",
            r"\b(share\s+(your\s+)?screen|install\s+(anydesk|teamviewer|quicksupport))\b",
        ],
    },
    "urgency": {
        "weight": 0.20,
        "severity": "HIGH",
        "description": "Artificial time pressure forcing immediate rash decisions",
        "patterns": [
            r"\b(immediately|right\s+now|urgent|urgently|at\s+once)\b",
            r"\b(within\s+\d+\s+(minutes?|hours?))\b",
            r"\b(act\s+fast|don'?t\s+delay|limited\s+time)\b",
            r"\b(don'?t\s+hang\s+up|stay\s+on\s+the\s+line)\b",
            r"\b(before\s+it'?s\s+too\s+late|final\s+notice|last\s+chance)\b",
            r"\b(immediately\s+block|terminate\s+immediately)\b",
        ],
    },
    "impersonation": {
        "weight": 0.20,
        "severity": "HIGH",
        "description": "Posing as a bank, law enforcement, tax official, or trusted service",
        "patterns": [
            r"\b(calling\s+from\s+(your\s+)?bank|bank\s+manager|fraud\s+department)\b",
            r"\b(reserve\s+bank|rbi|state\s+bank|sbi|hdfc|icici|axis)\b",
            r"\b(police\s+officer|police\s+department|crime\s+branch|cbi|fbi|interpol)\b",
            r"\b(income\s+tax|irs|tax\s+department|customs\s+department)\b",
            r"\b(courier|fedex|dhl|customs\s+officer|illegal\s+package)\b",
            r"\b(amazon\s+support|microsoft\s+technical\s+support|apple\s+support)\b",
            r"\b(telecom\s+department|sim\s+card\s+block|trai)\b",
        ],
    },
    "threats_legal": {
        "weight": 0.15,
        "severity": "CRITICAL",
        "description": "Threatening legal penalties, arrest, disconnection, or asset freezing",
        "patterns": [
            r"\b(arrest\s+warrant|police\s+arrest|jail|custody)\b",
            r"\b(legal\s+action|court\s+order|fir|lawsuit|criminal\s+case)\b",
            r"\b(account\s+(will\s+be\s+)?(blocked|frozen|suspended|terminated))\b",
            r"\b(sim\s+(card\s+)?(will\s+be\s+)?(blocked|disconnected|deactivated))\b",
            r"\b(money\s+laundering|illegal\s+transaction|drugs?\s+found)\b",
            r"\b(penalty|heavy\s+fine|confiscate)\b",
        ],
    },
    "reward_lure": {
        "weight": 0.10,
        "severity": "MEDIUM",
        "description": "Luring victim with fake prizes, lottery winnings, or unearned refunds",
        "patterns": [
            r"\b(congratulations|you\s+have\s+won|lottery\s+winner|lucky\s+draw)\b",
            r"\b(cash\s+prize|reward\s+points|free\s+gift)\b",
            r"\b(tax\s+refund|electricity\s+bill\s+refund|overpaid\s+amount)\b",
            r"\b(kyc\s+bonus|claim\s+your\s+(reward|prize|cash))\b",
        ],
    },
}

# Pre-compile all regexes for maximum performance
COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {
    category: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
    for category, data in SCAM_CATEGORIES.items()
}

