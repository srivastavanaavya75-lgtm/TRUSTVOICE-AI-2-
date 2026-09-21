"""
TRUSTVOICE AI
Explainable Conversation Risk Engine

Internal prototype:
- Deterministic
- Explainable
- Scripted scenarios
- Keyword/pattern based

For SIH final:
Replace the rule engine with trained ML/NLP/audio models.
"""

OTP_PATTERNS = [
    "otp",
    "one time password",
    "verification code",
    "passcode",
    "code you just received",
]

FINANCIAL_PATTERNS = [
    "transfer",
    "send money",
    "payment",
    "pay",
    "bank",
    "upi",
    "transaction",
    "invoice",
    "salary",
]

SENSITIVE_PATTERNS = [
    "employee database",
    "database",
    "password",
    "credential",
    "customer data",
    "personal email",
    "personal gmail",
    "confidential",
    "secret",
]

PERSONAL_INFO_PATTERNS = [
    "account number",
    "phone number",
    "address",
    "date of birth",
]

URGENCY_PATTERNS = [
    "immediately",
    "right now",
    "urgent",
    "hurry",
    "do this now",
    "as soon as possible",
]

THREAT_PATTERNS = [
    "blocked",
    "permanently blocked",
    "police",
    "legal action",
    "you will lose",
    "otherwise",
]

SECRECY_PATTERNS = [
    "don't tell anyone",
    "do not tell anyone",
    "keep this secret",
    "don't mention this",
    "no one should know",
]

UNUSUAL_CHANNEL_PATTERNS = [
    "personal email",
    "personal gmail",
    "send it privately",
    "outside the usual channel",
    "different account",
]


def matches(text, patterns):
    """
    Returns True when any pattern appears in the text.
    """

    text = text.lower()

    return any(
        pattern in text
        for pattern in patterns
    )


class RiskEngine:
    """
    Explainable interaction-risk engine.

    This prototype intentionally uses rules so judges
    can understand exactly why a risk score changed.
    """

    def evaluate(
        self,
        transcript,
        voice_authenticity="REAL",
        identity="UNVERIFIED",
        previous_score=None,
    ):

        risk = 0

        signals = []

        intent = "Normal conversation"

        behaviour = []


        # ====================================================
        # VOICE
        # ====================================================

        if voice_authenticity == "SUSPICIOUS":

            risk += 35

            signals.append(
                "Possible synthetic or manipulated voice"
            )

        elif voice_authenticity == "REAL":

            risk -= 3


        # ====================================================
        # IDENTITY
        # ====================================================

        if identity == "UNVERIFIED":

            risk += 4


        # ====================================================
        # INTENT DETECTION
        # ====================================================

        if matches(
            transcript,
            OTP_PATTERNS
        ):

            intent = "OTP / Credential Request"

            risk += 30

            signals.append(
                "OTP / credential request detected"
            )

        elif matches(
            transcript,
            FINANCIAL_PATTERNS
        ):

            intent = "Financial Request"

            risk += 22

            signals.append(
                "Financial request detected"
            )

        elif matches(
            transcript,
            SENSITIVE_PATTERNS
        ):

            intent = "Sensitive Data Request"

            risk += 27

            signals.append(
                "Sensitive data request detected"
            )

        elif matches(
            transcript,
            PERSONAL_INFO_PATTERNS
        ):

            intent = "Personal Information Request"

            risk += 16

            signals.append(
                "Personal information request detected"
            )


        # ====================================================
        # SOCIAL ENGINEERING
        # ====================================================

        if matches(
            transcript,
            URGENCY_PATTERNS
        ):

            risk += 18

            behaviour.append(
                "Urgency"
            )

            signals.append(
                "Urgency detected"
            )


        if matches(
            transcript,
            THREAT_PATTERNS
        ):

            risk += 20

            behaviour.append(
                "Threat"
            )

            signals.append(
                "Threatening language detected"
            )


        if matches(
            transcript,
            SECRECY_PATTERNS
        ):

            risk += 24

            behaviour.append(
                "Secrecy"
            )

            signals.append(
                "Secrecy request detected"
            )


        if matches(
            transcript,
            UNUSUAL_CHANNEL_PATTERNS
        ):

            risk += 20

            behaviour.append(
                "Unusual channel"
            )

            signals.append(
                "Unusual delivery channel detected"
            )


        # ====================================================
        # AUTHORITY CLAIM
        # ====================================================

        if (
            "bank" in transcript.lower()
            and identity == "UNVERIFIED"
        ):

            risk += 8

            signals.append(
                "Authority claim from unverified caller"
            )


        # ====================================================
        # KILLER FEATURE
        #
        # Real voice + verified identity
        # does NOT guarantee a safe interaction.
        # ====================================================

        if (
            identity == "VERIFIED"
            and (
                matches(
                    transcript,
                    SENSITIVE_PATTERNS
                )
                or matches(
                    transcript,
                    SECRECY_PATTERNS
                )
                or matches(
                    transcript,
                    UNUSUAL_CHANNEL_PATTERNS
                )
            )
        ):

            risk += 15

            signals.append(
                "Verified identity but request "
                "is contextually unusual"
            )


        # ====================================================
        # SCORE
        # ====================================================

        calculated_score = max(
            0,
            min(
                100,
                100 - risk
            )
        )


        # Scripted scores make the internal demonstration
        # repeatable and visually consistent.

        if previous_score is not None:

            score = previous_score

        else:

            score = calculated_score


        # ====================================================
        # STATUS
        # ====================================================

        if score >= 85:

            status = "SAFE"

        elif score >= 65:

            status = "CAUTION"

        elif score >= 45:

            status = "SUSPICIOUS"

        elif score >= 20:

            status = "HIGH RISK"

        else:

            status = "CRITICAL"


        # ====================================================
        # RESULT
        # ====================================================

        return {

            "score": int(score),

            "status": status,

            "intent": intent,

            "behaviour": (
                " + ".join(behaviour)
                if behaviour
                else
                "No major pressure indicators"
            ),

            "signals": list(
                dict.fromkeys(signals)
            ),
        }


# ============================================================
# DEMO SCENARIOS
# ============================================================

SCENARIOS = {


    # ========================================================
    # SCENARIO A
    # ========================================================

    "A": {

        "title":
            "Scenario A · Unverified Caller / Scam Escalation",

        "steps": [

            {

                "text":
                    "Hello, good afternoon.",

                "score": 94,

                "voice": "REAL",

                "identity": "UNVERIFIED",

            },

            {

                "text":
                    "I am calling from your bank's security department.",

                "score": 78,

                "voice": "REAL",

                "identity": "UNVERIFIED",

            },

            {

                "text":
                    "We noticed unusual activity. "
                    "Can you confirm your account number?",

                "score": 55,

                "voice": "REAL",

                "identity": "UNVERIFIED",

            },

            {

                "text":
                    "To verify you, please share "
                    "the OTP you just received.",

                "score": 28,

                "voice": "REAL",

                "identity": "UNVERIFIED",

            },

            {

                "text":
                    "You must do this immediately "
                    "or your account will be permanently blocked.",

                "score": 12,

                "voice": "REAL",

                "identity": "UNVERIFIED",

            },

        ],
    },


    # ========================================================
    # SCENARIO B
    # ========================================================

    "B": {

        "title":
            "Scenario B · Verified Real Voice / Dangerous Request",

        "steps": [

            {

                "text":
                    "Hey, it's me. Hope your day is going well.",

                "score": 92,

                "voice": "REAL",

                "identity": "VERIFIED",

            },

            {

                "text":
                    "I need the complete employee database "
                    "for something urgent.",

                "score": 60,

                "voice": "REAL",

                "identity": "VERIFIED",

            },

            {

                "text":
                    "Send it to my personal email, "
                    "not the usual channel.",

                "score": 35,

                "voice": "REAL",

                "identity": "VERIFIED",

            },

            {

                "text":
                    "And please don't mention this "
                    "to anyone on the team for now.",

                "score": 14,

                "voice": "REAL",

                "identity": "VERIFIED",

            },

        ],
    },
}