"""
Decision Support Service for Tanzania Climate Intelligence Platform
====================================================================
Uses OpenRouter API with a high-capability model (default: x-ai/grok-2-1212)
to generate stakeholder-grade climate adaptation decisions.

Inspired by Deltares Delft3D Flexible Mesh adaptation pathway methodology:
  - Pathways = branching decision trees mapped to risk thresholds
  - Scenarios = plausible future climate trajectories
  - Tipping points = risk levels that trigger a new pathway branch

Model is configurable via OPENROUTER_DECISION_MODEL in settings
(falls back to OPENROUTER_MODEL if decision model not set).
"""

import json
import urllib.request
import urllib.error
from datetime import datetime

from django.conf import settings


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _call_decision_model(messages, max_tokens=1200, temperature=0.35):
    """
    Call OpenRouter with the decision-support model.
    Returns assistant content string or raises.
    """
    api_key = getattr(settings, 'OPENROUTER_API_KEY', '')
    base_url = getattr(settings, 'OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

    # Use the dedicated decision model if set, otherwise fall back to advisory model
    model = getattr(settings, 'OPENROUTER_DECISION_MODEL', None) \
         or getattr(settings, 'OPENROUTER_MODEL', 'google/gemma-3-27b-it:free')

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured in settings.")

    payload = json.dumps({
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
    }).encode('utf-8')

    req = urllib.request.Request(
        f'{base_url}/chat/completions',
        data=payload,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://scadvisor.tz',
            'X-Title': 'Tanzania Climate Decision Support',
        },
        method='POST'
    )

    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data['choices'][0]['message']['content']


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences that some models add around JSON."""
    raw = raw.strip()
    if raw.startswith('```'):
        parts = raw.split('```')
        # parts[0] is empty, parts[1] is the code block content
        raw = parts[1]
        if raw.startswith('json'):
            raw = raw[4:]
    return raw.strip()


# ---------------------------------------------------------------------------
# Stakeholder decision generator
# ---------------------------------------------------------------------------

STAKEHOLDER_TYPES = {
    'government': 'National/Regional Government Disaster Management Committee',
    'farmer': 'Smallholder Farmer / Farmer Group',
    'ngo': 'NGO / Humanitarian Response Organisation',
    'infrastructure': 'Infrastructure Manager (Roads, Water, Utilities)',
    'health': 'Public Health Officer',
    'investor': 'Agricultural Investor / Private Sector',
}


def generate_stakeholder_decision(region_data: dict, risk_data: dict,
                                   forecast_data: dict, stakeholder_type: str = 'government') -> dict:
    """
    Generate a structured decision brief for a specific stakeholder type.

    Args:
        region_data:      {region_name, area_km2, population, agro_zone}
        risk_data:        {risk_type, risk_level (1-5), risk_score (0-1),
                           satellite_indicators, risk_factors}
        forecast_data:    {forecast_days, trend ('increasing'|'stable'|'decreasing'),
                           confidence}
        stakeholder_type: key from STAKEHOLDER_TYPES

    Returns:
        dict with decision brief sections
    """
    region_name = region_data.get('region_name', 'Tanzania')
    risk_type = risk_data.get('risk_type', 'drought')
    risk_level = risk_data.get('risk_level', 3)
    risk_score = risk_data.get('risk_score', 0.5)
    risk_labels = {1: 'EXTREME', 2: 'SEVERE', 3: 'MODERATE', 4: 'MILD', 5: 'NONE'}
    severity = risk_labels.get(risk_level, 'MODERATE')

    stakeholder_label = STAKEHOLDER_TYPES.get(stakeholder_type, stakeholder_type)
    forecast_days = forecast_data.get('forecast_days', 7)
    trend = forecast_data.get('trend', 'stable')

    indicators = risk_data.get('satellite_indicators', {})
    ind_text = '\n'.join(
        f'  - {k}: {v}' for k, v in indicators.items() if v not in (None, '', 'N/A')
    ) or '  - Real-time satellite data not available for this query'

    system_prompt = (
        "You are a world-class climate risk advisor embedded in Tanzania's National Adaptation "
        "Planning framework. Your outputs directly inform government and institutional decisions. "
        "Ground every recommendation in the satellite data provided. "
        "Use the Deltares adaptation pathways framework: identify the current pathway "
        "(present conditions), tipping points (risk thresholds that demand action), "
        "and branch actions (what to do if conditions worsen/improve). "
        "Respond ONLY with a valid JSON object — no prose outside JSON, no markdown fences."
    )

    user_prompt = f"""
CLIMATE DECISION BRIEF REQUEST

Stakeholder: {stakeholder_label}
Region: {region_name}, Tanzania
Risk Type: {risk_type.upper()}
Severity: {severity} (class {risk_level}/5, composite score {risk_score:.2f})
Forecast Horizon: {forecast_days} days | Trend: {trend.upper()}

Satellite Indicators:
{ind_text}

Risk Factors: {', '.join(risk_data.get('risk_factors', ['N/A']))}
Population: ~{region_data.get('population', 'unknown')}
Agricultural Zone: {region_data.get('agro_zone', 'Mixed agroecological zone')}
Query Date: {datetime.now().strftime('%B %Y')}

Return this JSON structure (all values must be strings or arrays of strings):
{{
  "executive_summary": "3-sentence situation brief for senior decision-makers",
  "current_pathway": "description of current climate pathway and what it means for this stakeholder",
  "tipping_points": ["threshold 1 that would trigger escalation", "threshold 2", "threshold 3"],
  "immediate_actions": ["action 1 (0-7 days)", "action 2", "action 3", "action 4"],
  "medium_term_actions": ["action 1 (1-4 weeks)", "action 2", "action 3"],
  "long_term_adaptation": ["structural adaptation measure 1", "measure 2", "measure 3"],
  "resource_requirements": "key resources, budget priorities, or capacity needs for this stakeholder",
  "branch_if_worsens": ["what to do if risk escalates to next level — action 1", "action 2", "action 3"],
  "branch_if_improves": ["what to do if conditions improve — action 1", "action 2"],
  "coordination_notes": "who this stakeholder should coordinate with and why",
  "confidence": "high|medium|low",
  "data_basis": "summary of satellite data quality and reliability note"
}}
"""

    try:
        raw = _call_decision_model(
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': user_prompt}],
            max_tokens=1100
        )
        result = json.loads(_strip_json_fences(raw))
        result['generated_by'] = 'OpenRouter Decision Model'
        result['model_used'] = getattr(settings, 'OPENROUTER_DECISION_MODEL', 'default')
        result['stakeholder_type'] = stakeholder_type
        return result
    except Exception as exc:
        return _fallback_decision(risk_type, severity, stakeholder_type, str(exc))


# ---------------------------------------------------------------------------
# Adaptation pathway generator (Deltares-inspired)
# ---------------------------------------------------------------------------

def generate_adaptation_pathways(risk_timeline: list, stakeholder_type: str = 'government',
                                  region_name: str = 'Tanzania') -> dict:
    """
    Generate Deltares-style adaptation pathway map.

    Args:
        risk_timeline: list of {year, scenario, risk_level, key_driver}
                       e.g. [{'year': 2026, 'scenario': 'baseline', 'risk_level': 3, ...}]
        stakeholder_type: key from STAKEHOLDER_TYPES
        region_name: target region

    Returns:
        dict with pathways, tipping_points, action_portfolio, pathway_diagram_data
    """
    stakeholder_label = STAKEHOLDER_TYPES.get(stakeholder_type, stakeholder_type)
    timeline_text = '\n'.join(
        f"  {e.get('year', '?')}: {e.get('scenario', 'N/A')} | "
        f"Risk Level {e.get('risk_level', '?')}/5 | Driver: {e.get('key_driver', 'N/A')}"
        for e in risk_timeline
    ) or '  No timeline data provided — generate representative pathways for Tanzania'

    system_prompt = (
        "You are applying the Deltares Adaptation Pathways methodology to climate risk for "
        "Tanzania. Generate structured adaptation pathways that map sequences of actions "
        "across risk thresholds over time. Each pathway represents a decision route that "
        "stakeholders can follow depending on how climate conditions evolve. "
        "Respond ONLY with valid JSON."
    )

    user_prompt = f"""
ADAPTATION PATHWAY ANALYSIS

Stakeholder: {stakeholder_label}
Region: {region_name}, Tanzania

Risk Timeline:
{timeline_text}

Tanzania climate context: bimodal rainfall (Masika March-May, Vuli Oct-Dec),
semi-arid central plateau, coastal humid zones, highland agriculture.
Key vulnerabilities: smallholder rain-fed agriculture, limited irrigation,
informal settlements in flood-prone areas.

Return this JSON structure:
{{
  "pathway_a": {{
    "name": "Short name for pathway A (low-regret early action)",
    "description": "What this pathway involves and when it applies",
    "trigger_condition": "specific risk threshold or event that activates this pathway",
    "actions": ["action 1", "action 2", "action 3", "action 4"],
    "horizon": "near-term|medium-term|long-term",
    "cost_level": "low|medium|high",
    "co_benefits": ["benefit 1", "benefit 2"]
  }},
  "pathway_b": {{
    "name": "Pathway B (medium-risk adaptive measures)",
    "description": "...",
    "trigger_condition": "...",
    "actions": ["action 1", "action 2", "action 3"],
    "horizon": "...",
    "cost_level": "...",
    "co_benefits": ["benefit 1"]
  }},
  "pathway_c": {{
    "name": "Pathway C (transformational/high-risk response)",
    "description": "...",
    "trigger_condition": "extreme risk threshold requiring transformational change",
    "actions": ["action 1", "action 2", "action 3"],
    "horizon": "long-term",
    "cost_level": "high",
    "co_benefits": []
  }},
  "global_tipping_points": [
    {{"risk_level": 2, "year_est": "estimated year", "description": "what changes at this level"}},
    {{"risk_level": 1, "year_est": "estimated year", "description": "what changes at this level"}}
  ],
  "recommended_starting_pathway": "A|B|C",
  "pathway_sequence": "narrative description of how to sequence pathways A → B → C",
  "key_uncertainties": ["uncertainty 1", "uncertainty 2", "uncertainty 3"],
  "monitoring_triggers": ["satellite indicator to watch 1", "indicator 2", "indicator 3"]
}}
"""

    try:
        raw = _call_decision_model(
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': user_prompt}],
            max_tokens=1200
        )
        result = json.loads(_strip_json_fences(raw))
        result['generated_by'] = 'OpenRouter — Deltares Pathway Methodology'
        return result
    except Exception as exc:
        return _fallback_pathways(stakeholder_type, str(exc))


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

def generate_scenario_analysis(current_risk: dict, region_name: str = 'Tanzania') -> dict:
    """
    Generate three climate scenario analyses (optimistic, baseline, pessimistic)
    with probabilities and recommended responses.

    Args:
        current_risk: {risk_type, risk_level, risk_score, satellite_indicators}
        region_name

    Returns:
        dict with optimistic, baseline, pessimistic scenarios + recommendation
    """
    risk_type = current_risk.get('risk_type', 'drought')
    risk_level = current_risk.get('risk_level', 3)
    risk_score = current_risk.get('risk_score', 0.5)

    indicators = current_risk.get('satellite_indicators', {})
    ind_text = ', '.join(f"{k}={v}" for k, v in indicators.items() if v not in (None, '', 'N/A')) \
               or 'limited satellite data'

    system_prompt = (
        "You are a climate scenario analyst for East Africa. Generate three plausible "
        "2-season climate scenarios for Tanzania based on current satellite conditions. "
        "Respond ONLY with valid JSON."
    )

    user_prompt = f"""
CLIMATE SCENARIO ANALYSIS

Region: {region_name}, Tanzania
Current Risk: {risk_type.upper()} at level {risk_level}/5 (score {risk_score:.2f})
Current Indicators: {ind_text}
Analysis Date: {datetime.now().strftime('%B %Y')}

Generate three scenarios for the next 2 growing seasons (6 months):

Return this JSON:
{{
  "optimistic": {{
    "probability_pct": 25,
    "name": "scenario name",
    "description": "what happens in this scenario",
    "rainfall_outlook": "specific mm range forecast",
    "risk_trajectory": "improving|stable|worsening",
    "end_risk_level": 1-5,
    "recommended_actions": ["action 1", "action 2", "action 3"],
    "early_warning_signs": ["sign 1", "sign 2"]
  }},
  "baseline": {{
    "probability_pct": 50,
    "name": "scenario name",
    "description": "most likely scenario",
    "rainfall_outlook": "specific mm range forecast",
    "risk_trajectory": "...",
    "end_risk_level": 1-5,
    "recommended_actions": ["action 1", "action 2", "action 3"],
    "early_warning_signs": ["sign 1", "sign 2"]
  }},
  "pessimistic": {{
    "probability_pct": 25,
    "name": "scenario name",
    "description": "worst-case scenario",
    "rainfall_outlook": "specific mm range forecast",
    "risk_trajectory": "worsening",
    "end_risk_level": 1-5,
    "recommended_actions": ["action 1", "action 2", "action 3", "action 4"],
    "early_warning_signs": ["sign 1", "sign 2", "sign 3"]
  }},
  "robust_actions": ["action valid across ALL scenarios 1", "action 2", "action 3"],
  "key_decision_trigger": "the single most important indicator to monitor",
  "enso_context": "current ENSO phase and likely Tanzania impact",
  "iod_context": "Indian Ocean Dipole context for East Africa rainfall"
}}
"""

    try:
        raw = _call_decision_model(
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': user_prompt}],
            max_tokens=1000
        )
        result = json.loads(_strip_json_fences(raw))
        result['generated_by'] = 'OpenRouter Scenario Model'
        return result
    except Exception as exc:
        return _fallback_scenarios(risk_type, str(exc))


# ---------------------------------------------------------------------------
# Conversational Q&A
# ---------------------------------------------------------------------------

def answer_decision_question(question: str, context: dict) -> dict:
    """
    Answer a free-form stakeholder question in the context of current risk data.

    Args:
        question: stakeholder's natural language question
        context: {region_name, risk_type, risk_level, satellite_indicators,
                  stakeholder_type}

    Returns:
        dict with answer, follow_up_questions, data_references
    """
    region = context.get('region_name', 'Tanzania')
    risk_type = context.get('risk_type', 'drought')
    risk_level = context.get('risk_level', 3)
    stakeholder_type = context.get('stakeholder_type', 'general')
    indicators = context.get('satellite_indicators', {})
    ind_text = ', '.join(f"{k}={v}" for k, v in indicators.items() if v not in (None, '', 'N/A')) \
               or 'satellite data not specified'

    system_prompt = (
        "You are an expert climate advisor for Tanzania, combining satellite remote sensing "
        "expertise with agricultural and disaster risk management knowledge. "
        "Answer questions concisely, reference the provided satellite data where relevant, "
        "and give actionable responses. Respond ONLY with valid JSON."
    )

    user_prompt = f"""
STAKEHOLDER QUESTION

Region: {region}, Tanzania
Current Risk: {risk_type} at level {risk_level}/5
Stakeholder: {stakeholder_type}
Satellite Indicators: {ind_text}

Question: {question}

Return this JSON:
{{
  "answer": "clear, actionable answer (3-6 sentences)",
  "confidence": "high|medium|low",
  "caveats": "any important limitations or uncertainties in this answer",
  "follow_up_questions": ["suggested follow-up 1", "suggested follow-up 2"],
  "data_references": ["satellite data point that supports this answer 1", "data point 2"]
}}
"""

    try:
        raw = _call_decision_model(
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': user_prompt}],
            max_tokens=500,
            temperature=0.4
        )
        return json.loads(_strip_json_fences(raw))
    except Exception as exc:
        return {
            'answer': f'Unable to generate AI response: {exc}. '
                      'Please check your OpenRouter API configuration and try again.',
            'confidence': 'low',
            'caveats': 'AI service unavailable',
            'follow_up_questions': [],
            'data_references': [],
        }


# ---------------------------------------------------------------------------
# Fallback responses (when AI is unavailable)
# ---------------------------------------------------------------------------

def _fallback_decision(risk_type, severity, stakeholder_type, error):
    return {
        'executive_summary': (
            f'A {severity} {risk_type} condition has been detected in the monitored region. '
            f'Satellite indicators confirm current conditions require stakeholder attention. '
            f'AI detailed analysis unavailable: {error}'
        ),
        'current_pathway': f'Operating under {severity} {risk_type} conditions.',
        'tipping_points': ['Risk escalation to next severity level', 'Extended dry/wet spell >2 weeks'],
        'immediate_actions': [
            'Review current early warning bulletins from TMA',
            'Assess community vulnerability in affected areas',
            'Activate existing contingency plans',
            'Coordinate with relevant line ministries',
        ],
        'medium_term_actions': ['Monitor satellite data daily', 'Prepare resource pre-positioning'],
        'long_term_adaptation': ['Invest in climate-resilient infrastructure'],
        'resource_requirements': 'Refer to national contingency plan for resource allocation.',
        'branch_if_worsens': ['Escalate to higher alert level', 'Request emergency resources'],
        'branch_if_improves': ['Stand down emergency measures', 'Document lessons learned'],
        'coordination_notes': 'Coordinate with NEMC, TMA, and district authorities.',
        'confidence': 'low',
        'data_basis': 'Fallback response — AI service temporarily unavailable.',
        'generated_by': 'Fallback (AI unavailable)',
        'error': str(error),
    }


def _fallback_pathways(stakeholder_type, error):
    return {
        'pathway_a': {
            'name': 'Early Action Pathway',
            'description': 'Low-regret measures implementable now regardless of scenario.',
            'trigger_condition': 'Current risk level >= 3 (moderate)',
            'actions': ['Strengthen early warning dissemination', 'Pre-position emergency supplies',
                        'Activate community preparedness committees', 'Review contingency budgets'],
            'horizon': 'near-term',
            'cost_level': 'low',
            'co_benefits': ['Improved community resilience', 'Reduced response costs'],
        },
        'pathway_b': {
            'name': 'Adaptive Management Pathway',
            'description': 'Medium-level measures as conditions deteriorate.',
            'trigger_condition': 'Risk level >= 2 (severe)',
            'actions': ['Deploy emergency water/food assistance', 'Activate evacuation plans',
                        'Request inter-agency coordination'],
            'horizon': 'medium-term',
            'cost_level': 'medium',
            'co_benefits': ['Reduced disaster impact'],
        },
        'pathway_c': {
            'name': 'Transformational Response',
            'description': 'Structural changes for extreme risk conditions.',
            'trigger_condition': 'Risk level = 1 (extreme)',
            'actions': ['Declare disaster emergency', 'International humanitarian appeal',
                        'Implement long-term livelihood recovery'],
            'horizon': 'long-term',
            'cost_level': 'high',
            'co_benefits': [],
        },
        'global_tipping_points': [
            {'risk_level': 2, 'year_est': '2026-2027', 'description': 'Severe conditions requiring emergency response'},
            {'risk_level': 1, 'year_est': '2027+', 'description': 'Extreme conditions requiring transformational response'},
        ],
        'recommended_starting_pathway': 'A',
        'pathway_sequence': 'Begin with Pathway A now. Transition to B if risk reaches level 2. Reserve C for extreme emergencies.',
        'key_uncertainties': ['Seasonal rainfall variability', 'ENSO forecast uncertainty'],
        'monitoring_triggers': ['NDVI anomaly > -20%', 'SPI-3 < -1.5', 'SAR flood extent > 5%'],
        'generated_by': 'Fallback (AI unavailable)',
        'error': str(error),
    }


def _fallback_scenarios(risk_type, error):
    return {
        'optimistic': {
            'probability_pct': 25,
            'name': 'Favourable Season Recovery',
            'description': 'Above-average rainfall leads to risk reduction.',
            'rainfall_outlook': '10-20% above long-term average',
            'risk_trajectory': 'improving',
            'end_risk_level': 4,
            'recommended_actions': ['Prepare for planting season', 'Rebuild food stocks'],
            'early_warning_signs': ['Positive IOD signal', 'Improved soil moisture'],
        },
        'baseline': {
            'probability_pct': 50,
            'name': 'Near-Normal Season',
            'description': 'Conditions persist near current levels with modest improvement.',
            'rainfall_outlook': 'Within ±10% of long-term average',
            'risk_trajectory': 'stable',
            'end_risk_level': 3,
            'recommended_actions': ['Maintain preparedness', 'Support vulnerable households'],
            'early_warning_signs': ['Continued satellite monitoring', 'Weekly rainfall tracking'],
        },
        'pessimistic': {
            'probability_pct': 25,
            'name': 'Continued Stress',
            'description': f'Below-average rainfall extends {risk_type} conditions.',
            'rainfall_outlook': '15-25% below long-term average',
            'risk_trajectory': 'worsening',
            'end_risk_level': 2,
            'recommended_actions': ['Activate emergency response', 'Scale up food assistance',
                                    'Deploy drought/flood relief'],
            'early_warning_signs': ['NDVI decline', 'SPI deterioration', 'Water level drops'],
        },
        'robust_actions': ['Strengthen early warning systems', 'Diversify crop portfolios',
                           'Build community water storage'],
        'key_decision_trigger': 'Monthly CHIRPS rainfall vs. climatological mean',
        'enso_context': 'Monitor ENSO monthly — La Niña typically reduces Tanzania rainfall.',
        'iod_context': 'Positive IOD generally enhances East Africa short rains (Oct-Dec).',
        'generated_by': 'Fallback (AI unavailable)',
        'error': str(error),
    }
