"""
OpenRouter AI Advisory Service for Tanzania Climate System
==========================================================
Uses OpenRouter API to generate intelligent, context-aware agricultural
and disaster advisory based on real satellite drought/flood data.

Model: configurable via OPENROUTER_MODEL in settings (default: gemma-3-27b-it:free)
"""

import json
import urllib.request
import urllib.error

from django.conf import settings


# ---------------------------------------------------------------------------
# Core request helper (no external dependencies — uses stdlib urllib)
# ---------------------------------------------------------------------------

def _call_openrouter(messages, max_tokens=800, temperature=0.4):
    """
    Send a chat completion request to OpenRouter.
    Returns the assistant message string, or raises on failure.
    """
    api_key = getattr(settings, 'OPENROUTER_API_KEY', '')
    base_url = getattr(settings, 'OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    model = getattr(settings, 'OPENROUTER_MODEL', 'google/gemma-3-27b-it:free')

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
            'X-Title': 'Tanzania Smart Climate Advisor',
        },
        method='POST'
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data['choices'][0]['message']['content']


# ---------------------------------------------------------------------------
# Advisory generators
# ---------------------------------------------------------------------------

def generate_drought_advisory(region_data, indices, forecast_days=7):
    """
    Generate an AI advisory for drought conditions in a Tanzania region.

    Args:
        region_data (dict): keys — region_name, risk_level (1-5),
                            affected_area_km2, population_at_risk,
                            realtime_indicators, risk_factors
        indices (dict):     keys — vci, tci, spi, smi (0-100 each)
        forecast_days (int): prediction horizon

    Returns:
        dict with keys: summary, immediate_actions, medium_term, crops_advice,
                        water_management, livestock, confidence, generated_by
    """
    risk_level = region_data.get('risk_level', 3)
    region_name = region_data.get('region_name', 'Tanzania')
    risk_labels = {1: 'extreme', 2: 'severe', 3: 'moderate', 4: 'mild', 5: 'none'}
    severity = risk_labels.get(risk_level, 'moderate')

    system_prompt = (
        "You are a senior agricultural meteorologist advising the Tanzania Government's "
        "Central Disaster Management Committee. Your advice is data-driven, precise, and "
        "actionable. Respond ONLY with a valid JSON object — no markdown, no prose outside JSON."
    )

    user_prompt = f"""
Satellite drought assessment for {region_name}, Tanzania:
- Drought severity: {severity.upper()} (class {risk_level}/5, where 1=extreme)
- Forecast window: {forecast_days} days
- Affected area: {region_data.get('affected_area_km2', 'unknown')} km²
- Population at risk: {region_data.get('population_at_risk', 'unknown')}
- VCI (vegetation): {indices.get('vci', 'N/A')}/100 — lower = worse vegetation health
- TCI (temperature): {indices.get('tci', 'N/A')}/100 — lower = more heat stress
- SPI (precipitation): {indices.get('spi', 'N/A')}/100 — lower = drier than normal
- SMI (soil moisture): {indices.get('smi', 'N/A')}/100 — lower = drier soils
- Key risk factors: {', '.join(region_data.get('risk_factors', ['N/A']))}

Return this JSON structure (strings only, no nested objects):
{{
  "summary": "2-sentence executive summary of drought situation",
  "immediate_actions": "3-4 concrete immediate actions (within 7 days)",
  "medium_term": "2-3 actions for 1-4 weeks ahead",
  "crops_advice": "specific advice for Tanzania main crops (maize, rice, cassava, sorghum)",
  "water_management": "water conservation and irrigation management guidance",
  "livestock": "livestock management advice",
  "food_security": "food security and early warning implications",
  "confidence": "high|medium|low"
}}
"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        raw = _call_openrouter(messages, max_tokens=700, temperature=0.3)
        # Strip any markdown code fences if the model adds them
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        parsed = json.loads(raw)
        parsed['generated_by'] = 'OpenRouter AI (satellite-grounded)'
        return parsed
    except Exception as exc:
        return _fallback_drought_advisory(region_data, indices, severity, str(exc))


def generate_flood_advisory(region_data, forecast_days=7):
    """
    Generate an AI advisory for flood conditions.

    Args:
        region_data (dict): region_name, risk_level, realtime_indicators,
                            risk_factors, affected_area_km2, population_at_risk
        forecast_days (int): prediction horizon

    Returns:
        dict with advisory sections + generated_by
    """
    risk_level = region_data.get('risk_level', 3)
    region_name = region_data.get('region_name', 'Tanzania')
    risk_labels = {1: 'very low', 2: 'low', 3: 'moderate', 4: 'high', 5: 'very high'}
    severity = risk_labels.get(risk_level, 'moderate')
    indicators = region_data.get('realtime_indicators', {})

    system_prompt = (
        "You are a senior hydrologist advising the Tanzania Government Flood Control Unit. "
        "Provide actionable, region-specific guidance. "
        "Respond ONLY with a valid JSON object — no markdown, no prose outside JSON."
    )

    user_prompt = f"""
Satellite flood assessment for {region_name}, Tanzania:
- Flood risk: {severity.upper()} (class {risk_level}/5)
- Forecast: {forecast_days} days ahead
- Affected area: {region_data.get('affected_area_km2', 'unknown')} km²
- Population at risk: {region_data.get('population_at_risk', 'unknown')}
- Precipitation (30d total): {indicators.get('precipitation_mm', 'N/A')} mm
- Water coverage (SAR): {indicators.get('water_coverage_percent', 'N/A')}%
- Soil saturation: {indicators.get('soil_moisture_index', 'N/A')}
- Risk factors: {', '.join(region_data.get('risk_factors', ['N/A']))}

Return this JSON structure:
{{
  "summary": "2-sentence executive summary of flood situation",
  "evacuation_guidance": "evacuation priorities and routes if risk >= moderate",
  "immediate_response": "immediate flood response actions (0-48 hours)",
  "infrastructure": "infrastructure protection steps (roads, bridges, utilities)",
  "agriculture": "crop and farmland protection advice",
  "health_risks": "waterborne disease and health risks to communicate",
  "recovery": "post-flood recovery preparation steps",
  "confidence": "high|medium|low"
}}
"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        raw = _call_openrouter(messages, max_tokens=700, temperature=0.3)
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        parsed = json.loads(raw)
        parsed['generated_by'] = 'OpenRouter AI (satellite-grounded)'
        return parsed
    except Exception as exc:
        return _fallback_flood_advisory(region_data, severity, str(exc))


def generate_crop_calendar_advisory(region_name, prediction_type, risk_level,
                                     precipitation_mm, ndvi, soil_moisture, month=None):
    """
    Generate AI-powered crop calendar and planting recommendations.

    Returns dict with: planting_window, recommended_crops, avoidance_crops,
                       water_needs, fertilizer_guidance, harvest_timing,
                       generated_by
    """
    from datetime import datetime
    current_month = month or datetime.now().month
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_name = month_names[current_month] if 1 <= current_month <= 12 else 'current month'

    system_prompt = (
        "You are a specialist in Tanzania agricultural systems. Provide practical crop "
        "management advice grounded in remote sensing data. Respond ONLY with valid JSON."
    )

    user_prompt = f"""
Agricultural planning data for {region_name}, Tanzania — {month_name}:
- Climate risk type: {prediction_type} (risk level {risk_level}/5)
- 30-day rainfall: {precipitation_mm} mm
- NDVI vegetation index: {ndvi} (0=bare, 1=lush)
- Soil moisture: {soil_moisture} (0=dry, 1=saturated)

Tanzania main growing seasons: March–May (Masika) and October–December (Vuli).
Main crops: maize, rice, cassava, beans, sorghum, millet, cotton, coffee, tea.

Return this JSON:
{{
  "season_status": "current season status and outlook",
  "planting_window": "optimal planting window advice for current conditions",
  "recommended_crops": ["crop1", "crop2", "crop3"],
  "crops_to_avoid": ["crop1", "crop2"],
  "drought_tolerant_varieties": "specific drought/flood-tolerant variety names for Tanzania",
  "water_management": "irrigation and water harvesting guidance",
  "fertilizer_guidance": "fertilizer application advice given soil conditions",
  "pest_disease_risk": "heightened pest/disease risks under current conditions",
  "harvest_timing": "harvest timing guidance if applicable",
  "post_harvest": "post-harvest handling given climate conditions"
}}
"""

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        raw = _call_openrouter(messages, max_tokens=600, temperature=0.4)
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        parsed = json.loads(raw)
        parsed['generated_by'] = 'OpenRouter AI (satellite-grounded)'
        return parsed
    except Exception as exc:
        return {
            'season_status': 'Advisory service temporarily unavailable.',
            'planting_window': 'Consult your local agricultural extension officer.',
            'recommended_crops': ['maize', 'sorghum', 'cassava'],
            'crops_to_avoid': [],
            'drought_tolerant_varieties': 'Contact TARI for recommended varieties.',
            'water_management': 'Practice water conservation.',
            'fertilizer_guidance': 'Follow standard recommendations.',
            'pest_disease_risk': 'Monitor regularly.',
            'harvest_timing': 'As per normal seasonal schedule.',
            'post_harvest': 'Ensure dry storage.',
            'error': str(exc),
            'generated_by': 'Static fallback (AI unavailable)',
        }


# ---------------------------------------------------------------------------
# Fallback advisories (rule-based — no AI)
# ---------------------------------------------------------------------------

def _fallback_drought_advisory(region_data, indices, severity, error_note=''):
    """Rule-based fallback when AI is unavailable."""
    risk_level = region_data.get('risk_level', 3)

    if risk_level <= 2:
        immediate = (
            "Activate emergency water distribution networks. "
            "Issue drought alert Level 2 to district offices. "
            "Begin livestock destocking in affected areas. "
            "Mobilise food security stockpiles."
        )
        medium = (
            "Coordinate with national grain reserves for emergency allocation. "
            "Initiate water trucking to affected communities. "
            "Establish drought relief committees at ward level."
        )
    elif risk_level == 3:
        immediate = (
            "Activate early warning systems and community advisories. "
            "Advise farmers to prioritise drought-tolerant crops. "
            "Promote water-efficient irrigation practices."
        )
        medium = (
            "Monitor rainfall forecasts weekly. "
            "Distribute drought-tolerant seed varieties. "
            "Promote conservation agriculture."
        )
    else:
        immediate = "Continue routine monitoring. No immediate action required."
        medium = "Maintain preparedness plans and seasonal advisories."

    return {
        'summary': (
            f"{severity.capitalize()} drought conditions detected in {region_data.get('region_name', 'Tanzania')}. "
            f"VCI={indices.get('vci', 'N/A')}, SPI={indices.get('spi', 'N/A')} indicate "
            f"{'severe vegetation and precipitation deficit' if risk_level <= 2 else 'below-normal conditions'}."
        ),
        'immediate_actions': immediate,
        'medium_term': medium,
        'crops_advice': (
            "Plant drought-tolerant varieties: Sorghum, Millet, Cassava. "
            "Delay planting maize until rainfall improves. "
            "Apply mulching to conserve soil moisture."
        ),
        'water_management': (
            "Promote water harvesting, check dams, and micro-irrigation. "
            "Ration water use for livestock and crops. "
            "Identify alternative water sources."
        ),
        'livestock': (
            "Move livestock to areas with better pasture. "
            "Reduce herd size if feed is critically scarce. "
            "Supplement feed with drought-tolerant fodder crops."
        ),
        'food_security': (
            "Activate early warning protocols. "
            "Monitor household food security indicators. "
            "Coordinate with WFP and national safety net programmes."
        ),
        'confidence': 'medium',
        'generated_by': f'Rule-based fallback (AI error: {error_note})',
    }


def _fallback_flood_advisory(region_data, severity, error_note=''):
    """Rule-based fallback for flood advisory."""
    risk_level = region_data.get('risk_level', 3)

    if risk_level >= 4:
        evacuation = (
            "Initiate evacuation of communities in low-lying flood plains. "
            "Prioritise schools, hospitals, and elderly/disabled persons. "
            "Use designated evacuation routes away from river channels."
        )
        immediate = (
            "Deploy emergency response teams immediately. "
            "Pre-position boats and rescue equipment. "
            "Open emergency shelters and coordinate with Red Cross."
        )
    else:
        evacuation = "Prepare evacuation plans but hold unless conditions worsen rapidly."
        immediate = "Issue community flood watches. Inspect drainage infrastructure."

    return {
        'summary': (
            f"{severity.capitalize()} flood risk in {region_data.get('region_name', 'Tanzania')}. "
            f"Satellite analysis shows elevated water coverage and precipitation."
        ),
        'evacuation_guidance': evacuation,
        'immediate_response': immediate,
        'infrastructure': (
            "Inspect and clear all drainage channels. "
            "Place sandbags at critical culverts and bridges. "
            "Shut down electrical infrastructure in flood-prone areas."
        ),
        'agriculture': (
            "Harvest mature crops immediately if safe to do so. "
            "Move stored crops and equipment to higher ground. "
            "Document damage for insurance and government relief purposes."
        ),
        'health_risks': (
            "Risk of cholera, typhoid, and malaria increases post-flood. "
            "Distribute water purification tablets. "
            "Increase surveillance for waterborne disease outbreaks."
        ),
        'recovery': (
            "Plan soil assessment and replanting calendar. "
            "Document livestock and crop losses. "
            "Engage TARI, irrigation authorities, and insurance schemes."
        ),
        'confidence': 'medium',
        'generated_by': f'Rule-based fallback (AI error: {error_note})',
    }
