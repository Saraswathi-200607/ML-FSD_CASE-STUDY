"""
Rule-based, interpretable lifestyle recommendation engine.

* Independent of the ML model: it may use factors the model never saw
  (diet, detailed sleep) and the model never uses these rules.
* Every recommendation carries a plain-language `reason`.
* It gives general lifestyle education only: no diagnoses, no medicines,
  no promise that anything will prevent a disease.
* Missing inputs (None / "") simply skip the related rules.

Input: a dict using the assessment field names (see health/models.py).
Output: list of dicts {id, category, priority, title, advice, reason}.
"""

PRIORITY_ORDER = {"high": 0, "medium": 1, "info": 2}


def _rec(rid, category, priority, title, advice, reason):
    return {"id": rid, "category": category, "priority": priority,
            "title": title, "advice": advice, "reason": reason}


def _has(value):
    return value not in (None, "")


def generate_recommendations(data: dict, risk_category: str | None = None) -> list[dict]:
    recs = []
    g = data.get

    # ---- smoking -----------------------------------------------------------
    if g("smoking") == "Yes":
        recs.append(_rec(
            "smoking", "Tobacco", "high", "Get support to stop smoking",
            "If you currently smoke, ask a doctor or a tobacco-cessation clinic about "
            "support options. If you smoked in the past and have quit, keep it up.",
            "You reported having smoked at least 100 cigarettes in your life. Tobacco "
            "use is a well-established risk factor for heart and lung disease."))

    # ---- alcohol -----------------------------------------------------------
    if g("alcohol_drinking") == "Yes":
        recs.append(_rec(
            "alcohol", "Alcohol", "medium", "Reduce alcohol intake",
            "Consider cutting down, with alcohol-free days each week. If cutting down "
            "is difficult, talk to a healthcare professional.",
            "You reported heavy alcohol drinking. Heavy drinking is linked with raised "
            "blood pressure and other health problems."))

    # ---- physical activity ---------------------------------------------------
    if g("physical_activity") == "No":
        extra = ""
        if g("diff_walking") == "Yes":
            extra = (" Because you reported difficulty walking, ask a doctor or "
                     "physiotherapist which activities are safe for you.")
        recs.append(_rec(
            "activity", "Physical activity", "high", "Build up regular physical activity",
            "Start gently (for example a 10-minute walk daily) and build up gradually "
            "toward the commonly advised 150 minutes of moderate activity per week for "
            "adults." + extra,
            "You reported no physical activity outside your regular job in the past 30 days."))
    elif g("physical_activity") == "Yes":
        recs.append(_rec(
            "activity-ok", "Physical activity", "info", "Keep up your activity",
            "Aim to stay regularly active and include muscle-strengthening activities "
            "on two days a week if you can.",
            "You reported being physically active in the past 30 days."))

    # ---- weight / BMI --------------------------------------------------------
    bmi = g("bmi")
    if _has(bmi):
        bmi = float(bmi)
        if bmi < 18.5:
            recs.append(_rec(
                "bmi-low", "Weight", "medium", "Discuss a low body weight with a professional",
                "A doctor or dietitian can check whether your weight is healthy for you "
                "and suggest suitable nutrition.",
                f"Your calculated BMI is {bmi:.1f}, below the usual 18.5 cut-off."))
        elif bmi >= 27.5:
            recs.append(_rec(
                "bmi-high", "Weight", "high", "Work on gradual, sustainable weight management",
                "Small, steady changes (portion sizes, more movement, fewer sugary drinks "
                "and fried snacks) are more sustainable than crash diets. A dietitian can "
                "personalise this.",
                f"Your calculated BMI is {bmi:.1f}. Many health bodies use lower BMI action "
                "points for South Asian adults (23 and 27.5) than the general 25 and 30, "
                "because health risks can rise at lower BMI."))
        elif bmi >= 23:
            recs.append(_rec(
                "bmi-mid", "Weight", "medium", "Watch your weight trend",
                "Staying active and choosing balanced meals helps prevent gradual weight "
                "gain. Your waist measurement is also worth tracking.",
                f"Your calculated BMI is {bmi:.1f}, in the 'overweight' range using "
                "South Asian cut-offs (23-27.4). BMI is only a rough screening measure."))
        else:
            recs.append(_rec(
                "bmi-ok", "Weight", "info", "Your BMI is in the commonly used healthy range",
                "Keep a balanced diet and regular activity.",
                f"Your calculated BMI is {bmi:.1f} (18.5-22.9). BMI does not show body fat "
                "distribution, so it is only a rough guide."))

    # ---- sleep ---------------------------------------------------------------
    sleep = g("sleep_hours")
    if _has(sleep):
        sleep = float(sleep)
        if sleep < 7:
            recs.append(_rec(
                "sleep-low", "Sleep", "medium", "Aim for more regular, longer sleep",
                "Most adults do best with about 7-9 hours. Keep fixed sleep and wake times, "
                "limit screens and caffeine late in the day. If you often snore loudly, "
                "wake unrefreshed or feel very sleepy by day, mention it to a doctor.",
                f"You reported about {sleep:g} hours of sleep per day."))
        elif sleep > 9:
            recs.append(_rec(
                "sleep-high", "Sleep", "info", "Mention long sleep durations to a doctor if you feel unwell",
                "Needing much more than 9 hours can sometimes relate to health or sleep "
                "problems. If you still feel tired, a check-up is sensible.",
                f"You reported about {sleep:g} hours of sleep per day."))
        else:
            recs.append(_rec(
                "sleep-ok", "Sleep", "info", "Your sleep duration is within the typical range",
                "Keep a regular sleep schedule.",
                f"You reported about {sleep:g} hours per day (typical adult range 7-9)."))

    # ---- diet (not used by the ML model) --------------------------------------
    veg = g("fruit_veg_servings")
    if _has(veg):
        if int(veg) < 5:
            recs.append(_rec(
                "diet-veg", "Diet", "medium", "Add more vegetables and fruit",
                "Try adding a serving of vegetables to each meal and having fruit instead "
                "of packaged snacks. Pulses, whole grains and nuts are also useful.",
                f"You reported {int(veg)} servings of fruit/vegetables per day; the WHO "
                "suggests at least 400 g (about five portions) daily."))
        else:
            recs.append(_rec(
                "diet-veg-ok", "Diet", "info", "Good fruit and vegetable intake",
                "Keep it up and choose a variety of colours.",
                f"You reported {int(veg)} servings of fruit/vegetables per day."))
    fried = g("fried_food_frequency")
    if fried == "daily":
        recs.append(_rec(
            "diet-fried", "Diet", "medium", "Cut back on fried and heavily processed food",
            "Try baking, steaming or grilling, and keep fried snacks and sweets for "
            "occasional treats. Check labels for salt and added sugar.",
            "You reported eating fried or heavily processed food daily."))
    if not _has(veg) and not _has(fried):
        recs.append(_rec(
            "diet-missing", "Diet", "info", "Add diet details for more tailored tips",
            "You can include fruit/vegetable servings and fried-food frequency in your "
            "next assessment.",
            "No diet information was provided, so no diet recommendations were generated."))

    # ---- reported medical history: follow-up, never treatment advice -------------
    diabetic = g("diabetic")
    if diabetic in ("Yes", "No, borderline diabetes"):
        recs.append(_rec(
            "diabetes", "Medical follow-up", "high", "Keep up regular diabetes check-ups",
            "Attend scheduled reviews and follow the plan your doctor gave you. Do not "
            "change any medicine without medical advice.",
            "You reported diabetes or borderline diabetes." if diabetic == "Yes"
            else "You reported borderline diabetes; a doctor can advise on monitoring."))
    if g("stroke") == "Yes":
        recs.append(_rec(
            "stroke", "Medical follow-up", "high", "Stay in regular follow-up after a stroke",
            "Keep appointments with your doctor and ask what monitoring suits you.",
            "You reported a past stroke."))
    if g("kidney_disease") == "Yes":
        recs.append(_rec(
            "kidney", "Medical follow-up", "high", "Keep kidney health monitoring up to date",
            "Ask your doctor before using over-the-counter pain relievers or "
            "supplements, and attend regular check-ups.",
            "You reported kidney disease."))
    if g("asthma") == "Yes" and g("physical_activity") == "No":
        recs.append(_rec(
            "asthma-activity", "Medical follow-up", "medium", "Ask how to exercise safely with asthma",
            "Many people with asthma can be active. A doctor can help you build a plan.",
            "You reported asthma and low recent physical activity."))
    if g("skin_cancer") == "Yes":
        recs.append(_rec(
            "skin", "Medical follow-up", "medium", "Continue skin checks and sun protection",
            "Use shade, protective clothing and sunscreen, and keep dermatology follow-ups.",
            "You reported a history of skin cancer."))

    # ---- general health / wellbeing ----------------------------------------------
    if g("gen_health") in ("Fair", "Poor") or (_has(g("physical_health_days")) and int(g("physical_health_days")) >= 14):
        recs.append(_rec(
            "checkup-physical", "Medical follow-up", "high", "Book a general check-up",
            "A doctor can review blood pressure, blood sugar and cholesterol and look "
            "into what is affecting how you feel.",
            "You reported fair/poor general health or 14+ days of poor physical health in the last 30 days."))
    if _has(g("mental_health_days")) and int(g("mental_health_days")) >= 14:
        recs.append(_rec(
            "mental", "Mental wellbeing", "high", "Consider talking to someone about your mental wellbeing",
            "A doctor or counsellor can help. If you ever feel unsafe or hopeless, "
            "contact local emergency services or a crisis helpline right away.",
            "You reported 14 or more days of poor mental health in the last 30 days."))
    if g("diff_walking") == "Yes" and g("physical_activity") != "No":
        recs.append(_rec(
            "walking", "Medical follow-up", "medium", "Get advice on mobility",
            "A doctor or physiotherapist can suggest safe ways to stay active.",
            "You reported difficulty walking or climbing stairs."))

    age = g("age")
    if _has(age) and int(age) >= 40:
        recs.append(_rec(
            "age-screening", "Screening", "medium", "Keep up periodic health screening",
            "Adults over 40 are commonly advised to have blood pressure, blood sugar and "
            "cholesterol checked regularly. Ask your doctor how often is right for you.",
            f"You are {int(age)}; the chance of several conditions rises with age."))

    # ---- depends on the model output ------------------------------------------------
    if risk_category in ("elevated", "borderline"):
        recs.append(_rec(
            "model-flag", "Medical follow-up", "high", "Consider discussing your results with a doctor",
            "This tool is not a diagnosis. A clinician can measure things the tool cannot "
            "(blood pressure, cholesterol, ECG) and give personal advice.",
            "The educational model flagged your answers as resembling those of people who "
            "reported heart disease in its training data."))

    recs.append(_rec(
        "disclaimer", "About these tips", "info", "These tips are general education",
        "They are not medical advice and cannot promise to prevent any disease. Speak to "
        "a qualified healthcare professional about your own situation.",
        "Provided with every assessment."))

    recs.sort(key=lambda r: PRIORITY_ORDER[r["priority"]])
    return recs
