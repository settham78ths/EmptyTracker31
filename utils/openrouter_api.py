import os
import json
import logging
import requests
import urllib.parse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen-2.5-72b-instruct:free"

DEEP_REASONING_PROMPT = """You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://cv-optimizer-pro.repl.co/"
}

def send_api_request(prompt, max_tokens=2000, language='pl'):
    """
    Send a request to the OpenRouter API with language specification and timeout
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not found")
        raise ValueError("🔑 Brak klucza API OpenRouter! Sprawdź plik .env - powinien zawierać OPENROUTER_API_KEY=twój_klucz")
    
    if len(OPENROUTER_API_KEY) < 10:
        logger.error("OpenRouter API key seems invalid")
        raise ValueError("🔑 Klucz API OpenRouter wydaje się nieprawidłowy. Sprawdź czy jest kompletny w pliku .env")

    # Language-specific system prompts
    language_prompts = {
        'pl': "Jesteś ekspertem w optymalizacji CV i doradcą kariery. ZAWSZE odpowiadaj w języku polskim, niezależnie od języka CV lub opisu pracy. Używaj polskiej terminologii HR i poprawnej polszczyzny.",
        'en': "You are an expert resume editor and career advisor. ALWAYS respond in English, regardless of the language of the CV or job description. Use proper English HR terminology and grammar."
    }

    system_prompt = DEEP_REASONING_PROMPT + "\n" + language_prompts.get(language, language_prompts['pl'])

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        logger.debug(f"Sending request to OpenRouter API (max_tokens: {max_tokens})")
        
        # Add timeout to prevent hanging requests
        response = requests.post(
            OPENROUTER_BASE_URL, 
            headers=headers, 
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        response.raise_for_status()

        result = response.json()
        logger.debug("Received response from OpenRouter API")

        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            logger.debug(f"Response length: {len(content)} characters")
            return content
        else:
            raise ValueError("Unexpected API response format")

    except requests.exceptions.Timeout:
        logger.error("OpenRouter API request timed out")
        raise Exception("Zapytanie do AI przekroczyło limit czasu. Spróbuj z krótszym tekstem.")

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        if "429" in str(e):
            raise Exception("Przekroczono limit zapytań do AI. Spróbuj ponownie za chwilę.")
        elif "500" in str(e):
            raise Exception("Serwer AI jest tymczasowo niedostępny. Spróbuj ponownie za chwilę.")
        else:
            raise Exception(f"Błąd komunikacji z AI: {str(e)}")

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing API response: {str(e)}")
        raise Exception(f"Błąd przetwarzania odpowiedzi AI: {str(e)}")

def analyze_cv_score(cv_text, job_description="", language='pl'):
    """
    Analizuje CV i przyznaje ocenę punktową 1-100 z szczegółowym uzasadnieniem
    """
    prompt = f"""
    Przeanalizuj poniższe CV i przyznaj mu ocenę punktową od 1 do 100, gdzie:
    - 90-100: Doskonałe CV, gotowe do wysłania
    - 80-89: Bardzo dobre CV z drobnymi usprawnieniami
    - 70-79: Dobre CV wymagające kilku poprawek
    - 60-69: Przeciętne CV wymagające znaczących poprawek
    - 50-59: Słabe CV wymagające dużych zmian
    - Poniżej 50: CV wymagające całkowitego przepisania

    CV do oceny:
    {cv_text}

    {"Wymagania z oferty pracy: " + job_description if job_description else ""}

    Uwzględnij w ocenie:
    1. Strukturę i organizację treści (20 pkt)
    2. Klarowność i zwięzłość opisów (20 pkt)
    3. Dopasowanie do wymagań stanowiska (20 pkt)
    4. Obecność słów kluczowych branżowych (15 pkt)
    5. Prezentację osiągnięć i rezultatów (15 pkt)
    6. Gramatykę i styl pisania (10 pkt)

    Odpowiedź w formacie JSON:
    {{
        "score": [liczba 1-100],
        "grade": "[A+/A/B+/B/C+/C/D/F]",
        "category_scores": {{
            "structure": [1-20],
            "clarity": [1-20], 
            "job_match": [1-20],
            "keywords": [1-15],
            "achievements": [1-15],
            "language": [1-10]
        }},
        "strengths": ["punkt mocny 1", "punkt mocny 2", "punkt mocny 3"],
        "weaknesses": ["słabość 1", "słabość 2", "słabość 3"],
        "recommendations": ["rekomendacja 1", "rekomendacja 2", "rekomendacja 3"],
        "summary": "Krótkie podsumowanie oceny CV"
    }}
    """

    return send_api_request(prompt, max_tokens=2500, language=language)

def analyze_keywords_match(cv_text, job_description, language='pl'):
    """
    Analizuje dopasowanie słów kluczowych z CV do wymagań oferty pracy
    """
    if not job_description:
        return "Brak opisu stanowiska do analizy słów kluczowych."

    prompt = f"""
    Przeanalizuj dopasowanie słów kluczowych między CV a wymaganiami oferty pracy.

    CV:
    {cv_text}

    Oferta pracy:
    {job_description}

    Odpowiedź w formacie JSON:
    {{
        "match_percentage": [0-100],
        "found_keywords": ["słowo1", "słowo2", "słowo3"],
        "missing_keywords": ["brakujące1", "brakujące2", "brakujące3"],
        "recommendations": [
            "Dodaj umiejętność: [nazwa]",
            "Podkreśl doświadczenie w: [obszar]",
            "Użyj terminów branżowych: [terminy]"
        ],
        "priority_additions": ["najważniejsze słowo1", "najważniejsze słowo2"],
        "summary": "Krótkie podsumowanie analizy dopasowania"
    }}
    """

    return send_api_request(prompt, max_tokens=2000, language=language)

def check_grammar_and_style(cv_text, language='pl'):
    """
    Sprawdza gramatykę, styl i poprawność językową CV
    """
    prompt = f"""
    Przeanalizuj poniższe CV pod kątem gramatyki, stylu i poprawności językowej.

    CV:
    {cv_text}

    Sprawdź:
    1. Błędy gramatyczne i ortograficzne
    2. Spójność czasów gramatycznych
    3. Profesjonalność języka
    4. Klarowność przekazu
    5. Zgodność z konwencjami CV

    Odpowiedź w formacie JSON:
    {{
        "grammar_score": [1-10],
        "style_score": [1-10],
        "professionalism_score": [1-10],
        "errors": [
            {{"type": "gramatyka", "text": "błędny tekst", "correction": "poprawka", "line": "sekcja"}},
            {{"type": "styl", "text": "tekst do poprawy", "suggestion": "sugestia", "line": "sekcja"}}
        ],
        "style_suggestions": [
            "Użyj bardziej dynamicznych czasowników akcji",
            "Unikaj powtórzeń słów",
            "Zachowaj spójny format dat"
        ],
        "overall_quality": "ocena ogólna jakości językowej",
        "summary": "Podsumowanie analizy językowej"
    }}
    """

    return send_api_request(prompt, max_tokens=1500)

def optimize_for_position(cv_text, job_title, job_description="", language='pl'):
    """
    Optymalizuje CV pod konkretne stanowisko
    """
    prompt = f"""
    Zoptymalizuj poniższe CV specjalnie pod stanowisko: {job_title}

    CV:
    {cv_text}

    {"Wymagania z oferty: " + job_description if job_description else ""}

    Stwórz zoptymalizowaną wersję CV, która:
    1. Podkreśla najważniejsze umiejętności dla tego stanowiska
    2. Reorganizuje sekcje według priorytetów dla tej roli
    3. Dostosowuje język do branżowych standardów
    4. Maksymalizuje dopasowanie do wymagań
    5. Zachowuje autentyczność i prawdziwość informacji

    Odpowiedź w formacie JSON:
    {{
        "optimized_cv": "Zoptymalizowana wersja CV",
        "key_changes": ["zmiana 1", "zmiana 2", "zmiana 3"],
        "focus_areas": ["obszar 1", "obszar 2", "obszar 3"],
        "added_elements": ["dodany element 1", "dodany element 2"],
        "positioning_strategy": "Strategia pozycjonowania kandydata",
        "summary": "Podsumowanie optymalizacji"
    }}
    """

    return send_api_request(prompt, max_tokens=2500)

def apply_recruiter_feedback_to_cv(cv_text, recruiter_feedback, job_description="", language='pl', is_premium=False, payment_verified=False):
    """
    Apply recruiter feedback suggestions directly to the CV - PAID FEATURE
    """
    
    if is_premium:
        max_tokens = 6000
        prompt_suffix = """
        
        TRYB PREMIUM - ZASTOSOWANIE POPRAWEK REKRUTERA:
        - Implementuj WSZYSTKIE sugestie z opinii rekrutera
        - Przepisz CV zgodnie z każdą rekomendacją
        - Stwórz profesjonalną, dopracowaną wersję
        - Użyj zaawansowanych technik optymalizacji
        - Dodaj elementy, które rekruter zasugerował
        """
    elif payment_verified:
        max_tokens = 4000  
        prompt_suffix = """
        
        TRYB PŁATNY - ZASTOSOWANIE POPRAWEK:
        - Implementuj główne sugestie z opinii rekrutera
        - Przepisz sekcje zgodnie z rekomendacjami
        - Popraw strukturę i formatowanie
        - Dodaj brakujące elementy wskazane przez rekrutera
        """
    else:
        # Nie powinno się zdarzyć dla tej funkcji - tylko płatne
        return "Ta funkcja wymaga płatności."

    prompt = f"""
    ZADANIE: Zastosuj konkretne poprawki z opinii rekrutera do CV, tworząc ulepszoną wersję.

    ORYGINALNY TEKST CV:
    {cv_text}

    OPINIA REKRUTERA DO ZASTOSOWANIA:
    {recruiter_feedback}

    {f"KONTEKST STANOWISKA: {job_description}" if job_description else ""}

    INSTRUKCJE:
    1. Przeanalizuj każdą sugestię z opinii rekrutera
    2. Zastosuj WSZYSTKIE wskazane poprawki do CV
    3. Przepisz sekcje zgodnie z rekomendacjami
    4. Dodaj brakujące elementy, które rekruter zasugerował
    5. Popraw strukturę, formatowanie i język według wskazówek
    6. Zachowaj autentyczność danych - NIE dodawaj fałszywych informacji
    7. Jeśli rekruter sugerował dodanie czegoś czego nie ma w CV, dodaj sekcję z informacją "Do uzupełnienia"

    UWAGA: Używaj WYŁĄCZNIE prawdziwych informacji z oryginalnego CV. Jeśli rekruter sugeruje dodanie czegoś czego nie ma, zaznacz to jako "Do uzupełnienia przez użytkownika".

    {prompt_suffix}

    Zwróć poprawione CV w formacie JSON:
    {{
        "improved_cv": "Poprawiona wersja CV z zastosowanymi sugestiami rekrutera",
        "applied_changes": ["lista zastosowanych zmian"],
        "sections_to_complete": ["sekcje do uzupełnienia przez użytkownika"],
        "improvement_summary": "Podsumowanie wprowadzonych ulepszeń"
    }}
    """

    return send_api_request(prompt, max_tokens=max_tokens, language=language)

def generate_interview_tips(cv_text, job_description="", language='pl'):
    """
    Generuje spersonalizowane tipy na rozmowę kwalifikacyjną
    """
    prompt = f"""
    Na podstawie CV i opisu stanowiska, przygotuj spersonalizowane tipy na rozmowę kwalifikacyjną.

    CV:
    {cv_text}

    {"Stanowisko: " + job_description if job_description else ""}

    Odpowiedź w formacie JSON:
    {{
        "preparation_tips": [
            "Przygotuj się na pytanie o [konkretny aspekt z CV]",
            "Przećwicz opowiadanie o projekcie [nazwa projektu]",
            "Badź gotowy na pytania techniczne o [umiejętność]"
        ],
        "strength_stories": [
            {{"strength": "umiejętność", "story_outline": "jak opowiedzieć o sukcesie", "example": "konkretny przykład z CV"}},
            {{"strength": "osiągnięcie", "story_outline": "struktura opowieści", "example": "przykład z doświadczenia"}}
        ],
        "weakness_preparation": [
            {{"potential_weakness": "obszar do poprawy", "how_to_address": "jak to przedstawić pozytywnie"}},
            {{"potential_weakness": "luka w CV", "how_to_address": "jak wytłumaczyć"}}
        ],
        "questions_to_ask": [
            "Przemyślane pytanie o firmę/zespół",
            "Pytanie o rozwój w roli",
            "Pytanie o wyzwania stanowiska"
        ],
        "research_suggestions": [
            "Sprawdź informacje o: [aspekt firmy]",
            "Poznaj ostatnie projekty firmy",
            "Zbadaj kulturę organizacyjną"
        ],
        "summary": "Kluczowe rady dla tego kandydata"
    }}
    """

    return send_api_request(prompt, max_tokens=2000)

def analyze_polish_job_posting(job_description, language='pl'):
    """
    Analizuje polskie ogłoszenia o pracę i wyciąga kluczowe informacje
    """
    prompt = f"""
    Przeanalizuj poniższe polskie ogłoszenie o pracę i wyciągnij z niego najważniejsze informacje.

    OGŁOSZENIE O PRACĘ:
    {job_description}

    Wyciągnij i uporządkuj następujące informacje:

    1. PODSTAWOWE INFORMACJE:
    - Stanowisko/pozycja
    - Branża/sektor
    - Lokalizacja pracy
    - Typ umowy/zatrudnienia

    2. WYMAGANIA KLUCZOWE:
    - Wykształcenie
    - Doświadczenie zawodowe
    - Specyficzne umiejętności techniczne
    - Uprawnienia/certyfikaty (np. prawo jazdy, kursy)
    - Umiejętności miękkie

    3. OBOWIĄZKI I ZAKRES PRACY:
    - Główne zadania
    - Odpowiedzialności
    - Specyficzne czynności

    4. WARUNKI PRACY:
    - Godziny pracy
    - System pracy (pełny etat, zmianowy, weekendy)
    - Wynagrodzenie (jeśli podane)
    - Benefity i dodatki

    5. SŁOWA KLUCZOWE BRANŻOWE:
    - Terminologia specjalistyczna
    - Najważniejsze pojęcia z ogłoszenia
    - Frazy które powinny pojawić się w CV

    Odpowiedź w formacie JSON:
    {{
        "job_title": "dokładny tytuł stanowiska",
        "industry": "branża/sektor",
        "location": "lokalizacja",
        "employment_type": "typ zatrudnienia",
        "key_requirements": [
            "wymóg 1",
            "wymóg 2", 
            "wymóg 3"
        ],
        "main_responsibilities": [
            "obowiązek 1",
            "obowiązek 2",
            "obowiązek 3"
        ],
        "technical_skills": [
            "umiejętność techniczna 1",
            "umiejętność techniczna 2"
        ],
        "soft_skills": [
            "umiejętność miękka 1",
            "umiejętność miękka 2"
        ],
        "work_conditions": {{
            "hours": "godziny pracy",
            "schedule": "harmonogram",
            "salary_info": "informacje o wynagrodzeniu",
            "benefits": ["benefit 1", "benefit 2"]
        }},
        "industry_keywords": [
            "słowo kluczowe 1",
            "słowo kluczowe 2",
            "słowo kluczowe 3",
            "słowo kluczowe 4",
            "słowo kluczowe 5"
        ],
        "critical_phrases": [
            "kluczowa fraza 1",
            "kluczowa fraza 2",
            "kluczowa fraza 3"
        ],
        "experience_level": "poziom doświadczenia",
        "education_requirements": "wymagane wykształcenie",
        "summary": "zwięzłe podsumowanie stanowiska i wymagań"
    }}
    """

    return send_api_request(prompt, max_tokens=2000, language=language)

def optimize_cv_for_specific_position(cv_text, target_position, job_description, company_name="", language='pl', is_premium=False, payment_verified=False):
    """
    ZAAWANSOWANA OPTYMALIZACJA CV - analizuje każde poprzednie stanowisko i inteligentnie je przepisuje
    pod kątem konkretnego stanowiska docelowego, zachowując pełną autentyczność danych
    """
    # Najpierw przeanalizuj opis stanowiska jeśli został podany
    job_analysis = None
    if job_description and len(job_description) > 50:
        try:
            job_analysis_result = analyze_polish_job_posting(job_description, language)
            job_analysis = parse_ai_json_response(job_analysis_result)
        except Exception as e:
            logger.warning(f"Nie udało się przeanalizować opisu stanowiska: {e}")

    prompt = f"""
    ZADANIE EKSPERCKIE: Przeprowadź zaawansowaną analizę CV i stwórz precyzyjną optymalizację pod konkretne polskie stanowisko pracy.

    ⚠️ ABSOLUTNE ZASADY BEZPIECZEŃSTWA:
    1. ❌ ZAKAZ WYMYŚLANIA: Używaj WYŁĄCZNIE faktów z oryginalnego CV
    2. ❌ ZAKAZ DODAWANIA: Nie twórz nowych firm, dat, projektów, osiągnięć
    3. ✅ INTELIGENTNE PRZEPISYWANIE: Przemyślnie sformułuj istniejące doświadczenia
    4. ✅ KONTEKSTOWE DOPASOWANIE: Podkreśl aspekty każdego stanowiska, które są relevant dla celu
    5. ✅ POLSKI RYNEK PRACY: Dostosuj terminologię do polskich standardów HR
    6. ✅ UNIKALNE OPISY: Jeśli w CV są podobne stanowiska (np. "Kurier" w różnych firmach), stwórz RÓŻNE opisy dla każdego z nich

    🎯 STANOWISKO DOCELOWE: {target_position}
    🏢 FIRMA DOCELOWA: {company_name}
    📋 OPIS STANOWISKA:
    {job_description}

    {"📊 ANALIZA STANOWISKA AI:" + str(job_analysis) if job_analysis else ""}

    STRATEGIA OPTYMALIZACJI - KROK PO KROKU:

    KROK 1: DEEP ANALYSIS
    Przeanalizuj każde poprzednie stanowisko z CV i zidentyfikuj:
    - Umiejętności transferowalne na stanowisko docelowe
    - Doświadczenia, które można przeformułować jako relevant
    - Obowiązki, które mają wspólne elementy z requirements
    - Branżowe słowa kluczowe do wykorzystania
    - UWAGA: Jeśli są podobne stanowiska - znajdź różne aspekty każdego z nich

    KROK 2: STRATEGIC REPOSITIONING  
    Dla każdego poprzedniego stanowiska:
    - Zachowaj oryginalne dane (firma, daty, tytuł)
    - Przepisz opisy obowiązków z perspektywą docelowego stanowiska
    - Użyj terminologii branżowej właściwej dla target position
    - Podkreśl soft skills i hard skills pasujące do requirements
    - KLUCZOWE: Dla podobnych stanowisk stwórz RÓŻNE opisy skupiające się na innych aspektach

    KROK 3: INTELLIGENT ENHANCEMENT
    - Stwórz podsumowanie zawodowe pozycjonujące kandydata na target role
    - Zorganizuj umiejętności według ważności dla docelowego stanowiska
    - Dostosuj język i styl do branży docelowej firmy
    - Zoptymalizuj pod kątem ATS keywords z job description

    PRZYKŁADY INTELIGENTNEGO PRZEPISYWANIA DLA POLSKIEGO RYNKU PRACY:

    STANOWISKO DOCELOWE: "Kierowca kat. B - bramowiec"
    Oryginał: "Kierowca - przewożenie towarów"
    ✅ Zoptymalizowane: "Realizowałem transport kontenerów i odpadów budowlanych, dbając o terminowość dostaw i bezpieczeństwo przewozu"

    STANOWISKO DOCELOWE: "Specjalista ds. logistyki"
    Oryginał: "Pracownik magazynu - obsługa towaru" 
    ✅ Zoptymalizowane: "Koordynowałem procesy magazynowe, optymalizowałem przepływy towarów i zarządzałem dokumentacją logistyczną"

    PRZYKŁAD RÓŻNICOWANIA PODOBNYCH STANOWISK:
    
    STANOWISKO 1: "Kurier - DHL" (2022-2023)
    ✅ Opis 1: "Wykonywałem ekspresowe dostawy międzynarodowe, obsługiwałem system śledzenia przesyłek i zapewniałem terminowość dostaw zgodnie z procedurami DHL"
    
    STANOWISKO 2: "Kurier - DPD" (2021-2022)  
    ✅ Opis 2: "Realizowałem dostawy lokalne na terenie miasta, utrzymywałem kontakt z klientami i optymalizowałem trasy dostaw dla maksymalnej efektywności"
    
    STANOWISKO 3: "Kurier - UPS" (2020-2021)
    ✅ Opis 3: "Odpowiadałem za dostawy biznesowe do firm, zarządzałem dokumentacją celną przesyłek zagranicznych i współpracowałem z działem obsługi klienta"

    ORYGINALNE CV DO ANALIZY:
    {cv_text}

    SZCZEGÓLNE UWAGI DLA PODOBNYCH STANOWISK:
    - Jeśli w CV są stanowiska o podobnych nazwach (np. "Kurier", "Kierowca", "Sprzedawca") w różnych firmach lub okresach
    - Stwórz dla każdego z nich UNIKALNY opis, który podkreśla RÓŻNE aspekty pracy
    - Wykorzystaj specyfikę każdej firmy (np. DHL = międzynarodowe, DPD = lokalne, UPS = biznesowe)
    - Użyj różnych terminów branżowych i skupiaj się na innych umiejętnościach dla każdego stanowiska

    WYGENERUJ ZAAWANSOWANE CV WEDŁUG SCHEMATU:

    {{
        "position_analysis": {{
            "target_role": "{target_position}",
            "key_requirements_identified": ["requirement1", "requirement2", "requirement3"],
            "transferable_skills_found": ["skill1", "skill2", "skill3"],
            "positioning_strategy": "Jak pozycjonujemy kandydata",
            "similar_positions_identified": ["lista podobnych stanowisk z CV jeśli są"],
            "differentiation_strategy": "Jak zróżnicować opisy podobnych stanowisk"
        }},
        "experience_optimization": {{
            "previous_position_1": {{
                "original_description": "Oryginalne zadania z CV",
                "optimized_description": "Przepisane zadania pod target position",
                "relevance_connection": "Dlaczego to pasuje do target role",
                "uniqueness_factor": "Jak ten opis różni się od innych podobnych stanowisk"
            }},
            "previous_position_2": {{
                "original_description": "Oryginalne zadania z CV", 
                "optimized_description": "Przepisane zadania pod target position",
                "relevance_connection": "Dlaczego to pasuje do target role",
                "uniqueness_factor": "Jak ten opis różni się od innych podobnych stanowisk"
            }}
        }},
        "optimized_cv": "KOMPLETNE ZOPTYMALIZOWANE CV gotowe do wysłania",
        "keyword_optimization": {{
            "primary_keywords": ["kluczowe słowo1", "kluczowe słowo2"],
            "secondary_keywords": ["dodatkowe słowo1", "dodatkowe słowo2"],
            "keyword_density_score": "[0-100]"
        }},
        "ats_compatibility": {{
            "score": "[0-100]",
            "structure_optimization": "Jak zoptymalizowano strukturę",
            "format_improvements": "Jakie poprawki formatowania"
        }},
        "competitive_advantage": {{
            "unique_selling_points": ["USP1", "USP2", "USP3"],
            "differentiation_strategy": "Jak kandydat wyróżnia się na tle konkurencji",
            "value_proposition": "Główna wartość jaką wnosi kandydat"
        }},
        "improvement_summary": {{
            "before_vs_after": "Podsumowanie zmian",
            "match_percentage": "[0-100]",
            "success_probability": "Szanse powodzenia aplikacji",
            "next_steps": "Rekomendacje dla kandydata",
            "position_diversity": "Jak zróżnicowano opisy podobnych stanowisk"
        }}
    }}
    """

    # Zwiększone limity tokenów dla zaawansowanej analizy
    if is_premium or payment_verified:
        max_tokens = 8000  # Maksymalny limit dla pełnej analizy
        prompt += f"""

        🔥 TRYB PREMIUM - MAKSYMALNA OPTYMALIZACJA:
        - Analizuj każde słowo z CV pod kątem potential value
        - Stwórz 7-10 bullet points dla każdego stanowiska
        - Dodaj section "Key Achievements" z reframed accomplishments  
        - Zoptymalizuj pod specific industry terminology
        - Przygotuj CV na poziomie executive search firm
        - Zastosuj advanced psychological positioning techniques
        - Stwórz compelling narrative arc w karrierze kandydata
        """
    else:
        max_tokens = 4000
        prompt += f"""

        TRYB STANDARD - PROFESJONALNA OPTYMALIZACJA:
        - Przepisz 3-5 bullet points dla każdego stanowiska
        - Dodaj professional summary section
        - Zoptymalizuj basic keyword matching
        - Popraw overall structure i readability
        """

    return send_api_request(prompt, max_tokens=max_tokens, language=language)

def generate_complete_cv_content(target_position, experience_level, industry, brief_background, language='pl'):
    """
    Generate complete CV content from minimal user input using AI
    """
    prompt = f"""
    ZADANIE: Wygeneruj kompletną treść CV na podstawie minimalnych informacji od użytkownika.

    DANE WEJŚCIOWE:
    - Docelowe stanowisko: {target_position}
    - Poziom doświadczenia: {experience_level} (junior/mid/senior)
    - Branża: {industry}
    - Krótki opis doświadczenia: {brief_background}

    WYGENERUJ REALISTYCZNĄ TREŚĆ CV:

    1. PROFESSIONAL SUMMARY (80-120 słów):
    - Stwórz przekonujące podsumowanie zawodowe
    - Dopasowane do poziomu doświadczenia i stanowiska
    - Użyj słów kluczowych z branży

    2. DOŚWIADCZENIE ZAWODOWE (3-4 stanowiska):
    - Wygeneruj realistyczne stanowiska progresywne w karierze
    - Każde stanowisko: tytuł, firma (prawdopodobna nazwa), okres, 3-4 obowiązki
    - Dostosuj do poziomu experience_level:
      * Junior: 1-2 lata doświadczenia, podstawowe role
      * Mid: 3-5 lat, stanowiska specjalistyczne
      * Senior: 5+ lat, role kierownicze/eksperckie

    3. WYKSZTAŁCENIE:
    - Wygeneruj odpowiednie wykształcenie dla branży
    - Kierunek studiów pasujący do stanowiska
    - Realistyczne nazwy uczelni (polskie)

    4. UMIEJĘTNOŚCI:
    - Lista 8-12 umiejętności kluczowych dla stanowiska
    - Mix hard skills i soft skills
    - Aktualne technologie/narzędzia branżowe

    WYMAGANIA JAKOŚCI:
    - Treść musi być realistyczna i wiarygodna
    - Używaj polskiej terminologii HR
    - Dostosuj język do poziomu stanowiska
    - Wszystkie informacje muszą być spójne logicznie

    PRZYKŁADY PROGRESJI KARIERY:

    JUNIOR LEVEL:
    - Praktykant/Stażysta → Młodszy Specjalista → Specjalista

    MID LEVEL:  
    - Specjalista → Starszy Specjalista → Kierownik Zespołu

    SENIOR LEVEL:
    - Kierownik → Menedżer → Dyrektor/Kierownik Działu

    Odpowiedź w formacie JSON:
    {{
        "professional_title": "Tytuł zawodowy do CV",
        "professional_summary": "Podsumowanie zawodowe 80-120 słów",
        "experience_suggestions": [
            {{
                "title": "Stanowisko",
                "company": "Nazwa firmy", 
                "startDate": "2022-01",
                "endDate": "obecnie",
                "description": "Opis obowiązków i osiągnięć (3-4 punkty)"
            }},
            {{
                "title": "Poprzednie stanowisko",
                "company": "Poprzednia firma",
                "startDate": "2020-06", 
                "endDate": "2021-12",
                "description": "Opis obowiązków z poprzedniej pracy"
            }}
        ],
        "education_suggestions": [
            {{
                "degree": "Kierunek studiów",
                "school": "Nazwa uczelni",
                "startYear": "2018",
                "endYear": "2022"
            }}
        ],
        "skills_list": "Umiejętność 1, Umiejętność 2, Umiejętność 3, Umiejętność 4, Umiejętność 5, Umiejętność 6, Umiejętność 7, Umiejętność 8",
        "career_level": "{experience_level}",
        "industry_focus": "{industry}",
        "generation_notes": "Informacje o logice generowania tego CV"
    }}
    """

    return send_api_request(prompt, max_tokens=4000, language=language)

def optimize_cv(cv_text, job_description, language='pl', is_premium=False, payment_verified=False):
    """
    Create an optimized version of CV using ONLY authentic data from the original CV
    Premium users get extended token limits for more detailed CV generation
    """
    prompt = f"""
    ZADANIE: Automatycznie rozpoznaj branżę/sektor na podstawie CV, a następnie zoptymalizuj je pod kątem tej branży i stanowiska, używając WYŁĄCZNIE prawdziwych informacji z oryginalnego CV.

    ABSOLUTNE ZASADY - PRZECZYTAJ UWAŻNIE:
    1. ❌ ZAKAZ WYMYŚLANIA: NIE dodawaj żadnych nowych informacji, firm, stanowisk, dat, liczb, procent, osiągnięć
    2. ❌ ZAKAZ DODAWANIA: NIE twórz nowych umiejętności, certyfikatów, projektów czy doświadczeń
    3. ✅ DOZWOLONE: Tylko przepisanie istniejących informacji z CV w bardziej profesjonalny sposób
    4. ✅ DOZWOLONE: Uporządkowanie i lepsze sformułowanie tego, co już jest w CV
    5. ✅ DOSTOSOWANIE BRANŻOWE: Użyj terminologii i stylu właściwego dla rozpoznanej branży

    KROK 1 - ROZPOZNANIE BRANŻY:
    Na podstawie doświadczenia zawodowego, umiejętności i wykształcenia określ główną branżę/sektor:
    - IT/Technologie (programowanie, systemy, data science)
    - Finanse/Bankowość (księgowość, analizy finansowe, banking)
    - Medycyna/Zdrowie (opieka zdrowotna, farmacja, medycyna)
    - Edukacja (nauczanie, szkolenia, rozwój)
    - Marketing/Sprzedaż (digital marketing, sprzedaż, reklama)
    - Logistyka/Transport (łańcuch dostaw, kurierzy, magazyny)
    - Inżynieria/Produkcja (mechanical, electrical, civil engineering)
    - HR/Zarządzanie (kadry, zarządzanie projektami)
    - Inne (określ konkretnie)

    INSTRUKCJE SZCZEGÓŁOWE Z KONTEKSTEM BRANŻOWYM:

    1. DANE OSOBOWE:
       - Przepisz dokładnie imię, nazwisko i kontakt z oryginalnego CV
       - NIE dodawaj nowych informacji kontaktowych

    2. PODSUMOWANIE ZAWODOWE:
       - Napisz krótkie podsumowanie bazując TYLKO na doświadczeniu z CV
       - Użyj tylko tych umiejętności, które są faktycznie wymienione w CV
       - NIE wymyślaj branży ani specjalizacji, których nie ma w oryginale

    3. DOŚWIADCZENIE ZAWODOWE:
       - Przepisz każde stanowisko DOKŁADNIE jak w oryginale (firma, stanowisko, daty)
       - Dla każdego miejsca pracy napisz 3-4 punkty opisujące obowiązki
       - Bazuj punkty TYLKO na informacjach z oryginalnego CV
       - Użyj profesjonalnych czasowników (planowałem, koordynowałem, obsługiwałem)
       - NIE dodawaj liczb, procent ani wymyślonych osiągnięć
       - KLUCZOWE: Jeśli są podobne stanowiska (np. "Kurier" w różnych firmach), stwórz RÓŻNE opisy dla każdego z nich:
         * Skupiaj się na różnych aspektach tej samej pracy
         * Użyj specyfiki każdej firmy (DHL = międzynarodowe, DPD = lokalne, UPS = biznesowe)
         * Zastosuj różne terminy branżowe dla każdego stanowiska
         * Podkreśl inne umiejętności w każdym opisie

    4. UMIEJĘTNOŚCI:
       - Przepisz TYLKO te umiejętności, które są wymienione w oryginalnym CV
       - Pogrupuj je w kategorie (np. Techniczne, Komunikacyjne, Zawodowe)
       - NIE dodawaj nowych umiejętności

    5. WYKSZTAŁCENIE:
       - Przepisz dokładnie informacje o wykształceniu z CV
       - NIE dodawaj kursów czy certyfikatów, których nie ma w oryginale

    6. DOPASOWANIE DO STANOWISKA:
       - Wyeksponuj te elementy z CV, które pasują do opisu stanowiska
       - Ale NIE dodawaj nowych elementów - tylko lepiej opisuj istniejące

    PRZYKŁAD DOBREGO PODEJŚCIA:
    Oryginał: "Dostarczanie paczek"
    Zoptymalizowane: "Realizowałem terminowe dostawy paczek do klientów"

    PRZYKŁAD ZŁEGO PODEJŚCIA (NIE RÓB TEGO):
    ❌ "Zwiększyłem efektywność dostaw o 25%" (gdy nie ma takiej informacji w CV)
    ❌ "Zarządzałem zespołem 5 kurierów" (gdy nie ma takiej informacji w CV)

    PRZYKŁADY RÓŻNICOWANIA PODOBNYCH STANOWISK:
    
    Jeśli w CV są 3 stanowiska "Kurier" w różnych firmach:
    
    STANOWISKO 1: "Kurier - DHL" (2022-2023)
    ✅ Opis 1: "Wykonywałem ekspresowe dostawy międzynarodowe, obsługiwałem system śledzenia przesyłek i zapewniałem terminowość dostaw zgodnie z procedurami DHL"
    
    STANOWISKO 2: "Kurier - DPD" (2021-2022)  
    ✅ Opis 2: "Realizowałem dostawy lokalne na terenie miasta, utrzymywałem kontakt z klientami i optymalizowałem trasy dostaw dla maksymalnej efektywności"
    
    STANOWISKO 3: "Kurier - UPS" (2020-2021)
    ✅ Opis 3: "Odpowiadałem za dostawy biznesowe do firm, zarządzałem dokumentacją celną przesyłek zagranicznych i współpracowałem z działem obsługi klienta"

    KLUCZOWE ZASADY RÓŻNICOWANIA:
    1. Użyj różnych aspektów tej samej pracy (międzynarodowe vs lokalne vs biznesowe)
    2. Podkreśl specyfikę każdej firmy (DHL = ekspresowe, DPD = lokalne, UPS = biznesowe)
    3. Zastosuj różne terminy branżowe dla każdego stanowiska
    4. Skupiaj się na innych umiejętnościach w każdym opisie

    ORYGINALNE CV:
    {cv_text}

    STANOWISKO, DO KTÓREGO OPTYMALIZUJEMY:
    {job_description}

    WYGENERUJ: Profesjonalne CV używając WYŁĄCZNIE informacji z oryginalnego CV, bez dodawania żadnych wymyślonych elementów.

    ODPOWIEDŹ W FORMACIE JSON:
    {{
        "detected_industry": "rozpoznana branża/sektor",
        "industry_keywords": ["słowo kluczowe 1", "słowo kluczowe 2", "słowo kluczowe 3"],
        "optimized_cv": "Kompletne zoptymalizowane CV z branżowym dostosowaniem",
        "key_improvements": [
            "Dostosowano terminologię do branży [nazwa branży]",
            "Podkreślono kluczowe umiejętności [konkretne umiejętności]",
            "Wzmocniono sekcję [nazwa sekcji]"
        ],
        "ats_compatibility_score": "[0-100]",
        "job_match_score": "[0-100]",
        "positioning_strategy": "Jak kandydat jest pozycjonowany w branży",
        "summary": "Zwięzłe podsumowanie wprowadzonych ulepszeń"
    }}"""

    # Rozszerzony limit tokenów dla płacących użytkowników
    if is_premium or payment_verified:
        # Płacący użytkownicy - znacznie rozszerzony limit tokenów
        max_tokens = 6000  # Bardzo duży limit dla kompletnego CV
        prompt += f"""

        INSTRUKCJE PREMIUM - PEŁNE CV:
        - Stwórz szczegółowe opisy każdego stanowiska (5-8 punktów na pozycję)
        - Dodaj rozbudowane podsumowanie zawodowe z kluczowymi osiągnięciami
        - Rozwiń każdą sekcję umiejętności z precyzyjnymi opisami
        - Zastosuj zaawansowane formatowanie profesjonalnego CV
        - Użyj branżowej terminologii i zaawansowanego języka biznesowego
        - Stwórz CV gotowe do wysłania do najlepszych firm i korporacji
        - Wykorzystaj pełny potencjał każdej informacji z oryginalnego CV
        """
    else:
        # Bezpłatni użytkownicy - podstawowy limit
        max_tokens = 3000  # Zwiększony z 2500 dla lepszej jakości
        prompt += f"""

        INSTRUKCJE STANDARD:
        - Stwórz solidną optymalizację CV (3-4 punkty na pozycję)
        - Dodaj profesjonalne podsumowanie zawodowe
        - Uporządkuj umiejętności w logiczne kategorie
        - Zastosuj czytelne i spójne formatowanie
        """

    return send_api_request(prompt, max_tokens=max_tokens, language=language)

def generate_recruiter_feedback(cv_text, job_description="", language='pl'):
    """
    Generate feedback on a CV as if from an AI recruiter
    """
    context = ""
    if job_description:
        context = f"Opis stanowiska do kontekstu:\n{job_description}"

    prompt = f"""
    ZADANIE: Jesteś doświadczonym rekruterem. Przeanalizuj to CV i udziel szczegółowej, konstruktywnej opinii w języku polskim.

    ⚠️ KLUCZOWE: Oceniaj TYLKO to co faktycznie jest w CV. NIE ZAKŁADAJ, NIE DOMYŚLAJ się i NIE DODAWAJ informacji, których tam nie ma.

    Uwzględnij w ocenie:
    1. Ogólne wrażenie i pierwsza reakcja na podstawie faktycznej treści CV
    2. Mocne strony i słabości wynikające z konkretnych informacji w CV
    3. Ocena formatowania i struktury CV
    4. Jakość treści i sposób prezentacji faktycznych doświadczeń
    5. Kompatybilność z systemami ATS
    6. Konkretne sugestie poprawek oparte na tym co jest w CV
    7. Ocena ogólna w skali 1-10
    8. Prawdopodobieństwo zaproszenia na rozmowę

    {context}

    CV do oceny:
    {cv_text}

    Odpowiedź w formacie JSON:
    {{
        "overall_impression": "Pierwsze wrażenie oparte na faktycznej treści CV",
        "rating": [1-10],
        "strengths": [
            "Mocna strona 1 (konkretnie z CV)",
            "Mocna strona 2 (konkretnie z CV)", 
            "Mocna strona 3 (konkretnie z CV)"
        ],
        "weaknesses": [
            "Słabość 1 z sugestią poprawy (bazując na CV)",
            "Słabość 2 z sugestią poprawy (bazując na CV)",
            "Słabość 3 z sugestią poprawy (bazując na CV)"
        ],
        "formatting_assessment": "Ocena layoutu, struktury i czytelności faktycznej treści",
        "content_quality": "Ocena jakości treści rzeczywiście obecnej w CV",
        "ats_compatibility": "Czy CV przejdzie przez systemy automatycznej selekcji",
        "specific_improvements": [
            "Konkretna poprawa 1 (oparta na faktach z CV)",
            "Konkretna poprawa 2 (oparta na faktach z CV)",
            "Konkretna poprawa 3 (oparta na faktach z CV)"
        ],
        "interview_probability": "Prawdopodobieństwo zaproszenia oparte na faktach z CV",
        "recruiter_summary": "Podsumowanie z perspektywy rekrutera - tylko fakty z CV"
    }}

    Bądź szczery, ale konstruktywny. Oceniaj tylko to co rzeczywiście jest w CV, nie dodawaj od siebie.
    """

    return send_api_request(prompt, max_tokens=2000)

def generate_cover_letter(cv_text, job_description, language='pl'):
    """
    Generate a cover letter based on a CV and job description
    """
    prompt = f"""
    ZADANIE: Napisz spersonalizowany list motywacyjny w języku polskim WYŁĄCZNIE na podstawie faktów z CV.

    ⚠️ ABSOLUTNE WYMAGANIA:
    - Używaj TYLKO informacji faktycznie obecnych w CV
    - NIE WYMYŚLAJ doświadczeń, projektów, osiągnięć ani umiejętności
    - NIE DODAWAJ informacji, których nie ma w oryginalnym CV
    - Jeśli w CV brakuje jakichś informacji - nie uzupełniaj ich

    List motywacyjny powinien:
    - Być profesjonalnie sformatowany
    - Podkreślać umiejętności i doświadczenia faktycznie wymienione w CV
    - Łączyć prawdziwe doświadczenie kandydata z wymaganiami stanowiska
    - Zawierać przekonujące wprowadzenie oparte na faktach z CV
    - Mieć około 300-400 słów
    - Być napisany naturalnym, profesjonalnym językiem polskim

    Struktura listu:
    1. Nagłówek z danymi kontaktowymi
    2. Zwrot do adresata
    3. Wprowadzenie - dlaczego aplikujesz
    4. Główna treść - dopasowanie doświadczenia do wymagań
    5. Zakończenie z wyrażeniem zainteresowania
    6. Pozdrowienia

    Opis stanowiska:
    {job_description}

    CV kandydata:
    {cv_text}

    Napisz kompletny list motywacyjny w języku polskim. Użyj profesjonalnego, ale ciepłego tonu.
    """

    return send_api_request(prompt, max_tokens=2000)

def analyze_job_url(url):
    """
    Extract job description from a URL with improved handling for popular job sites
    """
    try:
        logger.debug(f"Analyzing job URL: {url}")

        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")

        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        job_text = ""
        domain = parsed_url.netloc.lower()

        if 'linkedin.com' in domain:
            containers = soup.select('.description__text, .show-more-less-html, .jobs-description__content')
            if containers:
                job_text = containers[0].get_text(separator='\n', strip=True)

        elif 'indeed.com' in domain:
            container = soup.select_one('#jobDescriptionText')
            if container:
                job_text = container.get_text(separator='\n', strip=True)

        elif 'pracuj.pl' in domain:
            containers = soup.select('[data-test="section-benefit-expectations-text"], [data-test="section-description-text"]')
            if containers:
                job_text = '\n'.join([c.get_text(separator='\n', strip=True) for c in containers])

        elif 'olx.pl' in domain or 'praca.pl' in domain:
            containers = soup.select('.offer-description, .offer-content, .description')
            if containers:
                job_text = containers[0].get_text(separator='\n', strip=True)

        if not job_text:
            potential_containers = soup.select('.job-description, .description, .details, article, .job-content, [class*=job], [class*=description], [class*=offer]')
            if potential_containers:
                for container in potential_containers:
                    container_text = container.get_text(separator='\n', strip=True)
                    if len(container_text) > len(job_text):
                        job_text = container_text

            if not job_text and soup.body:
                for tag in soup.select('nav, header, footer, script, style, iframe'):
                    tag.decompose()

                job_text = soup.body.get_text(separator='\n', strip=True)

                if len(job_text) > 10000:
                    paragraphs = job_text.split('\n')
                    keywords = ['requirements', 'responsibilities', 'qualifications', 'skills', 'experience', 'about the job',
                                'wymagania', 'obowiązki', 'kwalifikacje', 'umiejętności', 'doświadczenie', 'o pracy']

                    relevant_paragraphs = []
                    found_relevant = False

                    for paragraph in paragraphs:
                        if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                            found_relevant = True
                        if found_relevant and len(paragraph.strip()) > 50:
                            relevant_paragraphs.append(paragraph)

                    if relevant_paragraphs:
                        job_text = '\n'.join(relevant_paragraphs)

        job_text = '\n'.join([' '.join(line.split()) for line in job_text.split('\n') if line.strip()])

        if not job_text:
            raise ValueError("Could not extract job description from the URL")

        logger.debug(f"Successfully extracted job description from URL")

        if len(job_text) > 4000:
            logger.debug(f"Job description is long ({len(job_text)} chars), summarizing with AI")
            job_text = summarize_job_description(job_text)

        return job_text

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching job URL: {str(e)}")
        raise Exception(f"Failed to fetch job posting from URL: {str(e)}")

    except Exception as e:
        logger.error(f"Error analyzing job URL: {str(e)}")
        raise Exception(f"Failed to analyze job posting: {str(e)}")

def summarize_job_description(job_text):
    """
    Summarize a long job description using the AI
    """
    prompt = f"""
    ZADANIE: Wyciągnij i podsumuj kluczowe informacje z tego ogłoszenia o pracę w języku polskim.

    Uwzględnij:
    1. Stanowisko i nazwa firmy (jeśli podane)
    2. Wymagane umiejętności i kwalifikacje
    3. Obowiązki i zakres zadań
    4. Preferowane doświadczenie
    5. Inne ważne szczegóły (benefity, lokalizacja, itp.)
    6. TOP 5 słów kluczowych krytycznych dla tego stanowiska

    Tekst ogłoszenia:
    {job_text[:4000]}...

    Stwórz zwięzłe ale kompletne podsumowanie tego ogłoszenia, skupiając się na informacjach istotnych dla optymalizacji CV.
    Na końcu umieść sekcję "KLUCZOWE SŁOWA:" z 5 najważniejszymi terminami.

    Odpowiedź w języku polskim.
    """

    return send_api_request(prompt, max_tokens=1500)

def ats_optimization_check(cv_text, job_description="", language='pl'):
    """
    Check CV against ATS (Applicant Tracking System) and provide suggestions for improvement
    """
    context = ""
    if job_description:
        context = f"Ogłoszenie o pracę dla odniesienia:\n{job_description[:2000]}"

    prompt = f"""
    TASK: Przeprowadź dogłębną analizę CV pod kątem kompatybilności z systemami ATS (Applicant Tracking System) i wykryj potencjalne problemy.

    Przeprowadź następujące analizy:

    1. WYKRYWANIE PROBLEMÓW STRUKTURALNYCH:
       - Znajdź sekcje, które są w nieodpowiednich miejscach (np. doświadczenie zawodowe w sekcji zainteresowań)
       - Wskaż niespójności w układzie i formatowaniu
       - Zidentyfikuj zduplikowane informacje w różnych sekcjach
       - Zaznacz fragmenty tekstu, które wyglądają na wygenerowane przez AI
       - Znajdź ciągi znaków bez znaczenia lub losowe znaki

    2. ANALIZA FORMATOWANIA ATS:
       - Wykryj problemy z formatowaniem, które mogą utrudnić odczyt przez systemy ATS
       - Sprawdź, czy nagłówki sekcji są odpowiednio wyróżnione
       - Zweryfikuj, czy tekst jest odpowiednio podzielony na sekcje
       - Oceń czytelność dla systemów automatycznych

    3. ANALIZA SŁÓW KLUCZOWYCH:
       - Sprawdź gęstość słów kluczowych i trafność ich wykorzystania
       - Zidentyfikuj brakujące słowa kluczowe z branży/stanowiska
       - Oceń rozmieszczenie słów kluczowych w dokumencie

    4. OCENA KOMPLETNOŚCI:
       - Zidentyfikuj brakujące sekcje lub informacje, które są często wymagane przez ATS
       - Wskaż informacje, które należy uzupełnić

    5. WERYFIKACJA AUTENTYCZNOŚCI:
       - Zaznacz fragmenty, które wyglądają na sztuczne lub wygenerowane przez AI
       - Podkreśl niespójności między różnymi częściami CV

    6. OCENA OGÓLNA:
       - Oceń ogólną skuteczność CV w systemach ATS w skali 1-10
       - Podaj główne powody obniżonej oceny

    {context}

    CV do analizy:
    {cv_text}

    Odpowiedz w tym samym języku co CV. Jeśli CV jest po polsku, odpowiedz po polsku.
    Format odpowiedzi:

    1. OCENA OGÓLNA (skala 1-10): [ocena]

    2. PROBLEMY KRYTYCZNE:
    [Lista wykrytych krytycznych problemów]

    3. PROBLEMY ZE STRUKTURĄ:
    [Lista problemów strukturalnych]

    4. PROBLEMY Z FORMATOWANIEM ATS:
    [Lista problemów z formatowaniem]

    5. ANALIZA SŁÓW KLUCZOWYCH:
    [Wyniki analizy słów kluczowych]

    6. BRAKUJĄCE INFORMACJE:
    [Lista brakujących informacji]

    7. PODEJRZANE ELEMENTY:
    [Lista elementów, które wydają się wygenerowane przez AI lub są niespójne]

    8. REKOMENDACJE NAPRAWCZE:
    [Konkretne sugestie, jak naprawić zidentyfikowane problemy]

    9. PODSUMOWANIE:
    [Krótkie podsumowanie i zachęta]
    """

    return send_api_request(prompt, max_tokens=1800)

def analyze_cv_strengths(cv_text, job_title="analityk danych", language='pl'):
    """
    Analyze CV strengths for a specific job position and provide improvement suggestions
    """
    prompt = f"""
    ZADANIE: Przeprowadź dogłębną analizę mocnych stron tego CV w kontekście stanowiska {job_title}.

    1. Zidentyfikuj i szczegółowo omów 5-7 najsilniejszych elementów CV, które są najbardziej wartościowe dla pracodawcy.
    2. Dla każdej mocnej strony wyjaśnij, dlaczego jest ona istotna właśnie dla stanowiska {job_title}.
    3. Zaproponuj konkretne ulepszenia, które mogłyby wzmocnić te mocne strony.
    4. Wskaż obszary, które mogłyby zostać dodane lub rozbudowane, aby CV było jeszcze lepiej dopasowane do stanowiska.
    5. Zaproponuj, jak lepiej zaprezentować osiągnięcia i umiejętności, aby były bardziej przekonujące.

    CV:
    {cv_text}

    Pamiętaj, aby Twoja analiza była praktyczna i pomocna. Używaj konkretnych przykładów z CV i odnoś je do wymagań typowych dla stanowiska {job_title}.
    """

    return send_api_request(prompt, max_tokens=2500)

def ai_personal_branding_consultant(cv_text, target_industry="IT", career_goals="", language='pl'):
    """
    🚀 AI PERSONAL BRANDING CONSULTANT - Unikalna funkcja AI
    Analizuje CV i tworzy kompletną strategię marki osobistej z rekomendacjami dla wszystkich kanałów
    """
    prompt = f"""
    🎯 ZADANIE: Jesteś ekspertem od personal branding z 15-letnim doświadczeniem. Przeanalizuj CV i stwórz kompletną strategię marki osobistej kandydata.

    📊 ANALIZA WEJŚCIOWA:
    CV: {cv_text}
    Branża docelowa: {target_industry}
    Cele kariery: {career_goals}

    🔍 PRZEPROWADŹ PEŁNĄ ANALIZĘ PERSONAL BRANDING:

    1. ANALIZA TOŻSAMOŚCI ZAWODOWEJ:
       - Zidentyfikuj unikalne punkty różnicujące kandydata (USP)
       - Określ główne kompetencje i specjalizacje
       - Znajdź "superpower" kandydata - to co robi go wyjątkowym
       - Wskaż potencjalne luki w profilu zawodowym

    2. STRATEGIA POZYCJONOWANIA:
       - Jak kandydat powinien być postrzegany w branży
       - Jakie słowa kluczowe powinny być z nim kojarzone
       - Jaka jest jego "elevator pitch" (30-sekundowy opis)
       - Jak odróżnić się od konkurencji

    3. STRATEGIA TREŚCI (Content Strategy):
       - 10 tematów branżowych, o których kandydat powinien pisać
       - Sugestie postów na LinkedIn (5 konkretnych pomysłów)
       - Tematy do artykułów/blogów branżowych
       - Hashtagi i słowa kluczowe do używania

    4. OPTYMALIZACJA PROFILI ONLINE:
       - Rekomendacje dla profilu LinkedIn (nagłówek, opis, sekcje)
       - Sugestie dla bio na Twitterze/X
       - Pomysły na portfolio online/stronę osobistą
       - GitHub profile optimization (jeśli IT)

    5. STRATEGIA NETWORKING:
       - Jakie wydarzenia branżowe kandydat powinien odwiedzać
       - Z jakimi influencerami/ekspertami powinien się łączyć
       - Jakie grupy LinkedIn/Facebook/Discord warto dołączyć
       - Strategia budowania relacji w branży

    6. PERSONAL BRANDING ROADMAP (12 miesięcy):
       - Krótkoterminowe cele (1-3 miesiące)
       - Średnioterminowe cele (3-6 miesięcy)
       - Długoterminowe cele (6-12 miesięcy)
       - KPI do mierzenia sukcesu personal branding

    7. CONTENT CALENDAR:
       - Plan publikacji na 4 tygodnie
       - Mix różnych typów treści (posty, artykuły, komentarze)
       - Optymalny timing publikacji
       - Template'y postów do wykorzystania

    8. CRISIS MANAGEMENT:
       - Jak reagować na negatywne komentarze
       - Jak zarządzać reputacją online
       - Backup plan jeśli coś pójdzie nie tak

    9. STORYTELLING FRAMEWORK:
       - Jak opowiadać swoją historię kariery
       - 3 kluczowe historie sukcesu do wykorzystania
       - Jak łączyć doświadczenie z wizją przyszłości
       - Template do "About me" sekcji

    10. MONETIZATION STRATEGY:
        - Jak personal branding może przełożyć się na lepsze oferty pracy
        - Możliwości dodatkowych źródeł dochodu (consulting, speaking, kursy)
        - Budowanie authority w branży

    ODPOWIEDŹ W FORMACIE JSON:
    {{
        "personal_brand_analysis": {{
            "unique_selling_points": ["USP1", "USP2", "USP3"],
            "core_competencies": ["kompetencja1", "kompetencja2", "kompetencja3"],
            "superpower": "główna unikalna siła kandydata",
            "brand_archetype": "archetyp marki (np. Ekspert, Innowator, Mentor)",
            "elevator_pitch": "30-sekundowy opis kandydata",
            "competitive_advantage": "przewaga konkurencyjna"
        }},
        "positioning_strategy": {{
            "target_perception": "jak kandydat powinien być postrzegany",
            "key_messages": ["komunikat1", "komunikat2", "komunikat3"],
            "brand_keywords": ["słowo1", "słowo2", "słowo3", "słowo4", "słowo5"],
            "differentiation_factors": ["różnicujący1", "różnicujący2"]
        }},
        "content_strategy": {{
            "content_pillars": [
                {{"pillar": "nazwa filaru", "description": "opis", "example_topics": ["temat1", "temat2"]}},
                {{"pillar": "nazwa filaru", "description": "opis", "example_topics": ["temat1", "temat2"]}}
            ],
            "linkedin_post_ideas": [
                {{"type": "typ postu", "topic": "temat", "template": "szablon treści"}},
                {{"type": "typ postu", "topic": "temat", "template": "szablon treści"}}
            ],
            "article_topics": ["temat artykułu 1", "temat artykułu 2", "temat artykułu 3"],
            "hashtag_strategy": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5"]
        }},
        "profile_optimization": {{
            "linkedin_headline": "zoptymalizowany nagłówek LinkedIn",
            "linkedin_summary": "zoptymalizowany opis LinkedIn (pierwsze 2 zdania)",
            "twitter_bio": "bio dla Twitter/X",
            "portfolio_suggestions": ["element portfolio 1", "element portfolio 2"],
            "github_optimization": "jeśli dotyczy - sugestie dla GitHub"
        }},
        "networking_strategy": {{
            "target_events": ["wydarzenie 1", "wydarzenie 2", "wydarzenie 3"],
            "key_influencers": ["influencer 1", "influencer 2", "influencer 3"],
            "communities_to_join": ["społeczność 1", "społeczność 2", "społeczność 3"],
            "networking_approach": "strategia budowania relacji"
        }},
        "roadmap_12_months": {{
            "months_1_3": {{
                "goals": ["cel 1", "cel 2"],
                "actions": ["akcja 1", "akcja 2"],
                "kpis": ["KPI 1", "KPI 2"]
            }},
            "months_3_6": {{
                "goals": ["cel 1", "cel 2"],
                "actions": ["akcja 1", "akcja 2"],
                "kpis": ["KPI 1", "KPI 2"]
            }},
            "months_6_12": {{
                "goals": ["cel 1", "cel 2"],
                "actions": ["akcja 1", "akcja 2"],
                "kpis": ["KPI 1", "KPI 2"]
            }}
        }},
        "content_calendar_4_weeks": [
            {{
                "week": 1,
                "monday": {{"type": "typ treści", "topic": "temat", "platform": "LinkedIn"}},
                "wednesday": {{"type": "typ treści", "topic": "temat", "platform": "LinkedIn"}},
                "friday": {{"type": "typ treści", "topic": "temat", "platform": "LinkedIn"}}
            }},
            {{
                "week": 2,
                "monday": {{"type": "typ treści", "topic": "temat", "platform": "LinkedIn"}},
                "wednesday": {{"type": "typ treści", "topic": "temat", "platform": "LinkedIn"}},
                "friday": {{"type": "typ treści", "topic": "temat", "platform": "LinkedIn"}}
            }}
        ],
        "storytelling_framework": {{
            "career_story_arc": "jak opowiedzieć swoją historię kariery",
            "success_stories": [
                {{"title": "tytuł historii", "situation": "sytuacja", "action": "działanie", "result": "rezultat"}},
                {{"title": "tytuł historii", "situation": "sytuacja", "action": "działanie", "result": "rezultat"}}
            ],
            "about_me_template": "szablon do sekcji 'o mnie'"
        }},
        "monetization_opportunities": {{
            "immediate_benefits": ["korzyść 1", "korzyść 2"],
            "medium_term_opportunities": ["możliwość 1", "możliwość 2"],
            "long_term_potential": ["potencjał 1", "potencjał 2"],
            "authority_building_steps": ["krok 1", "krok 2", "krok 3"]
        }},
        "implementation_guide": {{
            "first_week_actions": ["akcja 1", "akcja 2", "akcja 3"],
            "tools_needed": ["narzędzie 1", "narzędzie 2"],
            "time_investment": "szacowany czas na implementację",
            "success_metrics": ["metryka 1", "metryka 2"]
        }},
        "executive_summary": "Zwięzłe podsumowanie całej strategii personal branding w 2-3 zdaniach"
    }}

    💡 UWAGA: Stwórz strategię opartą wyłącznie na faktach z CV. Nie wymyślaj osiągnięć ani doświadczeń, których tam nie ma.
    🎯 CEL: Pomóc kandydatowi zbudować silną markę osobistą, która przyciągnie lepsze oferty pracy i możliwości kariery.
    """

    return send_api_request(prompt, max_tokens=8000, language=language)

def generate_interview_questions(cv_text, job_description="", language='pl'):
    """
    Generate likely interview questions based on CV and job description
    """
    context = ""
    if job_description:
        context = f"Uwzględnij poniższe ogłoszenie o pracę przy tworzeniu pytań:\n{job_description[:2000]}"

    prompt = f"""
    TASK: Wygeneruj zestaw potencjalnych pytań rekrutacyjnych, które kandydat może otrzymać podczas rozmowy kwalifikacyjnej.

    Pytania powinny być:
    1. Specyficzne dla doświadczenia i umiejętności kandydata wymienionych w CV
    2. Dopasowane do stanowiska (jeśli podano opis stanowiska)
    3. Zróżnicowane - połączenie pytań technicznych, behawioralnych i sytuacyjnych
    4. Realistyczne i często zadawane przez rekruterów

    Uwzględnij po co najmniej 3 pytania z każdej kategorii:
    - Pytania o doświadczenie zawodowe
    - Pytania techniczne/o umiejętności
    - Pytania behawioralne
    - Pytania sytuacyjne
    - Pytania o motywację i dopasowanie do firmy/stanowiska

    {context}

    CV:
    {cv_text}

    Odpowiedz w tym samym języku co CV. Jeśli CV jest po polsku, odpowiedz po polsku.
    Dodatkowo, do każdego pytania dodaj krótką wskazówkę, jak można by na nie odpowiedzieć w oparciu o informacje z CV.
    """

    return send_api_request(prompt, max_tokens=2000)