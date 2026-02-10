# ========================================
# 1. ENVIRONMENT SETUP & LLM INIT
# ========================================
import os
import json
import logging
from dotenv import load_dotenv

try:
    from langchain.chat_models import ChatOpenAI
    from langchain import LLMChain, PromptTemplate
    USE_LANGCHAIN = True
except Exception:
    # Fallback: langchain not available or incompatible in this environment.
    USE_LANGCHAIN = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Provider selection: 'openai' or 'gemini'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_URL = os.getenv("GEMINI_URL")

# Mock mode for offline/demo runs
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Validate provider credentials lazily
if LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise EnvironmentError("Set OPENAI_API_KEY in your environment or .env for OpenAI provider")
    import openai
    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI()
    if USE_LANGCHAIN:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
elif LLM_PROVIDER == "gemini":
    # GEMINI can authenticate via ADC (preferred) or via explicit bearer token in GEMINI_BEARER_TOKEN
    import requests
    if USE_LANGCHAIN:
        logger.info("LangChain detected but Gemini provider selected; LangChain chains will use OpenAI/generic fallback.")
else:
    raise ValueError("Unsupported LLM_PROVIDER: " + LLM_PROVIDER)

# If LangChain is not available, we'll fallback to using OpenAI/Gemini directly for chaining steps.
if not USE_LANGCHAIN:
    logger.info("LangChain not available; using direct provider SDK/fallback for chaining.")


def get_google_access_token():
    """Obtain an OAuth2 access token using Application Default Credentials.
    Tries google-auth first; if unavailable, falls back to gcloud CLI.
    """
    try:
        import google.auth
        from google.auth.transport.requests import Request
        creds, _ = google.auth.default()
        if not creds.valid:
            creds.refresh(Request())
        return creds.token
    except Exception:
        # Fallback to gcloud CLI (user must have run `gcloud auth application-default login`)
        try:
            import subprocess
            out = subprocess.check_output(["gcloud", "auth", "application-default", "print-access-token"], stderr=subprocess.STDOUT)
            return out.decode().strip()
        except Exception as e:
            raise EnvironmentError(
                "Unable to obtain Google access token. Install and configure Application Default Credentials (run `gcloud auth application-default login`) or provide service account credentials via GOOGLE_APPLICATION_CREDENTIALS. Error: " + str(e)
            )

# ========================================
# 2. CHAINING APPROACH (Primary Demo)
# ========================================
# Step 1: Test Plan Chain
role = "You are a Principal QA Engineer with 20+ years experience"
context = "Generate ISTQB-compliant test artifacts. Return JSON only, no markdown."
few_shot_examples = """
Example:
Input Feature: "Shopping cart checkout"
Output:
{
  "test_plan": "Validate checkout flow, payments, edge cases",
  "notes": "Use exploratory and regression testing"
}
"""

if USE_LANGCHAIN:
    plan_prompt = PromptTemplate(
        input_variables=["feature_description"],
        template=(
            "{role}\n{context}\n{few_shot}\n"
            "Task: Generate high-level test plan JSON for the feature.\n"
            "Schema: {\"test_plan\":\"...\"}\n"
            "If no test plan applicable, set test_plan to \"\".\n"
            "Return JSON only.\n\nFeature: {feature_description}"
        )
    )
    plan_prompt = PromptTemplate(
        input_variables=["feature_description"],
        template=plan_prompt.template.format(role=role, context=context, few_shot=few_shot_examples, feature_description="{feature_description}")
    )

    # Step 2: Test Cases Chain
    cases_prompt = PromptTemplate(
        input_variables=["feature_description", "test_plan"],
        template=(
            "{role}\n{context}\n{few_shot}\n"
            "Previous_Output: {test_plan}\n"
            "Task: Expand the previous_output into detailed test cases JSON following this schema:\n"
            "{\n  \"test_plan\": \"...\",\n  \"test_cases\": [\n    {\"id\": 1, \"title\": \"\", \"type\": \"positive|negative|edge\", \"steps\": [\"...\"], \"expected\": \"...\"}\n  ]\n}\n"
            "Validation: If no test cases found, return an empty array for \"test_cases\".\n"
            "Return JSON only.\n\nFeature: {feature_description}"
        )
    )
    cases_prompt = PromptTemplate(
        input_variables=["feature_description", "test_plan"],
        template=cases_prompt.template.format(role=role, context=context, few_shot=few_shot_examples, feature_description="{feature_description}", test_plan="{test_plan}")
    )
else:
    # fallback raw prompt strings
    plan_prompt_str = (
        f"{role}\n{context}\n{few_shot_examples}\n"
        "Task: Generate high-level test plan JSON for the feature.\n"
        "Schema: {\"test_plan\":\"...\"}\n"
        "If no test plan applicable, set test_plan to \"\".\n"
        "Return JSON only.\n\nFeature: "
    )
    cases_prompt_str = (
        f"{role}\n{context}\n{few_shot_examples}\n"
        "Previous_Output: {test_plan}\n"
        "Task: Expand the previous_output into detailed test cases JSON following the schema.\n"
        "Validation: If no test cases found, return an empty array for \"test_cases\".\n"
        "Return JSON only.\n\nFeature: "
    )

if USE_LANGCHAIN:
    plan_chain = LLMChain(llm=llm, prompt=plan_prompt, output_key="test_plan")
    cases_chain = LLMChain(llm=llm, prompt=cases_prompt, output_key="full_output")

    def run_chaining(feature_description):
        logger.info("Chaining: Generating high-level test plan...")
        try:
            test_plan_raw = plan_chain.run(feature_description=feature_description)
            logger.info("Chaining: Expanding into detailed test cases...")
            full_raw = cases_chain.run(feature_description=feature_description, test_plan=test_plan_raw)
            try:
                parsed = json.loads(full_raw)
            except Exception:
                import re
                m = re.search(r"\{.*\}", full_raw, re.S)
                if m:
                    parsed = json.loads(m.group(0))
                else:
                    raise
            if "test_cases" not in parsed:
                parsed["test_cases"] = []
            return parsed
        except Exception as e:
            logger.exception("Chaining failed")
            return {"error": str(e), "test_cases": []}
else:
    def run_chaining(feature_description):
        logger.info("Fallback Chaining: Generating high-level test plan via OpenAI...")
        try:
            # Step 1: plan
            # Use provider-agnostic chat wrapper
            msgs = [
                {"role": "system", "content": role + "\n" + context + "\n" + few_shot_examples},
                {"role": "user", "content": "Task: Generate high-level test plan JSON for the feature.\nFeature: " + feature_description},
            ]
            if MOCK_MODE:
                test_plan_raw = '{"test_plan":"Validate login flows, OAuth, error handling"}'
            else:
                resp_text = chat_complete(msgs, model="gpt-4o-mini")
                test_plan_raw = resp_text

            logger.info("Fallback Chaining: Expanding into detailed test cases via OpenAI...")
            msgs2 = [
                {"role": "system", "content": role + "\n" + context + "\n" + few_shot_examples},
                {"role": "user", "content": "Previous_Output: " + test_plan_raw + "\nTask: Expand into detailed test cases. Feature: " + feature_description},
            ]
            if MOCK_MODE:
                full_raw = json.dumps({
                    "test_plan": "Validate login flows, OAuth, error handling",
                    "test_cases": [
                        {"id": 1, "title": "Valid email login", "type": "positive", "steps": ["Enter valid email","Enter valid password","Click login"], "expected": "User is logged in"},
                        {"id": 2, "title": "Google OAuth success", "type": "positive", "steps": ["Click Google login","Select account","Consent"], "expected": "User is logged in via Google"}
                    ]
                })
            else:
                full_raw = chat_complete(msgs2, model="gpt-4o-mini")
            try:
                parsed = json.loads(full_raw)
            except Exception:
                import re
                # Try to find and extract the largest valid JSON object from the response
                # Search for objects starting with { and find valid closing }
                json_match = None
                for m in re.finditer(r'\{', full_raw):
                    start_pos = m.start()
                    # Count braces to find matching closing }
                    brace_count = 0
                    for i, char in enumerate(full_raw[start_pos:], start=start_pos):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                try:
                                    potential_json = full_raw[start_pos:i+1]
                                    parsed = json.loads(potential_json)
                                    json_match = parsed
                                    break
                                except json.JSONDecodeError:
                                    continue
                    if json_match:
                        break
                
                if json_match:
                    parsed = json_match
                else:
                    # If no valid JSON found, wrap minimally
                    parsed = {"test_plan": test_plan_raw, "test_cases": []}
            if "test_cases" not in parsed:
                parsed["test_cases"] = []
            return parsed
        except Exception as e:
            logger.exception("Fallback chaining failed")
            return {"error": str(e), "test_cases": []}

# ========================================
# 3. NO-CHAINING APPROACH (Direct OpenAI)
# ========================================
# Single comprehensive prompt
# Direct client.chat.completions.create()
try:
    import openai
except Exception:
    raise ImportError("Please `pip install openai` to run the direct OpenAI example")

openai.api_key = OPENAI_API_KEY

def run_single_prompt(feature_description):
    system_msg = f"{role}\n{context}\n{few_shot_examples}"
    user_msg = (
        "Task: Produce a single JSON following this schema:\n"
        "{\n  \"test_plan\":\"...\",\n  \"test_cases\":[{\"id\":1,\"title\":\"\",\"type\":\"positive|negative|edge\",\"steps\":[\"...\"],\"expected\":\"...\"}]\n}\n"
        "Validation: If no test cases found, return an empty array for \"test_cases\".\n"
        "Return JSON only.\n\nFeature: " + feature_description
    )
    try:
        if MOCK_MODE:
            text = json.dumps({
                "test_plan": "Validate login flows, OAuth, error handling",
                "test_cases": [
                    {"id": 1, "title": "Valid email login", "type": "positive", "steps": ["Enter valid email","Enter valid password","Click login"], "expected": "User is logged in"}
                ]
            })
        else:
            msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            # Use the same provider as configured (openai or gemini)
            model = "gemini-2.0-flash-lite" if LLM_PROVIDER == "gemini" else "gpt-4o-mini"
            text = chat_complete(msgs, model=model)
        try:
            parsed = json.loads(text)
        except Exception:
            import re
            # Improved JSON extraction: find valid JSON objects
            json_match = None
            for m in re.finditer(r'\{', text):
                start_pos = m.start()
                brace_count = 0
                for i, char in enumerate(text[start_pos:], start=start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                potential_json = text[start_pos:i+1]
                                parsed = json.loads(potential_json)
                                json_match = parsed
                                break
                            except json.JSONDecodeError:
                                continue
                if json_match:
                    break
            
            if not json_match:
                raise
        if "test_cases" not in parsed:
            parsed["test_cases"] = []
        return parsed
    except Exception as e:
        logger.exception("Single prompt failed")
        return {"error": str(e), "test_cases": []}

# ========================================
# 4. COMPARISON TABLE
# ========================================
def compare_results(chain_res, single_res):
    table = [
        {"Aspect":"Detail","Chaining":"High","Single Prompt":"Medium"},
        {"Aspect":"Steps","Chaining":"2 (plan -> cases)","Single Prompt":"1"},
        {"Aspect":"Maintainability","Chaining":"Better (modular)","Single Prompt":"Harder"},
        {"Aspect":"Traceability","Chaining":"Yes","Single Prompt":"Limited"},
    ]
    return table


def chat_complete(messages, model="gpt-4o-mini"):
    # messages: list of {role, content}
    joined = "\n".join([m.get("content", "") for m in messages])
    if LLM_PROVIDER == "openai":
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=1500)
        # OpenAI client response object
        return resp.choices[0].message.content
    elif LLM_PROVIDER == "gemini":
        # Build a simple payload (used for REST fallback)
        payload = {"prompt": {"text": joined}, "temperature": 0}
        # Prefer explicit bearer token override (useful for manual testing)
        token = os.getenv("GEMINI_BEARER_TOKEN")
        # If no manual token, try ADC (google-auth or gcloud) unless in MOCK_MODE
        if not token and not MOCK_MODE:
            try:
                token = get_google_access_token()
            except Exception:
                token = None

        # If a Vertex AI Python SDK is available, use it (handles ADC and request shape)
        if GEMINI_URL and "aiplatform.googleapis.com" in GEMINI_URL:
            try:
                # Try google-generativeai SDK (works with ADC)
                import google.generativeai as genai
                model_id = GEMINI_MODEL or None
                try:
                    path_tail = GEMINI_URL.split("/")[-1]
                    model_id = path_tail.split(":")[0]
                except Exception:
                    pass
                
                if not model_id:
                    raise RuntimeError("Unable to determine model id")
                
                gen = genai.GenerativeModel(model_id)
                response = gen.generate_content(joined)
                if hasattr(response, "text"):
                    return response.text
                try:
                    if isinstance(response, dict) and "text" in response:
                        return response["text"]
                    if isinstance(response, list) and len(response) > 0:
                        first = response[0]
                        if isinstance(first, dict) and "text" in first:
                            return first["text"]
                except Exception:
                    pass
                return json.dumps(response)
            except Exception as e:
                # If SDK not available or failed, fall back to REST below
                logger.info("google-generativeai SDK unavailable or failed, falling back to REST: %s", str(e))

        # Allow caller to provide an exact request body via env to handle provider-specific shapes
        custom_body = os.getenv("GEMINI_REQUEST_JSON")
        custom_body_file = os.getenv("GEMINI_REQUEST_FILE")
        if custom_body_file and os.path.exists(custom_body_file):
            try:
                with open(custom_body_file, "r") as fh:
                    custom_body = fh.read()
            except Exception:
                pass

        # REST paths: if explicit GEMINI_URL provided, call it with a request body matching generateContent expectations
        if GEMINI_URL:
            if custom_body:
                try:
                    rest_payload = json.loads(custom_body)
                except Exception:
                    # if it's plain text, send raw string under a conservative wrapper
                    rest_payload = {"content": str(custom_body)}
            else:
                # Vertex AI generateContent expects a JSON body with an `instances` structure.
                # Use the shape with content items to match the SDK: {"instances": [{"content":[{"type":"text","text":"..."}]}], "parameters": {...}}
                rest_payload = {
                    "instances": [
                        {"content": [{"type": "text", "text": joined}]}
                    ],
                    "parameters": {"temperature": 0}
                }
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = requests.post(GEMINI_URL, json=rest_payload, timeout=30, headers=headers)
            r.raise_for_status()
            data = r.json()
        else:
            # Try common API versions/endpoints to handle publisher model resource paths
            api_versions = ["v1", "v1beta2", "v1alpha2"]
            data = None
            last_err = None
            for api in api_versions:
                url = f"https://generativelanguage.googleapis.com/{api}/{GEMINI_MODEL}:generate"
                try:
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    r = requests.post(url, json=payload, timeout=30, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    last_err = e
                    continue
            if data is None:
                # surface last error
                raise last_err
        # Robust parsing for various Generative API shapes
        # Prefer 'candidates' -> candidate -> 'content' or 'output'
        if isinstance(data, dict):
            if "candidates" in data and len(data["candidates"]) > 0:
                cand = data["candidates"][0]
                if isinstance(cand, dict):
                    if "output" in cand:
                        return cand.get("output", "")
                    if "content" in cand:
                        cont = cand.get("content")
                        if isinstance(cont, str):
                            return cont
                        if isinstance(cont, list) and len(cont) > 0:
                            # look for text fields
                            for item in cont:
                                if isinstance(item, dict) and "text" in item:
                                    return item["text"]
                            return json.dumps(cont)
            # older shape: 'output' at top-level
            if "output" in data:
                return data.get("output", "")
        # fallback: stringify entire response
        return json.dumps(data)
    else:
        raise ValueError("Unsupported LLM_PROVIDER: " + LLM_PROVIDER)

if __name__ == "__main__":
    feature = "User login feature with email/password authentication and Google OAuth"
    chain_result = run_chaining(feature)
    single_result = run_single_prompt(feature)

    print("Chaining Result:")
    print(json.dumps(chain_result, indent=2))

    print("\nSingle Prompt Result:")
    print(json.dumps(single_result, indent=2))

    print("\nComparison Table:")
    print(json.dumps(compare_results(chain_result, single_result), indent=2))

print("✅ Implementation complete. Compare chaining vs single prompt outputs above.")
