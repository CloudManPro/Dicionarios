import json
import boto3
import os
import urllib3
import logging
import re  # Import regex module for cleanup

# ==============================================================================
# Configuration
# ==============================================================================

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Use DEBUG for development, INFO/WARNING for production

# AWS Environment Variables
region = os.getenv("REGION", "us-east-1")
parameter_name = os.getenv("AWS_SSM_PARAMETER_TARGET_NAME_0")

# Validate essential environment variables
if not parameter_name:
    logger.critical("FATAL: Environment variable AWS_SSM_PARAMETER_TARGET_NAME_0 is not set.")
    raise ValueError("Missing required environment variable: AWS_SSM_PARAMETER_TARGET_NAME_0")
if not region:
    logger.warning("Environment variable REGION not set, defaulting to us-east-1.")

# ==============================================================================
# AWS Client Initialization
# ==============================================================================
ssm = boto3.client('ssm', region_name=region)

# ==============================================================================
# Global Variables / Constants
# ==============================================================================
GEMINI_API_KEY = None  # Cached API Key

# Gemini Model/API Configuration
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"
STABLE_API_VERSION = "v1" # Renamed for clarity
BETA_API_VERSION = "v1beta" # Renamed for clarity
# Use Beta as default if using a 1.5 model
DEFAULT_API_VERSION = BETA_API_VERSION if "1.5" in DEFAULT_GEMINI_MODEL else STABLE_API_VERSION

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_gemini_api_key():
    """Retrieves and caches the Gemini API Key from SSM Parameter Store."""
    global GEMINI_API_KEY
    if GEMINI_API_KEY:
        return GEMINI_API_KEY

    if not parameter_name:
        raise ValueError("SSM Parameter name is not configured.")

    try:
        logger.info(f"Attempting to retrieve API Key from SSM Parameter: {parameter_name}")
        ssm_response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        GEMINI_API_KEY = ssm_response['Parameter']['Value']
        if not GEMINI_API_KEY:
            raise ValueError("Retrieved API Key from SSM is empty.")
        logger.info(f"API Key retrieved successfully (first 4 chars): {GEMINI_API_KEY[:4]}...")
        return GEMINI_API_KEY
    except ssm.exceptions.ParameterNotFound:
        logger.critical(f"FATAL: SSM Parameter '{parameter_name}' not found in region '{region}'.")
        raise
    except Exception as e:
        logger.critical(f"FATAL: Error retrieving API key from SSM: {e}", exc_info=True)
        raise

def make_gemini_request(api_key, api_version, model_name, payload_contents, timeout=90.0):
    """Sends a request to the Gemini API and returns the HTTP response object."""
    gemini_api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
    logger.info(f"Making request to Gemini API URL: {gemini_api_url}") # Changed to INFO for visibility

    headers = {"Content-Type": "application/json"}
    payload = {"contents": payload_contents}

    # Log payload safely (redact large data)
    log_payload = payload
    try:
        if any('inline_data' in part for content in payload_contents for part in content.get('parts', [])):
            log_payload = json.loads(json.dumps(payload))  # Deep copy
            for content in log_payload['contents']:
                for part in content.get('parts', []):
                    if 'inline_data' in part and 'data' in part['inline_data']:
                        data_len = len(part['inline_data']['data'])
                        part['inline_data']['data'] = f"<base64 data ({data_len} bytes)>"
        logger.debug(f"Payload sent to Gemini: {json.dumps(log_payload, ensure_ascii=False)}")
    except Exception as log_e:
        logger.warning(f"Could not redact payload for logging: {log_e}")
        logger.debug(f"Payload structure might be complex. Logging keys only: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}")

    http = urllib3.PoolManager()
    try:
        response = http.request(
            "POST",
            gemini_api_url,
            body=json.dumps(payload).encode("utf-8"),
            headers=headers,
            timeout=timeout
        )
        logger.debug(f"Gemini response status: {response.status}") # Add status log
        return response
    except urllib3.exceptions.ReadTimeoutError:
        logger.error(f"Gemini API call timed out after {timeout} seconds.")
        raise TimeoutError(f"Gemini API timeout after {timeout}s")
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}", exc_info=True)
        raise ConnectionError(f"Gemini API request failed: {e}")

def create_proxy_response(status_code, body_object):
    """Creates a standard AWS Lambda Proxy response dictionary with CORS."""
    try:
        body_string = json.dumps(body_object, ensure_ascii=False)
    except TypeError as e:
        logger.error(f"Failed to serialize response body object: {e}", exc_info=True)
        body_string = json.dumps({"error": "Internal server error: Failed to serialize response.", "details": str(e)})
        status_code = 500

    return {
        "statusCode": status_code,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": body_string
    }

# ==============================================================================
# Main Lambda Handler
# ==============================================================================

def lambda_handler(event, context):
    """
    Main Lambda handler: parses input, calls Gemini, processes response, returns proxy format.
    """
    final_payload = {}
    tokens_info = {}

    try:
        api_key = get_gemini_api_key()

        logger.debug(f"Lambda execution started. Request ID: {context.aws_request_id}")
        logger.debug(f"Received raw event: {event}")

        input_data = {}
        action = None
        if isinstance(event, dict):
            action = event.get("action")
            if action:
                logger.info("Reading parameters directly from event object.")
                input_data = event
            else:
                logger.info("Action not direct, trying event['body'].")
                try:
                    body_content = event.get('body')
                    if isinstance(body_content, str):
                        input_data = json.loads(body_content)
                        action = input_data.get("action")
                    elif isinstance(body_content, dict):
                        input_data = body_content
                        action = input_data.get("action")
                    else:
                        logger.warning(f"event['body'] is missing or has unexpected type: {type(body_content).__name__}")
                        input_data = {} # Ensure input_data is initialized
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in event['body'].")
                    return create_proxy_response(400, {"error": "Invalid JSON format in request body."})
                except Exception as e:
                    logger.error(f"Error processing event['body']: {e}", exc_info=True)
                    return create_proxy_response(400, {"error": "Could not process request body."})
        else:
            logger.error(f"Received event is not a dictionary: {type(event).__name__}")
            return create_proxy_response(400, {"error": "Invalid event format."})

        # Re-check action after potential body parsing
        if not action:
            action = input_data.get("action")
            if not action:
                 logger.error("Missing 'action' parameter in request.")
                 return create_proxy_response(400, {"error": "O parâmetro 'action' é obrigatório."})


        logger.info(f"Processing action: {action}")

        payload_contents = []
        # Use the default determined by model type initially
        api_version = DEFAULT_API_VERSION
        model_name = DEFAULT_GEMINI_MODEL

        # --- Action-Specific Logic ---
        if action == "SyntesysAI":
            text_input = input_data.get("text")
            prompt_input = input_data.get("prompt")
            if not text_input or not prompt_input:
                return create_proxy_response(400, {"error": "'text' e 'prompt' obrigatórios para 'SyntesysAI'."})
            combined_prompt = (
            f"{text_input} com base neste texto, responda {prompt_input}, destacando os principais trechos dos textos "
            f"e indicando o título. Não mencione os textos que não contribuírem diretamente para a questão. "
            f"Faça um resumo do seu entendimento ao final."
        )

            payload_contents = [{"role": "user", "parts": [{"text": combined_prompt}]}]
            # No need to set api_version = "v1" here anymore

        elif action == "KeyWords":
            keyWords_input = input_data.get("keyWords")
            if not keyWords_input:
                return create_proxy_response(400, {"error": "'keyWords' obrigatório para 'KeyWords'."})
            combined_prompt = f"selecione as palavras chave...({keyWords_input})"
            payload_contents = [{"role": "user", "parts": [{"text": combined_prompt}]}]
            # No need to set api_version = "v1" here anymore

        elif action == "FixText":
            text_input = input_data.get("text")
            if not text_input:
                return create_proxy_response(400, {"error": "'text' obrigatório para 'FixText'."})
            combined_prompt = f"{text_input}\nCorrija o texto..."
            payload_contents = [{"role": "user", "parts": [{"text": combined_prompt}]}]
            # No need to set api_version = "v1" here anymore

        elif action == "convertHTML":
            pdf_base64 = input_data.get("pdf_base64")
            if not pdf_base64:
                return create_proxy_response(400, {"error": "'pdf_base64' obrigatório para 'convertHTML'."})
            prompt_input = (
                "Converta o conteúdo do PDF anexo em HTML respeitando ao máximo a formatação, size e fontes."
                "Sua resposta DEVE ser APENAS um objeto JSON válido e nada mais. Este objeto JSON DEVE conter exatamente uma chave: "
                "'html_content': Uma string contendo APENAS o código HTML completo do PDF convertido. "
                "NÃO inclua nenhuma frase introdutória, explicação, comentário ou blocos de código Markdown. "
                "Remova qualquer header que contém paginação com 'Dicionario da Consciencioterapeuticologia' ou 'OIC'."
                "Remova toda hifenização.  inclua a conversão da bibliografia completa."
                "nunca separe nehuma palavra de um paragrafo, mesmo que parecer com um titulo."
            )
            payload_contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_input},
                        {"inline_data": {"mime_type": "application/pdf", "data": pdf_base64}}
                    ]
                }
            ]
            # Explicitly confirm BETA for multimodal, overriding default if needed
            api_version = BETA_API_VERSION
            # Model is already default, but could override here if needed
            # model_name = DEFAULT_GEMINI_MODEL
            logger.info(f"Using Gemini API {api_version} for action '{action}'.")

        elif action == "convertTXT":
            pdf_base64 = input_data.get("pdf_base64")
            if not pdf_base64:
                return create_proxy_response(400, {"error": "'pdf_base64' obrigatório para 'convertTXT'."})
            prompt_input = (
                "Extraia o texto puro do conteúdo do PDF anexo. "
                '1- Preserve toda quebra de linha.'
                "2- primeiro Remova todos headers, que pode conter numero de página e/ou 'Dicionario da Consciencioterapeuticologia' e/ou 'OIC'. siga ao próximo passo somente após remover os headers."
                "3- Remova toda hifenização."
                "4- inclua a conversão da bibliografia completa."
                "5- nunca separe nehuma palavra de um paragrafo, mesmo que parecer com um titulo."
                "Sua resposta DEVE ser APENAS um objeto JSON válido e nada mais. Este objeto JSON DEVE conter exatamente uma chave: "
                "'txt_content': Uma string contendo APENAS o texto extraído do PDF. "
                "NÃO inclua nenhuma frase introdutória, explicação, comentário ou blocos de código Markdown. "
            )
            payload_contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_input},
                        {"inline_data": {"mime_type": "application/pdf", "data": pdf_base64}}
                    ]
                }
            ]
            # Explicitly confirm BETA for multimodal
            api_version = BETA_API_VERSION
            # model_name = DEFAULT_GEMINI_MODEL
            logger.info(f"Using Gemini API {api_version} for action '{action}'.")

        else:
            logger.warning(f"Ação '{action}' não suportada.")
            return create_proxy_response(400, {"error": f"Ação '{action}' não suportada."})

        # --- FINAL CHECK ---
        # Ensure Beta API is used for 1.5 models if not already set by specific actions
        if "1.5" in model_name and api_version != BETA_API_VERSION:
             logger.warning(f"Model '{model_name}' requires API version '{BETA_API_VERSION}'. Overriding current version '{api_version}'.")
             api_version = BETA_API_VERSION

        # --- Call Gemini ---
        gemini_response = make_gemini_request(api_key, api_version, model_name, payload_contents)

        # --- Process Gemini Response ---
        response_status = gemini_response.status
        response_data = gemini_response.data.decode("utf-8")
        # Log status BEFORE decoding attempt for easier debugging
        logger.debug(f"Gemini API Status Code: {response_status}")
        logger.debug(f"Gemini Raw Response (start): {response_data[:500]}{'...' if len(response_data) > 500 else ''}")

        if response_status != 200:
            error_details = response_data
            try:
                # Try parsing even non-200 responses, Gemini often sends JSON errors
                error_json = json.loads(response_data)
                error_details = error_json.get("error", {}).get("message", response_data)
            except json.JSONDecodeError:
                logger.warning("Gemini error response was not valid JSON.")
            except Exception as e:
                logger.warning(f"Could not parse Gemini error JSON: {e}")

            logger.error(f"Erro da API Gemini ({response_status}): {error_details}")
            # Map Gemini status codes to proxy status codes reasonably
            proxy_status = response_status if response_status >= 400 else 502 # Keep 4xx, treat others as gateway issues
            return create_proxy_response(proxy_status, {"error": f"Erro da API Gemini ({response_status}): {error_details}"})

        try:
            result_json = json.loads(response_data)
        except json.JSONDecodeError as e:
            logger.error(f"Falha ao decodificar JSON da resposta Gemini BEM SUCEDIDA (Status {response_status}): {e}", exc_info=True)
            logger.error(f"Resposta bruta que falhou no parse: {response_data}")
            return create_proxy_response(500, {"error": "Erro interno: Resposta da IA não era JSON válido."})

        # --- Validate Gemini response structure & check safety ---
        # Check for promptFeedback first, as it can exist even without candidates on blocks
        if p_feedback := result_json.get("promptFeedback"):
             if reason := p_feedback.get("blockReason"):
                 logger.warning(f"Gemini prompt block: {reason}. Ratings: {p_feedback.get('safetyRatings')}")
                 return create_proxy_response(400, {"error": f"Requisição bloqueada pela IA (prompt): {reason}"})

        # Now check candidates structure
        candidates = result_json.get('candidates')
        if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
             logger.error(f"Resposta Gemini sem 'candidates' válidos: {response_data}")
             return create_proxy_response(500, {"error": "Erro interno: Estrutura de resposta inesperada da IA (sem candidatos)."})

        candidate = candidates[0] # Process first candidate

        # Check candidate finish reason and safety
        if finish_reason := candidate.get("finishReason"):
            if finish_reason not in ["STOP", "MAX_TOKENS"]: # Other reasons might indicate issues
                 logger.warning(f"Candidate finished with reason: {finish_reason}")
                 if finish_reason == "SAFETY":
                     ratings = candidate.get("safetyRatings")
                     logger.warning(f"Gemini response block: SAFETY. Ratings: {ratings}")
                     # Find the category that triggered the block
                     blocked_category = "Unknown"
                     if ratings:
                         for rating in ratings:
                             if rating.get("blocked"):
                                 blocked_category = rating.get("category", "Unknown")
                                 break
                     return create_proxy_response(400, {"error": f"Resposta bloqueada pela IA (segurança): {blocked_category}"})
                 # Handle other potential non-STOP reasons if necessary
                 # return create_proxy_response(500, {"error": f"Erro interno: IA terminou inesperadamente ({finish_reason})."})


        content = candidate.get('content')
        if not content or not (parts := content.get('parts')) or not isinstance(parts, list) or len(parts) == 0:
            logger.error(f"Resposta Gemini com estrutura de 'content'/'parts' inválida: {response_data}")
            return create_proxy_response(500, {"error": "Erro interno: Estrutura de conteúdo inesperada da IA."})

        # Check if the first part actually contains text (might be empty if blocked earlier but not caught)
        if 'text' not in parts[0]:
             logger.error(f"Primeira parte da resposta Gemini não contém 'text': {response_data}")
             # Could be due to an undetected block or other issue
             if finish_reason and finish_reason != "STOP":
                 return create_proxy_response(500, {"error": f"Erro interno: IA terminou sem texto ({finish_reason})."})
             else:
                 return create_proxy_response(500, {"error": "Erro interno: IA retornou conteúdo sem texto."})

        gemini_output_text = parts[0]['text']

        # --- Construct Final Payload based on Action ---
        if action in ["convertHTML", "convertTXT"]:
            logger.debug(f"Processing '{action}'. Expecting JSON in Gemini text.")
            cleaned_text = gemini_output_text.strip()
            # Regex to find JSON within optional markdown fences (more robust)
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL | re.IGNORECASE)
            json_string_to_parse = match.group(1) if match else cleaned_text
            if match:
                logger.debug("Markdown fences found and JSON extracted.")
            else:
                logger.debug("No Markdown fences found, attempting to parse entire text as JSON.")
            try:
                parsed_gemini_output = json.loads(json_string_to_parse)
                if not isinstance(parsed_gemini_output, dict):
                     raise ValueError("Parsed content is not a JSON object (dictionary).")

                if action == "convertHTML":
                    # Allow for slight variations in the key name
                    html_content = parsed_gemini_output.get("html_content") or parsed_gemini_output.get("html")
                    if html_content is None:
                        logger.error(f"Parsed JSON does not contain 'html_content' or 'html' key. Keys found: {list(parsed_gemini_output.keys())}")
                        raise ValueError("Estrutura JSON da IA incompleta/inesperada para HTML.")
                    if not isinstance(html_content, str):
                         raise ValueError("'html_content'/'html' value is not a string.")
                    final_payload["html_content"] = html_content
                elif action == "convertTXT":
                    # Allow for slight variations in the key name
                    txt_content = parsed_gemini_output.get("txt_content") or parsed_gemini_output.get("text")
                    if txt_content is None:
                        logger.error(f"Parsed JSON does not contain 'txt_content' or 'text' key. Keys found: {list(parsed_gemini_output.keys())}")
                        raise ValueError("Estrutura JSON da IA incompleta/inesperada para TXT.")
                    if not isinstance(txt_content, str):
                         raise ValueError("'txt_content'/'text' value is not a string.")
                    final_payload["txt_content"] = txt_content
                logger.info(f"Conversão para '{action}' processada com sucesso.")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Erro ao processar/validar JSON interno da Gemini: {e}", exc_info=True)
                logger.error(f"String JSON (após limpeza) que falhou: {json_string_to_parse[:1000]}{'...' if len(json_string_to_parse) > 1000 else ''}") # Log snippet
                return create_proxy_response(500, {"error": "Erro interno: Formato de dados inválido da IA.", "details": str(e)})
        else:
            # Standard text processing actions
            processed_response = gemini_output_text
            if action == "KeyWords":
                # Clean up potential extra whitespace/newlines
                processed_response = " ".join(processed_response.replace("\n", " ").split())
            final_payload["response"] = processed_response
            logger.info(f"Resposta simples processada para '{action}'.")

        # Extract token usage metadata
        usage_metadata = result_json.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        candidates_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", 0)

        # Ensure totalTokenCount is populated if missing
        if total_tokens == 0 and (prompt_tokens > 0 or candidates_tokens > 0):
             total_tokens = prompt_tokens + candidates_tokens

        tokens_info = {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "totalTokenCount": total_tokens
        }
        final_payload["tokens"] = tokens_info
        logger.info(f"Token Usage: {json.dumps(tokens_info)}")

        logger.info(f"Action '{action}' completed successfully.")
        return create_proxy_response(200, final_payload)

    except (TimeoutError, ConnectionError) as api_err:
        logger.error(f"API Call Error: {api_err}")
        return create_proxy_response(504 if isinstance(api_err, TimeoutError) else 502, {"error": f"Erro de comunicação com a IA: {api_err}"})
    except ValueError as ve:
        # Distinguish between config errors and runtime errors
        if "Missing required environment variable" in str(ve) or "SSM Parameter" in str(ve):
             logger.critical(f"Configuration Error: {ve}", exc_info=True)
             return create_proxy_response(500, {"error": f"Erro de configuração do servidor: {ve}"})
        else:
             logger.error(f"Data validation or processing error: {ve}", exc_info=True)
             return create_proxy_response(400, {"error": f"Erro nos dados da requisição ou processamento: {ve}"})
    except Exception as e:
        logger.critical(f"Unhandled exception processing action '{action}': {e}", exc_info=True)
        return create_proxy_response(500, {"error": "Erro interno inesperado do servidor.", "details": str(e)})
